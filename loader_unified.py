import os
import csv
import random
import pickle
from tqdm import tqdm
from collections import defaultdict

import albumentations
import albumentations.pytorch

import torch
from torch.utils.data import Dataset

from vae import VQGanVAE

random.seed(42)

class UnifiedCXRDataset(Dataset):

    def __init__(self,
                 metadata_file,
                 img_root_dir,
                 text_root_dir,
                 vqgan_model_path,
                 vqgan_config_path,
                 codebook_indices_path,
                 vqgan,
                 max_img_num,
                 max_text_len,
                 tokenizer,
                 target_count,
                 target_view,
                 under_sample="fixed"
                 ):
        super().__init__()

        assert max_img_num <= target_count, f'max_img_num({max_img_num}) should be less than target_count({target_count}).'

        self.under_sample = under_sample.split('_')[0]  # fixed
        self.select_studies = under_sample.split('_')[1]  # 'each' or 'all', 'all': using all groups (S w/1, w/2, w/3), 'each': using only selected single group
        self.training_mode = under_sample.split('_')[-1]  # unified

        self.dict_by_studyid = defaultdict(list)

        f = open(metadata_file, 'r')
        rdr = csv.reader(f)
        for i, line in tqdm(enumerate(rdr)):
            dicom_id, subject_id, study_id, ViewPosition, count = line  # [427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5,10046166,50051329,LATERAL,2]
            if self.select_studies == 'each':
                if (int(count) == int(target_count) and ViewPosition in target_view):
                    self.dict_by_studyid[study_id].append(line)
            elif self.select_studies == 'all':
                if (int(count) <= int(target_count) and ViewPosition in target_view):
                    self.dict_by_studyid[study_id].append(line)

        if self.select_studies == 'all':
            self.dict_by_studyid = {k: self.dict_by_studyid[k] for k in self.dict_by_studyid.keys() if len(self.dict_by_studyid[k]) == int(self.dict_by_studyid[k][0][-1])}
        elif self.select_studies == 'each':
            self.dict_by_studyid = {k: self.dict_by_studyid[k] for k in self.dict_by_studyid.keys() if len(self.dict_by_studyid[k]) == target_count}

        self.key_list = list(self.dict_by_studyid.keys())

        self.img_root_dir = img_root_dir
        self.text_root_dir = text_root_dir

        self.vae = VQGanVAE(vqgan_model_path, vqgan_config_path)

        if vqgan == 512:
            self.img_fmap_size = 32
            self.img_reso = 512  # eg. 256 or 512 in my case
            self.img_len = 1024 + 2  # eg. 32**2 = 1024
            self.img_vocab_size = self.vae.num_tokens  # eg. 1024

        else:
            NotImplemented

        with open(codebook_indices_path, 'rb') as f:
            self.indices_dict = pickle.load(f)

        # 2 of 3: max_img_num = 2, target_count = 3
        self.max_img_num = max_img_num
        self.target_count = target_count

        self.max_text_len = max_text_len

        self.tokenizer = tokenizer
        self.text_vocab_size = self.tokenizer.get_vocab_size()

        # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.img_reso)
        self.cropper = albumentations.CenterCrop(height=self.img_reso, width=self.img_reso)
        self.totensor = albumentations.pytorch.transforms.ToTensorV2()
        self.preprocessor = albumentations.Compose([
            self.rescaler,
            self.cropper,
        ])

        self.slots = []


        self.modes = ['txt']
        for i in range(self.max_img_num):
            y = [self.img_vocab_size + i] * (self.img_len)
            self.slots.extend(y)
            self.modes.append(f'img{i + 1}')

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        study_id = self.key_list[idx]

        if self.select_studies == 'each':
            assert len(self.dict_by_studyid[study_id]) == self.target_count, f'{study_id} has {len(self.dict_by_studyid[study_id])} data, but target_count is {self.target_count}.'
        elif self.select_studies == 'all':
            assert len(self.dict_by_studyid[study_id]) <= self.target_count, f'{study_id} has {len(self.dict_by_studyid[study_id])} data, but target_count is {self.target_count}.'

        if self.max_img_num == self.target_count:
            imgs_meta = self.dict_by_studyid[study_id]

        elif self.max_img_num < self.target_count:
            if self.under_sample == 'fixed':
                imgs_meta = self.dict_by_studyid[study_id][:self.max_img_num]
            elif self.under_sample == 'random':
                imgs_meta = random.sample(self.dict_by_studyid[study_id], self.max_img_num)

        if self.select_studies == 'all':
            num_img_in_study = int(self.dict_by_studyid[study_id][0][-1])
        elif self.select_studies == 'each':
            num_img_in_study = self.max_img_num


        # imgs
        img_paths = ''
        image_output = []
        view_position = []

        for i in range(num_img_in_study):
            dicom_id, subject_id, studyid, ViewPosition = imgs_meta[i][:4]
            img_path = os.path.join(self.img_root_dir, 'p' + subject_id[:2], 'p' + subject_id, 's' + studyid, dicom_id + '.jpg')
            image_indices = self.indices_dict[dicom_id].copy()  # indices list
            if ViewPosition == 'AP':
                image_indices.insert(0, 1025)  # self.tokenizer.token_to_id("[SOS1]")
                image_indices.append(1026)  # self.tokenizer.token_to_id("[EOS1]"
                image_output.append(torch.tensor(image_indices))
            elif ViewPosition == 'PA':
                image_indices.insert(0, 1027)  # self.tokenizer.token_to_id("[SOS2]")
                image_indices.append(1028)  # self.tokenizer.token_to_id("[EOS2]")
                image_output.append(torch.tensor(image_indices))
            elif ViewPosition == 'LATERAL':
                image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
                image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
                image_output.append(torch.tensor(image_indices))
            elif ViewPosition == 'LL':
                image_indices.insert(0, 1029)  # self.tokenizer.token_to_id("[SOS3]")
                image_indices.append(1030)  # self.tokenizer.token_to_id("[EOS3]")
                image_output.append(torch.tensor(image_indices))
            else:
                raise ValueError
            img_paths += (img_path + '|')
            view_position.append(ViewPosition)

        # PAD imgs
        if num_img_in_study < self.max_img_num:
            assert self.select_studies == 'all'
            for i in range(self.max_img_num - num_img_in_study):
                image_indices = [1024] * self.img_len
                image_output.append(torch.tensor(image_indices))
                img_paths += ('PAD' + '|')
                view_position.append('PAD')

            self.modes = ['txt']
            for i in range(num_img_in_study):
                self.modes.append(f'img{i + 1}')
            random.shuffle(self.modes)
            for i in range(num_img_in_study, self.max_img_num):
                self.modes.append(f'img{i + 1}')
        else:
            random.shuffle(self.modes)

        # report
        text_path = os.path.join(self.text_root_dir, 's' + study_id + '.txt')
        with open(text_path, 'r') as f:
            data = f.read()
        src = data.replace('  ', ' ').replace('  ', ' ').lower()
        ids_list = self.tokenizer.encode(src).ids
        text_output = torch.tensor(ids_list)


        outputs = {'txt': text_output, 'modes': self.modes, 'study_id': study_id,
                   'img_paths': img_paths, 'view_position': view_position}

        for i in range(self.max_img_num):
            outputs[f'img{i+1}'] = image_output[i]
        return outputs
