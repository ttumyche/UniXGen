import os
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from vae import VQGanVAE
from helpers import str2bool

models = {
    'path/to/saved_files'
}

for infer_path in models:
    output_path = glob(os.path.join(infer_path, 'test_output_*.pt'))[:1]
    for output_pt_file in output_path:
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_save', default=True, type=str2bool, help='')
        parser.add_argument('--save_dir', default='/path/to/decoded_imgs', type=str, help='')
        parser.add_argument('--infer_num', default=str(32), type=str, help='infer num when load eval ckpt')
        parser.add_argument('--vqgan_model_path', default='mimiccxr_vqgan/last.ckpt', type=str)
        parser.add_argument('--vqgan_config_path', default='mimiccxr_vqgan/2021-12-17T08-58-54-project.yaml', type=str)

        args = parser.parse_args()

        if args.img_save:
            os.makedirs(args.save_dir, exist_ok=True)

        vae = VQGanVAE(args.vqgan_model_path, args.vqgan_config_path).cuda()
        output = torch.load(output_pt_file)
        for i, row in tqdm(enumerate(output), desc='len of file'):
            max_img_num = 0
            bsz = len(row['img_paths'])

            for k in row.keys():
                if k.startswith('GT_image'):
                    max_img_num += 1

            for b in tqdm(range(bsz), desc='bsz'):
                name_paths = row['img_paths'][b].split('|')[0].split('/')
                name = name_paths[-4] + "_" + name_paths[-3] + "_" + name_paths[-2]

                for i in range(1, max_img_num+1):
                    ngpus, bsz, num_codes = row[f'GT_image{i}'].size()

                    GT_tensor = row[f'GT_image{i}'].reshape(-1, num_codes)[b][1:-1].unsqueeze(0)
                    gen_tensor = row[f'gen_image{i}'].reshape(-1, num_codes)[b][1:-1].unsqueeze(0)

                    GT_img1 = vae.decode(GT_tensor)
                    torch.cuda.empty_cache()
                    gen_img1 = vae.decode(gen_tensor)
                    torch.cuda.empty_cache()

                    if args.img_save:
                        GT_img1 = GT_img1[0].permute(1, 2, 0).detach().cpu().numpy()
                        gen_img1 = gen_img1[0].permute(1, 2, 0).detach().cpu().numpy()
                        plt.imsave(os.path.join(args.save_dir, name + f'_gen_img{i}.jpeg'), gen_img1)
                        plt.imsave(os.path.join(args.save_dir, name + f'_GT_img{i}.jpeg'), GT_img1)
                    del GT_img1
                    del gen_img1
                    torch.cuda.empty_cache()

        if args.img_save:
            print("\n >>> image saving done!")