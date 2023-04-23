import os
import csv
import math
import time
import random
import numpy as np

import torch
from torch.optim import AdamW
from torch.nn.functional import cross_entropy

import pytorch_lightning as pl

from nltk.translate.bleu_score import corpus_bleu
from transformer_pytorch.transformer_unified import TransformerLM_unified
from transformers.optimization import get_cosine_schedule_with_warmup

random.seed(42)

class TransformerLightning_unified(pl.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=0.01,
                 pad_token_idx=0, sos_token_idx=1, eos_token_idx=2,
                 save_dir="", causal_trans='conditioned_causal', **kargs):
        super().__init__()
        self.kargs = kargs
        self.max_img_num = kargs['max_img_num']
        self.under_sample = kargs['under_sample']
        self.attn_type = kargs['attn_type']
        self.num_txt_tokens = kargs['num_tokens']
        self.num_img_tokens = kargs['num_img_tokens']

        self.ckpt_path = None
        self.target_count = None
        self.test_meta_file_name = None

        self.transformerLM_unified = TransformerLM_unified(**kargs)
        self.weight_decay = weight_decay
        self.lr = lr
        self.pad_token_idx = pad_token_idx
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.save_dir = save_dir
        self.causal = causal_trans

        self.save_hyperparameters(ignore=['tokenizer'])

    def forward(self, batch):
        logit = self.transformerLM_unified(batch, causal=self.causal)
        return logit

    def training_step(self, batch, batch_idx):
        img1, txt, modes, view = batch['img1'], batch['txt'], batch['modes'], batch['view_position']

        assert txt.shape[0] == img1.shape[0]
        batch_size = txt.shape[0]
        txt_seq_len = txt.shape[1]
        img_seq_len = img1.shape[1]
        n = img_seq_len + txt_seq_len
        if 'img2' in batch.keys():
            img2 = batch['img2']
            n += img2.shape[1]
        if 'img3' in batch.keys():
            img3 = batch['img3']
            n += img3.shape[1]

        logit = self(batch)[:, :-1, :]
        max_neg_value = -torch.finfo(logit.dtype).max

        for bsz in range(batch_size):
            if np.array(modes)[:, bsz][0] == 'txt':
                first_modal = txt_seq_len - 1
                logit[bsz, :first_modal, self.num_txt_tokens:] = max_neg_value
                logit[bsz, first_modal:, :self.num_txt_tokens] = max_neg_value
            else:
                first_modal = img_seq_len - 1
                if np.array(modes)[:, bsz][1] == 'txt':
                    logit[bsz, :first_modal, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, first_modal: (first_modal + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, (first_modal + txt_seq_len):, :self.num_txt_tokens] = max_neg_value
                elif np.array(modes)[:, bsz][-1] == 'txt':
                    logit[bsz, :-txt_seq_len, :self.num_txt_tokens] = max_neg_value
                    logit[bsz, -txt_seq_len:, self.num_txt_tokens:] = max_neg_value
                if 'img3' in batch.keys() and np.array(modes)[:, bsz][2] == 'txt':  # [i, i, t, i]
                    logit[bsz, :(first_modal + img_seq_len), :self.num_txt_tokens] = max_neg_value
                    logit[bsz, (first_modal + img_seq_len):(first_modal + img_seq_len + txt_seq_len), self.num_txt_tokens:] = max_neg_value
                    logit[bsz, -img_seq_len:, :self.num_txt_tokens] = max_neg_value

        logit = logit.reshape(-1, logit.size(-1))

        target_lst = []
        for bsz in range(batch_size):
            for idx, mode in enumerate(np.array(modes)[:, bsz]):
                if idx == 0:
                    target = batch[mode][bsz, 1:]
                else:
                    target = batch[mode][bsz]
                if mode.startswith('img'):
                    target_lst.append(target + self.num_txt_tokens)
                else:
                    target_lst.append(target)
        target = torch.cat(target_lst, dim=0)

        ignore_classes = torch.ones(self.num_txt_tokens + self.num_img_tokens)
        ignore_classes[1024 + self.num_txt_tokens] = 0.
        loss = cross_entropy(logit, target, ignore_index=self.pad_token_idx, weight=ignore_classes.to(logit.device))

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

        output = {
            'batch_idx': batch_idx,
            'loss': loss
        }
        return output

    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        img_paths, study_ids = batch['img_paths'], batch['study_id']
        img1, txt, view = batch['img1'], batch['txt'], batch['view_position']
        n = img1.shape[1] + txt.shape[1]
        self.transformerLM_unified.max_img_num = self.max_img_num

        if self.max_img_num == 1:
            modes_txt = [['img1'], ['txt']]
            modes_img1 = [['txt'], ['img1']]

        elif self.max_img_num == 2:
            n += batch['img2'].shape[1]

            modes_txt = random.sample([[['img1'], ['img2'], ['txt']], [['img2'], ['img1'], ['txt']]], 1)[0]
            modes_img1 = random.sample([[['img2'], ['txt'], ['img1']], [['txt'], ['img2'], ['img1']]], 1)[0]
            modes_img2 = random.sample([[['img1'], ['txt'], ['img2']], [['txt'], ['img1'], ['img2']]], 1)[0]

        elif self.max_img_num == 3:
            n += (batch['img2'].shape[1] + batch['img3'].shape[1])

            modes_txt = random.sample([['img1'], ['img2'], ['img3']], 3)
            modes_txt.append(['txt'])
            modes_img1 = random.sample([['txt'], ['img2'], ['img3']], 3)
            modes_img1.append(['img1'])
            modes_img2 = random.sample([['img1'], ['txt'], ['img3']], 3)
            modes_img2.append(['img2'])
            modes_img3 = random.sample([['img1'], ['img2'], ['txt']], 3)
            modes_img3.append(['img3'])

        # generate texts
        batch['modes'] = modes_txt

        gen_texts = self.transformerLM_unified.generate_texts(
            batch,
            sos_token_idx=self.sos_token_idx,
            eos_token_idx=self.eos_token_idx,
            pad_token_idx=self.pad_token_idx,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )

        # generate img1
        batch['modes'] = modes_img1

        gen_images1 = self.transformerLM_unified.generate_image(
            batch,
            filter_logits_fn='top_p',
            filter_thres=0.9,
            temperature=0.7,
            causal=self.causal
        )


        if 'img2' in batch.keys():
            # generate img2
            batch['modes'] = modes_img2

            gen_images2 = self.transformerLM_unified.generate_image(
                batch,
                filter_logits_fn='top_p',
                filter_thres=0.9,
                temperature=0.7,
                causal=self.causal
            )

        if 'img3' in batch.keys():
            # generate img3
            batch['modes'] = modes_img3

            gen_images3 = self.transformerLM_unified.generate_image(
                batch,
                filter_logits_fn='top_p',
                filter_thres=0.9,
                temperature=0.7,
                causal=self.causal
            )

        output = {
            'GT_text': txt,
            'gen_text': gen_texts,
            'GT_image1': img1,
            'gen_image1': gen_images1,
            'img_paths': img_paths,
            'modes_txt': modes_txt,
            'modes_img1': modes_img1,
            'view': view,
        }

        if 'img2' in batch.keys():
            output['GT_image2'] = batch['img2']
            output['gen_image2'] = gen_images2
            output['modes_img2'] = modes_img2

        if 'img3' in batch.keys():
            output['GT_image3'] = batch['img3']
            output['gen_image3'] = gen_images3
            output['modes_img3'] = modes_img3

        return output


    def test_epoch_end(self, test_step_outputs):
        from tokenizers import ByteLevelBPETokenizer
        from tokenizers.processors import BertProcessing
        tokenizer = ByteLevelBPETokenizer('BBPE_tokenizer/vocab.json', 'BBPE_tokenizer/merges.txt')
        tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
        )
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=256)


        gathered_test_step_outputs = self.all_gather(test_step_outputs)

        img_paths = gathered_test_step_outputs[0]['img_paths']
        if self.max_img_num != -1:
            max_text_len = gathered_test_step_outputs[0]['GT_text'].size(-1)
            total_GT_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])
            total_gen_text = torch.empty(0, max_text_len).type_as(gathered_test_step_outputs[0]['GT_text'])

            for i, out in enumerate(gathered_test_step_outputs):
                GT_text = out['GT_text'].reshape(-1, max_text_len)
                gen_text = out['gen_text'].reshape(-1, max_text_len)
                total_GT_text = torch.cat((total_GT_text, GT_text), dim=0)
                total_gen_text = torch.cat((total_gen_text, gen_text), dim=0)

        if self.global_rank == 0:
            if self.max_img_num != -1:
                torch.save(gathered_test_step_outputs, os.path.join(self.save_dir, f"test_output_{self.ckpt_path.split('/')[-1].split('-')[0]}_{str(self.max_img_num)}_of_{str(self.target_count)}_{self.test_meta_file_name}.pt"))
                # !# For generated texts
                GT_decoded_texts, gen_decoded_texts = [], []
                for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
                    gt_text_i = gt_text_i.tolist()
                    gen_text_i = gen_text_i.tolist()
                    gt_decoded_text_i = tokenizer.decode(gt_text_i, skip_special_tokens=True)
                    gen_decoded_text_i = tokenizer.decode(gen_text_i, skip_special_tokens=True)
                    GT_decoded_texts.append(gt_decoded_text_i)
                    gen_decoded_texts.append(gen_decoded_text_i)
                # calculate BLEU
                references = []
                candidates = []
                for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                    reference = [gt_decoded_text_i.split(' ')]
                    candidate = gen_decoded_text_i.split(' ')
                    references.append(reference)
                    candidates.append(candidate)

                bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
                bleu2 = corpus_bleu(references, candidates, weights=(1 / 2, 1 / 2, 0, 0))
                bleu3 = corpus_bleu(references, candidates, weights=(1 / 3, 1 / 3, 1 / 3, 0))
                bleu4 = corpus_bleu(references, candidates, weights=(1 / 4, 1 / 4, 1 / 4, 1 / 4))
                print(f'Cumulative 1-gram: {bleu1:.3f}')
                print(f'Cumulative 2-gram: {bleu2:.3f}')
                print(f'Cumulative 3-gram: {bleu3:.3f}')
                print(f'Cumulative 4-gram: {bleu4:.3f}')
                self.log("test_BLEU-1", bleu1)
                self.log("test_BLEU-2", bleu2)
                self.log("test_BLEU-3", bleu3)
                self.log("test_BLEU-4", bleu4)
                # save csv files for labeler
                GT_REPORTS_PATH = os.path.join(self.save_dir, 'GT_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                    round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
                GEN_REPORTS_PATH = os.path.join(self.save_dir, 'GEN_reports_test_' + str(round(bleu1, 3)) + '_' + str(
                    round(bleu2, 3)) + '_' + str(round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
                IMG_PATHS = os.path.join(self.save_dir, 'IMG_paths_test_' + str(round(bleu1, 3)) + '_' + str(round(bleu2, 3)) + '_' + str(
                                             round(bleu3, 3)) + '_' + str(round(bleu4, 3)) + '_' + self.ckpt_path.split('/')[-1].split('-')[0] + '_' + str(self.max_img_num) + '_of_' + str(self.target_count) + self.test_meta_file_name + '.csv')
                f_gt = open(GT_REPORTS_PATH, 'a')
                wr_gt = csv.writer(f_gt)
                f_gen = open(GEN_REPORTS_PATH, 'a')
                wr_gen = csv.writer(f_gen)
                f_img = open(IMG_PATHS, 'a')
                wr_img = csv.writer(f_img)
                for gt_decoded_text_i, gen_decoded_text_i in zip(GT_decoded_texts, gen_decoded_texts):
                    wr_gt.writerow([gt_decoded_text_i])
                    wr_gen.writerow([gen_decoded_text_i])
                for img_paths_i in img_paths:
                    wr_img.writerow([img_paths_i])
                f_gt.close()
                f_gen.close()
                f_img.close()
                print("GEN_reports_test saved.")
                print(f'\n\n')

        time.sleep(0.5)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        train_loader = self.train_dataloader()
        scheduler = {
            'scheduler':
                get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=self.kargs['epochs'] * len(train_loader)),
            'interval': 'step',
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}