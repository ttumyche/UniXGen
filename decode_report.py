import os

import torch

from nltk.translate.bleu_score import corpus_bleu
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer('../BBPE_tokenizer/vocab.json', '../BBPE_tokenizer/merges.txt')
tokenizer.add_special_tokens(["[PAD]", "[SOS]", "[EOS]", "[SEP]", "[MASK]"])
tokenizer._tokenizer.post_processor = BertProcessing(
    ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ("[SOS]", tokenizer.token_to_id("[SOS]")),
)
tokenizer.enable_truncation(max_length=256)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=256)

model_path = {
    'path/to/saved_files'
}

for model_dir in model_path:
    GT_decoded_texts, gen_decoded_texts = [], []
    path = os.path.join(model_dir, f"test_output_*.pt")
    output = torch.load(path)
    for row in output:
        total_GT_text = row['GT_text'][0]
        total_gen_text = row['gen_text'][0]
        for gt_text_i, gen_text_i in zip(total_GT_text, total_gen_text):
            gt_text_i = gt_text_i.tolist()
            gen_text_i = gen_text_i.tolist()
            gt_decoded_text_i = tokenizer.decode(gt_text_i, skip_special_tokens=True)
            gen_decoded_text_i = tokenizer.decode(gen_text_i, skip_special_tokens=True)
            GT_decoded_texts.append(gt_decoded_text_i)
            gen_decoded_texts.append(gen_decoded_text_i)

    '''
    Save the decoded outputs ("GT_decoded_texts", "gen_decoded_texts") according to your preference.
    '''