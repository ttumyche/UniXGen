import math
import numpy as np
from functools import partial

from transformer_pytorch.model_utils import *
from axial_positional_embedding import AxialPositionalEmbedding
from transformer_pytorch.FAVOR_unified import FAVORAttention, ProjectionUpdater

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None):
        super().__init__()

        activation = default(activation, nn.GELU)

        self.w1 = nn.Linear(dim, mult * dim)
        self.act = activation()
        self.w2 = nn.Linear(mult * dim, dim)
        self.dropout = dropout

    def forward(self, x, **kwargs):
        out = self.w1(x)
        out = self.act(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.w2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            local_attn_heads=0,
            causal='conditioned_causal',
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            ff_mult=4,
            nb_features=None,
            feature_redraw_interval=1000,
            use_scalenorm=False,
            use_rezero=False,
            ff_dropout=0.,
            attn_dropout=0.,
            cross_attend=False,
            auto_check_redraw=True,
            qkv_bias=True,
            attn_out_bias=True,
            no_projection=False,
            FAVOR=False,
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)  # eg. dim=512

        if FAVOR:
            for _, local_heads in zip(range(depth), local_attn_heads):
                layers.append(nn.ModuleList([
                    wrapper_fn(FAVORAttention(dim=dim, causal=causal, attn_type=attn_type,
                                              generalized_attention=generalized_attention,
                                              kernel_fn=kernel_fn,
                                              heads=heads, local_heads=local_heads, nb_features=nb_features,
                                              dropout=attn_dropout, no_projection=no_projection,
                                              qkv_bias=qkv_bias, attn_out_bias=attn_out_bias)),
                    wrapper_fn(PositionWiseFeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, activation=None))
                ]))
                if not cross_attend:
                    continue

        execute_type = SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)  # len(): 2*depth if cross_attend else depth
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'pad_mask': route_attn, 'pos_emb': route_attn, 'causal': route_attn, 'condition_len': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route={**attn_route_map, **context_route_map})

        if FAVOR:
            self.auto_check_redraw = auto_check_redraw  # auto_check_redraw = True
            self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)
        else:
            self.auto_check_redraw = False

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)


class TransformerLM_unified(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,  # text vocab size
            num_img_tokens,  # img vocab size + num img pad
            img_vocab_size,
            max_seq_len,  # total max len; img_len * max_img_num + max_text_len
            max_img_len,
            max_img_num,  # num img slot
            img_len,
            dim,
            depth,
            heads=8,
            local_attn_heads=0,
            causal='conditioned_causal',
            attn_type="conditioned_noncuda",
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            ff_mult=4,
            nb_features=None,
            feature_redraw_interval=1000,
            reversible=False,
            emb_dropout=0.,
            ff_dropout=0.,
            attn_dropout=0.,
            use_scalenorm=False,
            use_rezero=False,
            cross_attend=False,
            no_projection=False,
            tie_embed=False,
            rotary_position_emb=True,
            axial_position_emb=False,
            axial_position_shape=None,
            auto_check_redraw=True,
            qkv_bias=False,
            attn_out_bias=False,
            img_fmap_size=0,
            FAVOR=False,

            mask_prob=0.15,
            replace_prob=0.9,
            random_token_prob=0.,
            mask_token_id=4,
            pad_token_id=0,
            mask_ignore_token_ids=[],
            **kwargs
    ):
        super().__init__()

        self.img_len = img_len
        self.num_txt_tokens = num_tokens
        self.num_img_tokens = num_img_tokens
        self.img_vocab_size = img_vocab_size
        self.max_seq_len = max_seq_len
        self.max_img_num = max_img_num
        local_attn_heads = cast_tuple(local_attn_heads)
        self.dim = dim
        dim_head = dim // heads

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_token_prob = random_token_prob
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = mask_ignore_token_ids

        self.attn_type = attn_type

        # !# img
        self.image_token_emb = nn.Embedding(num_img_tokens, dim)
        self.ap_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)  # max_seq_len = img_len * max_img_num + max_text_len
        self.pa_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.la_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.pad_img_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.image_pos_emb = AxialPositionalEmbedding(dim=dim, axial_shape=(img_fmap_size + 1, img_fmap_size + 1))

        # !# text
        self.token_emb = nn.Embedding(num_tokens, dim)
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(max_seq_len / 64), 64))
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)
        self.txt_att_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, local_attn_heads, causal,
                                       attn_type, generalized_attention,
                                       kernel_fn,
                                       ff_mult, nb_features, feature_redraw_interval, use_scalenorm, use_rezero,
                                       ff_dropout, attn_dropout, cross_attend, auto_check_redraw,
                                       qkv_bias, attn_out_bias, no_projection, FAVOR)
        self.norm = nn.LayerNorm(dim)

        self.to_out_txt = nn.Linear(dim, num_tokens)  # if not tie_embed else None
        self.to_out_img = nn.Linear(dim, num_img_tokens)  # if not tie_embed else None
        self.to_out_combined_txt_img = nn.Linear(dim, (num_tokens + num_img_tokens))

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, batch, causal, return_encodings=False, **kwargs):  # kwargs = {'mask': tensor with same shape x}
        img1, txt, modes, view = batch['img1'], batch['txt'], batch['modes'], batch['view_position']
        b, n_img1, device = *img1.shape, img1.device
        b, n_txt, device = *txt.shape, txt.device
        n = n_img1 + n_txt

        imgs = [img1]
        if 'img2' in batch.keys():
            assert self.max_img_num >= 2
            img2 = batch['img2']
            b, n_img2, device = *img2.shape, img2.device
            n += n_img2
            imgs.append(img2)
        if 'img3' in batch.keys():
            assert self.max_img_num == 3
            img3 = batch['img3']
            b, n_img3, device = *img3.shape, img3.device
            n += n_img3
            imgs.append(img3)

        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # !# image; token and positional embeddings
        x_images = []
        for i, x_img_slot in enumerate(imgs):
            outs = []
            x_img = self.image_token_emb(x_img_slot)  # [b, img_len] --> [b, img_len, dim]

            for bsz in range(b):
                slot_view = view[i][bsz]
                if slot_view == 'AP':
                    out = self.ap_att_emb(x_img[bsz:bsz + 1])  # out: [img_len, dim]
                elif slot_view == 'PA':
                    out = self.pa_att_emb(x_img[bsz:bsz + 1])  # out: [img_len, dim]
                elif slot_view == 'LATERAL':
                    out = self.la_att_emb(x_img[bsz:bsz + 1])  # out: [img_len, dim]
                elif slot_view == 'LL':
                    out = self.la_att_emb(x_img[bsz:bsz + 1])  # out: [img_len, dim]
                elif slot_view == 'PAD':
                    out = self.pad_img_att_emb(x_img[bsz:bsz + 1])  # out: [img_len, dim]
                else:
                    raise ValueError
                out = torch.unsqueeze(out, dim=0)  # out: [1, img_len, dim]
                outs.append(out)
            att = torch.cat(outs, dim=0)  # -> [b, img_len, dim]

            pos_out = self.image_pos_emb(x_img)  # out: [B, img_len, dim]

            x_images.append(x_img + att + pos_out)

        # !# text; token and positional embeddings
        x_text = self.token_emb(txt)
        x_text += self.txt_att_emb(x_text)
        x_text += self.pos_emb(x_text)

        if len([1 for m in modes if len(set(m)) == 1]) == len(modes):
            # FIx
            if modes[-1][0] == 'txt':  # p(txt|imgs)
                n_condition = n - n_txt
                if self.max_img_num == 1:
                    assert modes[0][0] == 'img1'
                    x = torch.cat((x_images[0], x_text), dim=1)

                elif self.max_img_num == 2:
                    if modes[0][0] == 'img1':
                        x = torch.cat((x_images[0], x_images[1], x_text), dim=1)
                    elif modes[0][0] == 'img2':
                        x = torch.cat((x_images[1], x_images[0], x_text), dim=1)
                elif self.max_img_num == 3:
                    if modes[0][0] == 'img1':
                        if modes[1][0] == 'img2':
                            x = torch.cat((x_images[0], x_images[1], x_images[2], x_text), dim=1)
                        else:
                            x = torch.cat((x_images[0], x_images[2], x_images[1], x_text), dim=1)
                    elif modes[0][0] == 'img2':
                        if modes[1][0] == 'img1':
                            x = torch.cat((x_images[1], x_images[0], x_images[2], x_text), dim=1)
                        else:
                            x = torch.cat((x_images[1], x_images[2], x_images[0], x_text), dim=1)
                    elif modes[0][0] == 'img3':
                        if modes[1][0] == 'img1':
                            x = torch.cat((x_images[2], x_images[0], x_images[1], x_text), dim=1)
                        else:
                            x = torch.cat((x_images[2], x_images[1], x_images[0], x_text), dim=1)

                elif self.max_img_num == 0:
                    n_condition = 0
                    x = x_text

            elif modes[-1][0] == 'img3':  # p(img3|img1, img2, txt)
                n_condition = n - n_img3
                assert self.max_img_num == 3 or self.max_img_num == -1
                if modes[0][0] == 'img1':
                    if modes[1][0] == 'img2':
                        x = torch.cat((x_images[0], x_images[1], x_text, x_images[2]), dim=1)
                    else:
                        x = torch.cat((x_images[0], x_text, x_images[1], x_images[2]), dim=1)
                elif modes[0][0] == 'img2':
                    if modes[1][0] == 'img1':
                        x = torch.cat((x_images[1], x_images[0], x_text, x_images[2]), dim=1)
                    else:
                        x = torch.cat((x_images[1], x_text, x_images[0], x_images[2]), dim=1)
                elif modes[0][0] == 'txt':
                    if modes[1][0] == 'img1':
                        x = torch.cat((x_text, x_images[0], x_images[1], x_images[2]), dim=1)
                    else:
                        x = torch.cat((x_text, x_images[1], x_images[0], x_images[2]), dim=1)

            elif modes[-1][0] == 'img2':
                n_condition = n - n_img2
                assert self.max_img_num >= 2 or self.max_img_num == -1
                if self.max_img_num == 2:
                    if modes[0][0] == 'img1':
                        x = torch.cat((x_images[0], x_text, x_images[1]), dim=1)
                    elif modes[0][0] == 'txt':
                        x = torch.cat((x_text, x_images[0], x_images[1]), dim=1)

                elif self.max_img_num == 3:
                    if modes[0][0] == 'img1':
                        if modes[1][0] == 'img3':
                            x = torch.cat((x_images[0], x_images[2], x_text, x_images[1]), dim=1)
                        else:
                            x = torch.cat((x_images[0], x_text, x_images[2], x_images[1]), dim=1)
                    elif modes[0][0] == 'img3':
                        if modes[1][0] == 'img1':
                            x = torch.cat((x_images[2], x_images[0], x_text, x_images[1]), dim=1)
                        else:
                            x = torch.cat((x_images[2], x_text, x_images[0], x_images[1]), dim=1)
                    elif modes[0][0] == 'txt':
                        if modes[1][0] == 'img1':
                            x = torch.cat((x_text, x_images[0], x_images[2], x_images[1]), dim=1)
                        else:
                            x = torch.cat((x_text, x_images[2], x_images[0], x_images[1]), dim=1)

                elif self.max_img_num == -1:
                    if len(set(sum(modes, []))) == 4:  # txt, 3
                        if modes[0][0] == 'img1':
                            if modes[1][0] == 'img3':
                                x = torch.cat((x_images[0], x_images[2], x_text, x_images[1]), dim=1)
                            else:
                                x = torch.cat((x_images[0], x_text, x_images[2], x_images[1]), dim=1)
                        elif modes[0][0] == 'img3':
                            if modes[1][0] == 'img1':
                                x = torch.cat((x_images[2], x_images[0], x_text, x_images[1]), dim=1)
                            else:
                                x = torch.cat((x_images[2], x_text, x_images[0], x_images[1]), dim=1)
                        elif modes[0][0] == 'txt':
                            if modes[1][0] == 'img1':
                                x = torch.cat((x_text, x_images[0], x_images[2], x_images[1]), dim=1)
                            else:
                                x = torch.cat((x_text, x_images[2], x_images[0], x_images[1]), dim=1)
                    elif len(set(sum(modes, []))) == 3:  # txt, 2
                        if modes[0][0] == 'img1':
                            x = torch.cat((x_images[0], x_text, x_images[1]), dim=1)
                        elif modes[0][0] == 'txt':
                            x = torch.cat((x_text, x_images[0], x_images[1]), dim=1)

                else:
                    ValueError

            elif modes[-1][0] == 'img1':
                n_condition = n - n_img1
                if self.max_img_num == 1:
                    x = torch.cat((x_text, x_images[0]), dim=1)

                elif self.max_img_num == 2:
                    if modes[0][0] == 'img2':
                        x = torch.cat((x_images[1], x_text, x_images[0]), dim=1)
                    elif modes[0][0] == 'txt':
                        x = torch.cat((x_text, x_images[1], x_images[0]), dim=1)
                elif self.max_img_num == 3:
                    if modes[0][0] == 'img2':
                        if modes[1][0] == 'img3':
                            x = torch.cat((x_images[1], x_images[2], x_text, x_images[0]), dim=1)
                        else:
                            x = torch.cat((x_images[1], x_text, x_images[2], x_images[0]), dim=1)
                    elif modes[0][0] == 'img3':
                        if modes[1][0] == 'img2':
                            x = torch.cat((x_images[2], x_images[1], x_text, x_images[0]), dim=1)
                        else:
                            x = torch.cat((x_images[2], x_text, x_images[1], x_images[0]), dim=1)
                    elif modes[0][0] == 'txt':
                        if modes[1][0] == 'img2':
                            x = torch.cat((x_text, x_images[1], x_images[2], x_images[0]), dim=1)
                        else:
                            x = torch.cat((x_text, x_images[2], x_images[1], x_images[0]), dim=1)

                elif self.max_img_num == 0:
                    n_condition = 0
                    x = x_images[0]

                elif self.max_img_num == -1:
                    if 'img3' in np.array(modes):
                        if modes[0][0] == 'img2':
                            if modes[1][0] == 'img3':
                                x = torch.cat((x_images[1], x_images[2], x_text, x_images[0]), dim=1)
                            else:
                                x = torch.cat((x_images[1], x_text, x_images[2], x_images[0]), dim=1)
                        elif modes[0][0] == 'img3':
                            if modes[1][0] == 'img2':
                                x = torch.cat((x_images[2], x_images[1], x_text, x_images[0]), dim=1)
                            else:
                                x = torch.cat((x_images[2], x_text, x_images[1], x_images[0]), dim=1)
                        elif modes[0][0] == 'txt':
                            if modes[1][0] == 'img2':
                                x = torch.cat((x_text, x_images[1], x_images[2], x_images[0]), dim=1)
                            else:
                                x = torch.cat((x_text, x_images[2], x_images[1], x_images[0]), dim=1)
                    elif 'img2' in np.array(modes):
                        if modes[0][0] == 'img2':
                            x = torch.cat((x_images[1], x_text, x_images[0]), dim=1)
                        elif modes[0][0] == 'txt':
                            x = torch.cat((x_text, x_images[1], x_images[0]), dim=1)
                    elif 'img1' in np.array(modes):
                        x = torch.cat((x_text, x_images[0]), dim=1)
            else:
                print(modes[-1][0])
                raise ValueError
        else:
            x = []
            for bsz in range(b):
                if np.array(modes)[:, bsz][-1] == 'txt':
                    n_condition = n - n_txt
                    if self.max_img_num == 1:
                        assert np.array(modes)[:, bsz][0] == 'img1'
                        x.append(torch.cat((x_images[0][bsz], x_text[bsz]), dim=0).unsqueeze(0))

                    elif self.max_img_num == 2:
                        if np.array(modes)[:, bsz][0] == 'img1':
                            x.append(torch.cat((x_images[0][bsz], x_images[1][bsz], x_text[bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'img2':
                            x.append(torch.cat((x_images[1][bsz], x_images[0][bsz], x_text[bsz]), dim=0).unsqueeze(0))

                    elif self.max_img_num == 3:
                        if np.array(modes)[:, bsz][0] == 'img1':
                            if np.array(modes)[:, bsz][1] == 'img2':
                                x.append(torch.cat((x_images[0][bsz], x_images[1][bsz], x_images[2][bsz], x_text[bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_images[0][bsz], x_images[2][bsz], x_images[1][bsz], x_text[bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'img2':
                            if np.array(modes)[:, bsz][1] == 'img1':
                                x.append(torch.cat((x_images[1][bsz], x_images[0][bsz], x_images[2][bsz], x_text[bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_images[1][bsz], x_images[2][bsz], x_images[0][bsz], x_text[bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'img3':
                            if np.array(modes)[:, bsz][1] == 'img1':
                                x.append(torch.cat((x_images[2][bsz], x_images[0][bsz], x_images[1][bsz], x_text[bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_images[2][bsz], x_images[1][bsz], x_images[0][bsz], x_text[bsz]), dim=0).unsqueeze(0))

                elif np.array(modes)[:, bsz][-1] == 'img3':  # p(img3|img1, img2, txt)
                    n_condition = n - n_img3
                    assert self.max_img_num == 3 or self.max_img_num == -1
                    if np.array(modes)[:, bsz][0] == 'img1':
                        if np.array(modes)[:, bsz][1] == 'img2':
                            x.append(torch.cat((x_images[0][bsz], x_images[1][bsz], x_text[bsz], x_images[2][bsz]), dim=0).unsqueeze(0))
                        else:
                            x.append(torch.cat((x_images[0][bsz], x_text[bsz], x_images[1][bsz], x_images[2][bsz]), dim=0).unsqueeze(0))
                    elif np.array(modes)[:, bsz][0] == 'img2':
                        if np.array(modes)[:, bsz][1] == 'img1':
                            x.append(torch.cat((x_images[1][bsz], x_images[0][bsz], x_text[bsz], x_images[2][bsz]), dim=0).unsqueeze(0))
                        else:
                            x.append(torch.cat((x_images[1][bsz], x_text[bsz], x_images[0][bsz], x_images[2][bsz]), dim=0).unsqueeze(0))
                    elif np.array(modes)[:, bsz][0] == 'txt':
                        if np.array(modes)[:, bsz][1] == 'img1':
                            x.append(torch.cat((x_text[bsz], x_images[0][bsz], x_images[1][bsz], x_images[2][bsz]), dim=0).unsqueeze(0))
                        else:
                            x.append(torch.cat((x_text[bsz], x_images[1][bsz], x_images[0][bsz], x_images[2][bsz]), dim=0).unsqueeze(0))

                elif np.array(modes)[:, bsz][-1] == 'img2':
                    n_condition = n - n_img2
                    assert self.max_img_num >= 2 or self.max_img_num == -1
                    if self.max_img_num == 2:
                        if np.array(modes)[:, bsz][0] == 'img1':
                            x.append(torch.cat((x_images[0][bsz], x_text[bsz], x_images[1][bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'txt':
                            x.append(torch.cat((x_text[bsz], x_images[0][bsz], x_images[1][bsz]), dim=0).unsqueeze(0))

                    elif self.max_img_num == 3:
                        if np.array(modes)[:, bsz][0] == 'img1':
                            if np.array(modes)[:, bsz][1] == 'img3':
                                x.append(torch.cat((x_images[0][bsz], x_images[2][bsz], x_text[bsz], x_images[1][bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_images[0][bsz], x_text[bsz], x_images[2][bsz], x_images[1][bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'img3':
                            if np.array(modes)[:, bsz][1] == 'img1':
                                x.append(torch.cat((x_images[2][bsz], x_images[0][bsz], x_text[bsz], x_images[1][bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_images[2][bsz], x_text[bsz], x_images[0][bsz], x_images[1][bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'txt':
                            if np.array(modes)[:, bsz][1] == 'img1':
                                x.append(torch.cat((x_text[bsz], x_images[0][bsz], x_images[2][bsz], x_images[1][bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_text[bsz], x_images[2][bsz], x_images[0][bsz], x_images[1][bsz]), dim=0).unsqueeze(0))
                    else:
                        ValueError

                elif np.array(modes)[:, bsz][-1] == 'img1':
                    n_condition = n - n_img1
                    if self.max_img_num == 1:
                        x.append(torch.cat((x_text[bsz], x_images[0][bsz]), dim=0).unsqueeze(0))

                    elif self.max_img_num == 2:
                        if np.array(modes)[:, bsz][0] == 'img2':
                            x.append(torch.cat((x_images[1][bsz], x_text[bsz], x_images[0][bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'txt':
                            x.append(torch.cat((x_text[bsz], x_images[1][bsz], x_images[0][bsz]), dim=0).unsqueeze(0))
                    elif self.max_img_num == 3:
                        if np.array(modes)[:, bsz][0] == 'img2':
                            if np.array(modes)[:, bsz][1] == 'img3':
                                x.append(torch.cat((x_images[1][bsz], x_images[2][bsz], x_text[bsz], x_images[0][bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_images[1][bsz], x_text[bsz], x_images[2][bsz], x_images[0][bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'img3':
                            if np.array(modes)[:, bsz][1] == 'img2':
                                x.append(torch.cat((x_images[2][bsz], x_images[1][bsz], x_text[bsz], x_images[0][bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_images[2][bsz], x_text[bsz], x_images[1][bsz], x_images[0][bsz]), dim=0).unsqueeze(0))
                        elif np.array(modes)[:, bsz][0] == 'txt':
                            if np.array(modes)[:, bsz][1] == 'img2':
                                x.append(torch.cat((x_text[bsz], x_images[1][bsz], x_images[2][bsz], x_images[0][bsz]), dim=0).unsqueeze(0))
                            else:
                                x.append(torch.cat((x_text[bsz], x_images[2][bsz], x_images[1][bsz], x_images[0][bsz]), dim=0).unsqueeze(0))

                else:
                    print(modes[-1][0])
                    raise ValueError

            x = torch.cat(x, dim=0)

        assert x.size(0) == b

        x = self.dropout(x)

        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb=layer_pos_emb, causal=causal, condition_len=n_condition, **kwargs)  # x: [B, seq_len, dim] -> [B, seq_len, dim]
        x = self.norm(x)

        if return_encodings:  # usually False
            return x

        if self.attn_type in ['all_modality_causal_noncuda', 'all_modality_causal_cuda']:
            return self.to_out_combined_txt_img(x)
        return x @ self.token_emb.weight.t()

    # !# Generate Report
    @torch.no_grad()
    @eval_decorator
    def generate_texts(
            self,
            # img1,  # tensor[B, img1_len]
            # img2,  # tensor[B, img2_len]
            # view,
            # modes,
            batch,
            *,
            sos_token_idx=None,
            eos_token_idx=None,
            pad_token_idx=None,
            filter_logits_fn='top_k',
            filter_thres=0.9,
            temperature=1.,
            causal='conditioned_causal'
    ):
        total_len = self.max_seq_len
        txt, img1, modes, view = batch['txt'], batch['img1'], batch['modes'], batch['view_position']
        B, img1_seq_len, device = *img1.shape, img1.device
        _, txt_seq_len = txt.size()

        if 'img2' in batch.keys():
            assert self.max_img_num >= 2
            img2 = batch['img2']
        if 'img3' in batch.keys():
            assert self.max_img_num == 3
            img3 = batch['img3']

        if self.max_img_num == 1:
            assert modes[0][0] == 'img1'
            images = img1
        elif self.max_img_num == 2:
            if modes[0][0] == 'img1':
                images = torch.cat((img1, img2), dim=1)  # -> [B, image_seq_len]
            elif modes[0][0] == 'img2':
                images = torch.cat((img2, img1), dim=1)  # -> [B, image_seq_len]
            else:
                raise ValueError
        elif self.max_img_num == 3:
            if modes[0][0] == 'img1':
                if modes[1][0] == 'img2':
                    images = torch.cat((img1, img2, img3), dim=1)
                else:
                    images = torch.cat((img1, img3, img2), dim=1)
            elif modes[0][0] == 'img2':
                if modes[1][0] == 'img1':
                    images = torch.cat((img2, img1, img3), dim=1)
                else:
                    images = torch.cat((img2, img3, img1), dim=1)
            elif modes[0][0] == 'img3':
                if modes[1][0] == 'img1':
                    images = torch.cat((img3, img1, img2), dim=1)
                else:
                    images = torch.cat((img3, img2, img1), dim=1)

        B, image_seq_len, device = *images.shape, images.device
        out = torch.cat((images, torch.tensor([[sos_token_idx]] * B).to(device)), dim=-1)
        batch['txt'] = out[:, image_seq_len:]

        if filter_logits_fn == 'top_k':
            filter_logits_fn = top_k
        elif filter_logits_fn == 'top_p':
            filter_logits_fn = top_p
        else:
            raise ValueError('filter_logits_fn must be in (top_k, top_p)')

        for cur_len in range(txt_seq_len - 1):
            batch['txt'] = out[:, image_seq_len:]
            logits = self(batch, causal=causal)
            max_neg_value = -torch.finfo(logits.dtype).max
            logits[:, :, self.num_txt_tokens:] = max_neg_value
            logits = logits[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)  # [B, num_text_tokens]
            sample = torch.multinomial(probs, 1)  # [B, 1]
            out = torch.cat((out, sample), dim=-1)
            # break check
            if ((out[:, image_seq_len:] == eos_token_idx).sum(dim=-1) > 0).sum() == B:
                break

        text_seq = out[:, image_seq_len:]

        # postprocess
        indices = [list(row).index(eos_token_idx) if eos_token_idx in row else -1 for row in text_seq]
        for row, idx in enumerate(indices):
            if idx >= 0:
                text_seq[row, idx + 1:] = pad_token_idx

        batch['txt'] = txt
        pad_size = (0, txt_seq_len - text_seq.size(-1))
        gen_texts = F.pad(text_seq, pad_size, 'constant', pad_token_idx)
        return gen_texts

    # !# Generate Certain Image
    @torch.no_grad()
    @eval_decorator
    def generate_image(self,
                       # txt,
                       # img,
                       # view,
                       # modes,
                       batch,
                       *,
                       filter_logits_fn='top_k',
                       filter_thres=0.9,
                       temperature=1.,
                       causal='conditioned_causal',
                       target_gen_view='AP',
                       ):
        txt, img1, modes, view = batch['txt'], batch['img1'], batch['modes'], batch['view_position']

        if 'img2' in batch.keys():
            assert self.max_img_num >= 2 or self.max_img_num == -1
            img2 = batch['img2']
        if 'img3' in batch.keys():
            assert self.max_img_num == 3 or self.max_img_num == -1
            img3 = batch['img3']

        B, n_txt, device = *txt.shape, txt.device

        att_sos_special_tokens = {'AP': 1025, 'PA': 1027, 'LATERAL': 1029, 'LL': 1029, 'PAD': 1024}

        if self.max_img_num == 1:
            out = txt
        elif self.max_img_num == 2:
            if modes[-1][0] == 'img2':
                if modes[0][0] == 'img1':
                    out = torch.cat((batch['img1'], txt), dim=1).to(device)
                else:
                    out = torch.cat((txt, batch['img1']), dim=1).to(device)
            elif modes[-1][0] == 'img1':
                if modes[0][0] == 'img2':
                    out = torch.cat((batch['img2'], txt), dim=1).to(device)
                else:
                    out = torch.cat((txt, batch['img2']), dim=1).to(device)
        elif self.max_img_num == 3:
            mode_to_data = {'img1': batch['img1'], 'img2': batch['img2'], 'img3': batch['img3'], 'txt': batch['txt']}
            conditioned_data = []
            for mode in modes[:-1]:
                conditioned_data.append(mode_to_data[mode[0]])
            out = torch.cat(conditioned_data, dim=1).to(device)
        B, seq_len, device = *out.shape, out.device
        out = torch.cat([out, torch.tensor([[att_sos_special_tokens[i]] for i in view[-1]]).to(device)], dim=-1)

        if filter_logits_fn == 'top_k':
            filter_logits_fn = top_k
        elif filter_logits_fn == 'top_p':
            filter_logits_fn = top_p
        else:
            raise ValueError('filter_logits_fn must be either top_k or top_p')

        for cur_len in range(self.img_len-1):
            batch[modes[-1][0]] = out[:, seq_len:]

            logits = self(batch, causal=causal)
            max_neg_value = -torch.finfo(logits.dtype).max
            logits[:, :, :self.num_txt_tokens] = max_neg_value
            logits = logits[:, -1, :]

            if cur_len != (self.img_len-2):
                logits[:, (self.img_vocab_size + self.num_txt_tokens):] = float('-inf')

            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)  # [B, 1]

            sample -= self.num_txt_tokens

            if cur_len != (self.img_len - 2):
                assert not set(sum(sample.tolist(), [])) & set(range(1024, self.num_img_tokens)), f'{sample}, Special token are sampled in wrong position.'

            out = torch.cat((out, sample), dim=-1)
        image_seq = out[:, seq_len:]

        if modes[-1][0] == 'img1':
            batch[modes[-1][0]] = img1
        elif modes[-1][0] == 'img2':
            batch[modes[-1][0]] = img2
        elif modes[-1][0] == 'img3':
            batch[modes[-1][0]] = img3

        return image_seq
