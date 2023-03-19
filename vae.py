import importlib
from math import sqrt, log
from einops import rearrange
from omegaconf import OmegaConf

import torch
from torch import nn
import torch.nn.functional as F

from taming.models.vqgan import GumbelVQ

# helper

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()
        model_path = vqgan_model_path
        config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        self.f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        self.fmap_size = config.model.params.ddconfig.attn_resolutions[0]
        self.num_layers = int(log(self.f)/log(2))
        self.image_size = config.model.params.ddconfig.resolution
        self.num_tokens = config.model.params.n_embed
        self.embed_dim = config.model.params.embed_dim
        self.is_gumbel = isinstance(self.model, GumbelVQ)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1  # 0. ~ 1. -> -1. ~ 1.
        z_q, emb_loss, [perplexity, min_encodings, indices] = self.model.encode(img)
        indices = indices.squeeze(-1)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return z_q, emb_loss, perplexity, rearrange(indices, '(b n) -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float() #1024
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)
        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))

        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img # 0. ~ 1.

    def forward(self, img):
        raise NotImplemented