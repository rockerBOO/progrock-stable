from itertools import islice
import torch
from safetensors.torch import load_file

from ldm.util import instantiate_from_config
from torch.types import (
     _device, 
)

# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


chckpoint_dict_replacements = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text) :]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")

    if ckpt.endswith(".safetensors"):
        pl_sd = load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    model = instantiate_from_config(config.model)

    if model == None:
        raise RuntimeError("Could not load the model, %s" % config.model)

    sd = get_state_dict_from_checkpoint(pl_sd)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)

    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    return model


# VAE

import os


def load_vae(model: torch.nn.Module, device: _device, vae_file, map_location):
    assert os.path.isfile(vae_file), f"VAE file doesn't exist: {vae_file}"
    print(f"Loading VAE weights from: {vae_file}")
    vae_ckpt = torch.load(vae_file, map_location=map_location)
    state_dict = get_state_dict_from_checkpoint(vae_ckpt)
    vae_dict_1 = {k: v for k, v in state_dict.items() if k[0:4] != "loss"}
    load_dict_to_device(model, device, vae_dict_1)


def load_dict_to_device(model: torch.nn.Module, device: _device, vae_dict):
    model.load_state_dict(vae_dict, strict=False)
    model.to(device)

def load_embedding(model: torch.nn.Module, device: _device, embed_file):
    assert os.path.isfile(embed_file), f"Embedding file doesn't exist: {embed_file}"
    print(f"Loading Embedding from: {embed_file}")
    embed_ckpt = torch.load(embed_file, map_location="cpu")
    state_dict = get_state_dict_from_checkpoint(embed_ckpt)
    embed_dict_1 = {k: v for k, v in state_dict.items() if k[0:4] != "loss"}
    load_dict_to_device(model, device, embed_dict_1)
