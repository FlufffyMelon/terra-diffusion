"""
Models and their presets.

In train.py the model is chosen by name:  --model unet
The hyperparameters of each model live here, in CONFIGS, and are edited by hand.
To add a variant, add a new line to CONFIGS.
To add a new architecture, put it in a separate file and register it in ARCHS.
"""
from .unet import UNet

# architectures: name -> class
ARCHS = {
    "unet": UNet,
}

# presets: name -> architecture and its hyperparameters (img_size comes from train.py)
CONFIGS = {
    # the base model that cat64_eps and cat64_uniq were trained with, ~21M parameters
    "unet": dict(arch="unet", base=64, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.1, norm_groups=32),
    # the same base model, but GroupNorm with 8 groups, as in unet_small: differs from it only in size and dropout
    "unet_gn8": dict(arch="unet", base=64, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.1, norm_groups=8),
    # ~4 times smaller and with strong dropout: an experiment against memorization.
    # norm_groups=8 so that a GroupNorm group has 4 channels, as in DDPM
    "unet_small": dict(arch="unet", base=32, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.3,
                       norm_groups=8),
}


def build_model(name, img_size):
    """Builds a model from a preset name. Returns the model and the full configuration for the checkpoint."""
    cfg = dict(CONFIGS[name])
    arch = cfg.pop("arch")
    model = ARCHS[arch](img_size=img_size, **cfg)
    return model, {"arch": arch, **model.config}


def config_from_checkpoint(ck):
    """Model configuration from a checkpoint. Understands new checkpoints (model_cfg) and old ones (parameters in args)."""
    if "model_cfg" in ck:
        cfg = dict(ck["model_cfg"])
    else:
        a = ck["args"]
        cfg = dict(arch="unet", img_size=a["img_size"], in_ch=3, base=a["base_ch"],
                   mults=a["ch_mult"].split(","), num_res=a["num_res"],
                   attn_res=[v for v in a["attn_res"].split(",") if v], dropout=a["dropout"])
    cfg.setdefault("arch", "unet")
    cfg.setdefault("norm_groups", 32)  # before the parameter appeared it was always 32
    for key in ("mults", "attn_res"):
        if key in cfg:
            cfg[key] = tuple(int(v) for v in cfg[key])
    return cfg


def model_from_checkpoint(ck, weights="ema"):
    """Builds a model from the checkpoint configuration and loads the weights (EMA by default)."""
    cfg = config_from_checkpoint(ck)
    arch = cfg.pop("arch")
    model = ARCHS[arch](**cfg)
    if weights:
        model.load_state_dict(ck[weights])
    return model


__all__ = ["UNet", "ARCHS", "CONFIGS", "build_model", "config_from_checkpoint", "model_from_checkpoint"]
