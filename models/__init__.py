"""
Модели и их пресеты.

В train.py модель выбирается по имени:  --model unet
Гиперпараметры каждой модели живут здесь, в CONFIGS, и правятся руками.
Чтобы добавить вариант, допиши новую строку в CONFIGS.
Чтобы добавить новую архитектуру, положи ее в отдельный файл и зарегистрируй в ARCHS.
"""
from .unet import UNet

# архитектуры: имя -> класс
ARCHS = {
    "unet": UNet,
}

# пресеты: имя -> архитектура и ее гиперпараметры (img_size приходит из train.py)
CONFIGS = {
    # базовая модель, на которой учились cat64_eps и cat64_uniq, ~21M параметров
    "unet": dict(arch="unet", base=64, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.1, norm_groups=32),
    # та же базовая модель, но GroupNorm на 8 групп, как у unet_small: отличие от него только в размере и dropout
    "unet_gn8": dict(arch="unet", base=64, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.1, norm_groups=8),
    # в ~4 раза меньше и с сильным dropout: эксперимент против запоминания.
    # norm_groups=8, чтобы в группе GroupNorm было по 4 канала, как в DDPM
    "unet_small": dict(arch="unet", base=32, mults=(1, 2, 3, 4), num_res=2, attn_res=(16,), dropout=0.3,
                       norm_groups=8),
}


def build_model(name, img_size):
    """Собирает модель по имени пресета. Возвращает модель и полную конфигурацию для чекпоинта."""
    cfg = dict(CONFIGS[name])
    arch = cfg.pop("arch")
    model = ARCHS[arch](img_size=img_size, **cfg)
    return model, {"arch": arch, **model.config}


def config_from_checkpoint(ck):
    """Конфигурация модели из чекпоинта. Понимает новые чекпоинты (model_cfg) и старые (параметры в args)."""
    if "model_cfg" in ck:
        cfg = dict(ck["model_cfg"])
    else:
        a = ck["args"]
        cfg = dict(arch="unet", img_size=a["img_size"], in_ch=3, base=a["base_ch"],
                   mults=a["ch_mult"].split(","), num_res=a["num_res"],
                   attn_res=[v for v in a["attn_res"].split(",") if v], dropout=a["dropout"])
    cfg.setdefault("arch", "unet")
    cfg.setdefault("norm_groups", 32)  # до появления параметра всегда было 32
    for key in ("mults", "attn_res"):
        if key in cfg:
            cfg[key] = tuple(int(v) for v in cfg[key])
    return cfg


def model_from_checkpoint(ck, weights="ema"):
    """Собирает модель по конфигурации из чекпоинта и загружает веса (по умолчанию EMA)."""
    cfg = config_from_checkpoint(ck)
    arch = cfg.pop("arch")
    model = ARCHS[arch](**cfg)
    if weights:
        model.load_state_dict(ck[weights])
    return model


__all__ = ["UNet", "ARCHS", "CONFIGS", "build_model", "config_from_checkpoint", "model_from_checkpoint"]
