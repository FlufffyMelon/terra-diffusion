"""
Датасеты для латентной диффузии.

FrameCrops  : кадры из manifest.jsonl -> несколько случайных кропов, для подсчета латентов.
CatLatents  : готовые латенты из <dataset>/latents, для обучения.
CondLatents : то же с условиями «кто» (Терра или другая кошка) и «тип» (кластер DINOv2), для ldm_v4.

Правило кропа то же, что в train.py: бокс кошки всегда целиком внутри, кроп не растягивает фото.
VAE кодирует около 25 кадров/с на GPU, а U-Net принимает около 100, поэтому кодировать на лету нельзя:
на кадр заранее считается несколько кропов, и при обучении берется случайный из них.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from train import crop_offsets, min_crop_side, to_unit


def read_manifest(path, flag="fit_1x1"):
    """Строки манифеста, годные для формата flag."""
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return [r for r in rows if r[flag]]


class FrameCrops(Dataset):
    """
    Кадр -> crops случайных квадратов size x size в [-1, 1], каждый со своим случайным отражением.
    Возвращает (crops, 3, size, size), номер строки и ok. Битый файл дает ok = False.
    """

    def __init__(self, rows, root, size=512, crops=4, seed=0):
        self.rows, self.root, self.size, self.crops, self.seed = rows, Path(root), size, crops, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        S = self.size
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        try:
            im = Image.open(self.root / r["path"])
            im.draft("RGB", (2 * S, 2 * S))  # быстрое декодирование JPEG, разрешения хватает для кропа S
            img = TF.pil_to_tensor(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception:
            return torch.zeros(self.crops, 3, S, S), i, False
        _, H, W = img.shape
        short = min(H, W)
        b = r["box"]
        box = (max(math.floor(b[0] * W), 0), max(math.floor(b[1] * H), 0), min(math.ceil(b[2] * W), W), min(math.ceil(b[3] * H), H))
        # в исходном фото кроп не меньше S, а фото могло декодироваться уменьшенным
        lo = min_crop_side(box, H, W, min(math.ceil(S * short / r["short"]), short))
        out = []
        for _ in range(self.crops):
            side = torch.randint(lo, short + 1, (1,), generator=g).item()
            top_min, top_max, left_min, left_max = crop_offsets(box, H, W, side)
            top = torch.randint(top_min, top_max + 1, (1,), generator=g).item()
            left = torch.randint(left_min, left_max + 1, (1,), generator=g).item()
            c = TF.resized_crop(img, top, left, side, side, [S, S], antialias=True)
            if torch.rand(1, generator=g).item() < 0.5:
                c = c.flip(-1)
            out.append(to_unit(c))
        return torch.stack(out), i, True


class CatLatents(Dataset):
    """Латенты всех кадров в памяти: (кадров, кропов, 4, 64, 64). При каждом обращении случайный кроп кадра.
    Берутся только кадры, которые есть в текущем manifest.jsonl: после дочистки датасета латенты не нужно пересчитывать."""

    def __init__(self, dataset_dir, limit=0):
        d = Path(dataset_dir) / "latents"
        shards = sorted(d.glob("shard_*.npy"))
        if not shards:
            raise SystemExit(f"нет латентов в {d}")
        keep = {r["path"] for r in read_manifest(Path(dataset_dir) / "manifest.jsonl")}
        z, self.dropped = [], 0
        for p in shards:
            names = json.loads(p.with_suffix(".json").read_text())
            m = np.array([n in keep for n in names], dtype=bool)
            z.append(np.load(p)[m])
            self.dropped += int((~m).sum())
        self.z = torch.from_numpy(np.concatenate(z))
        if limit:
            self.z = self.z[:limit]
        sample = self.z[torch.randperm(len(self.z))[:4096]].float()
        self.scale = 1 / sample.std().item()   # в модель латенты идут с дисперсией 1, как и картинки в [-1, 1]

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        k = torch.randint(self.z.shape[1], (1,)).item()
        return self.z[i, k].float() * self.scale


class CondLatents(Dataset):
    """
    Латенты с условиями: (латент, y), y = (кто, тип). Кто: 0 другая кошка, 1 Терра. Тип: кластер из clusters.json,
    по признакам DINOv2 это в основном окрас и порода, а не поза или фон.
    Другие кошки из latents/, Терра из latents_terra/ (больше кропов на кадр) без кадров terra_holdout.json.
    Терра составляет примерно terra_frac примеров: после n_other обычных индексов идут индексы Терры,
    каждый берет случайный кадр Терры, так что перемешивание и DistributedSampler работают как обычно.
    """

    def __init__(self, dataset_dir, terra_frac=0.1, limit=0):
        root = Path(dataset_dir)
        keep = {r["path"] for r in read_manifest(root / "manifest.jsonl")}
        clusters = json.loads((root / "clusters.json").read_text())
        self.k, assign = clusters["k"], clusters["assign"]
        hold = set(json.loads((root / "terra_holdout.json").read_text()))

        def load(sub, ok):
            z, names, out = [], [], 0
            for p in sorted((root / sub).glob("shard_*.npy")):
                n = json.loads(p.with_suffix(".json").read_text())
                m = np.array([ok(v) for v in n], dtype=bool)
                z.append(np.load(p)[m])
                names += [v for v, f in zip(n, m) if f]
                out += int((~m).sum())
            return torch.from_numpy(np.concatenate(z)), names, out

        # dropped: кадры latents/, которые не берутся отсюда (нет в манифесте или это Терра, она идет из latents_terra/)
        self.z, names, self.dropped = load("latents", lambda v: v in keep and not v.startswith("terra/"))
        self.zt, tnames, _ = load("latents_terra", lambda v: v in keep and v not in hold)
        if limit:
            self.z, names = self.z[:limit], names[:limit]
        self.scene = torch.tensor([assign[v] for v in names])
        self.scene_t = torch.tensor([assign[v] for v in tnames])
        self.n_other = len(self.z)
        self.length = round(self.n_other / (1 - terra_frac))
        sample = self.z[torch.randperm(len(self.z))[:4096]].float()
        self.scale = 1 / sample.std().item()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i < self.n_other:
            z, who, scene = self.z[i], 0, self.scene[i]
        else:
            j = torch.randint(len(self.zt), (1,)).item()
            z, who, scene = self.zt[j], 1, self.scene_t[j]
        k = torch.randint(z.shape[0], (1,)).item()
        return z[k].float() * self.scale, torch.tensor([who, int(scene)])
