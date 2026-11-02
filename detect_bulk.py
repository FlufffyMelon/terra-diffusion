#!/usr/bin/env python
"""
Прогоняет детектор по всем скачанным фото кошек большими батчами. Одна модель на один GPU.

Запуск можно прервать и повторить: готовые файлы пропускаются. С --watch после обработки всего
ждет новые файлы, поэтому задачу можно запускать, пока идет скачивание.

В <root>/detections/<model>_<source>.jsonl одна строка на файл:
  f : путь относительно папки источника,  W, H : размер фото после поворота по EXIF,
  b : все боксы кошек [уверенность, x1, y1, x2, y2], координаты в долях кадра.
Без фильтра по уверенности выше --conf: пороги и согласие моделей считаются потом, без повторного прогона.

Пример:
  sbatch -p AMD7V12-RTX4500-CCP --gpus=1 -c 16 --mem=48G -t 08:00:00 -o runs/detect_%j.log \
      --wrap ".venv/bin/python -I detect_bulk.py --model yolo26x --watch 60"
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from ultralytics import RTDETR, YOLO

IMG_EXT = {".jpg", ".jpeg", ".png"}
CAT = 15            # класс COCO
PAD = 114 / 255     # серое поле, как в letterbox у ultralytics
MODELS = {"yolo26x": (YOLO, "yolo26x.pt"), "rtdetr-x": (RTDETR, "rtdetr-x.pt")}
SOURCES = {"popcats": "probe/popcats", "cati": "probe/cati", "oi": "oi/images", "inat": "inat/images"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--root", type=str, default=str(Path.home() / "DiffusionLearn_data"))
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0, help="для проверки: сколько файлов обработать")
    p.add_argument("--watch", type=int, default=0, help="сколько минут ждать новые файлы после обработки всего")
    return p.parse_args()


class Images(Dataset):
    """Фото -> квадрат size x size с серыми полями. Возвращает картинку, размеры и поля. Битый файл: W = 0."""

    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        s = self.size
        try:
            im = Image.open(self.paths[i])
            w0, h0 = im.size
            im.draft("RGB", (2 * s, 2 * s))  # быстрое декодирование JPEG сразу в уменьшенном виде
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = (w0, h0) if (im.width >= im.height) == (w0 >= h0) else (h0, w0)
            k = s / max(im.size)
            nw, nh = max(round(im.width * k), 1), max(round(im.height * k), 1)
            x = TF.pil_to_tensor(im.resize((nw, nh), Image.BILINEAR))
            px, py = (s - nw) // 2, (s - nh) // 2
            out = torch.full((3, s, s), round(PAD * 255), dtype=torch.uint8)
            out[:, py:py + nh, px:px + nw] = x
            return out, i, W, H, nw, nh, px, py
        except Exception:
            return torch.zeros((3, s, s), dtype=torch.uint8), i, 0, 0, 1, 1, 0, 0


def pending(src_dir, done):
    """Файлы папки, которых еще нет в результатах. Незаконченные загрузки (.tmp) не трогаем."""
    if not src_dir.exists():
        return []
    return sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()
                  and str(p.relative_to(src_dir)) not in done)


def run_source(model, a, src_dir, paths, out_path):
    loader = DataLoader(Images(paths, a.size), batch_size=a.batch, num_workers=a.workers, pin_memory=True,
                        prefetch_factor=4, persistent_workers=False)
    t0, n = time.time(), 0
    with open(out_path, "a") as f:
        for b, (x, idx, W, H, nw, nh, px, py) in enumerate(loader):
            x = x.to(0, non_blocking=True).float() / 255
            res = model.predict(x, conf=a.conf, classes=[CAT], imgsz=a.size, half=True, device=0, verbose=False)
            for j, r in enumerate(res):
                boxes = []
                for s, q in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    gx, gy, kx, ky = px[j].item(), py[j].item(), nw[j].item(), nh[j].item()
                    c = [(q[0] - gx) / kx, (q[1] - gy) / ky, (q[2] - gx) / kx, (q[3] - gy) / ky]
                    boxes.append([round(s, 4)] + [round(min(max(v, 0.0), 1.0), 4) for v in c])
                f.write(json.dumps(dict(f=str(paths[idx[j]].relative_to(src_dir)), W=W[j].item(), H=H[j].item(), b=boxes)) + "\n")
            f.flush()
            n += len(res)
            if b % 20 == 0:
                print(f"  {out_path.stem}: {n} из {len(paths)}, {n / (time.time() - t0):.1f} кадров/с", flush=True)
    print(f"  {out_path.stem}: готово {n}, {n / (time.time() - t0):.1f} кадров/с", flush=True)


def main():
    a = parse_args()
    root = Path(a.root)
    out = root / "detections"
    out.mkdir(exist_ok=True)
    cls, weights = MODELS[a.model]
    model = cls(str(root / "weights" / weights))
    idle = 0
    while True:
        worked = False
        for name in a.sources:
            src_dir = root / SOURCES[name]
            out_path = out / f"{a.model}_{name}.jsonl"
            done = {json.loads(l)["f"] for l in open(out_path)} if out_path.exists() else set()
            paths = pending(src_dir, done)
            paths = paths[:a.limit] if a.limit else paths
            if paths:
                print(f"{name}: новых файлов {len(paths)}, готово раньше {len(done)}", flush=True)
                run_source(model, a, src_dir, paths, out_path)
                worked = True
        if a.limit or (not worked and idle >= a.watch):
            break
        idle = 0 if worked else idle + 1
        if not worked:
            time.sleep(60)
    print("готово", flush=True)


if __name__ == "__main__":
    main()
