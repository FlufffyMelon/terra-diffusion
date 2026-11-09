"""
Image helpers shared by the pixel-space and the latent pipelines: loading photos, the crop rule that keeps
the whole cat inside the crop, the [-1, 1] value range and image grids.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode, v2
from torchvision.utils import save_image

IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_images(data_dir, pre_size, exclude=()):
    """Reads all photos from the folder root (subfolders are ignored), downscales to pre_size on the short side."""
    files = sorted(
        p for p in Path(data_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT and p.name not in exclude
    )
    imgs = []
    for p in files:
        im = Image.open(p)
        im.draft("RGB", (2 * pre_size, 2 * pre_size))  # fast JPEG decoding directly at reduced size
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = TF.resize(im, pre_size, interpolation=InterpolationMode.LANCZOS)  # short side = pre_size
        imgs.append(TF.pil_to_tensor(im))                                       # uint8, (3, H, W)
    return files, imgs


def min_crop_side(box, H, W, size):
    """Smallest crop side: fits the box and does not force stretching the crop to size. The box must fit in the photo."""
    return min(max(box[2] - box[0], box[3] - box[1], size), H, W)


def crop_offsets(box, H, W, side):
    """top_min, top_max, left_min, left_max: where a square of side `side` can sit so that the box is inside."""
    x1, y1, x2, y2 = box
    top_max, left_max = min(y1, H - side), min(x1, W - side)
    return min(max(0, y2 - side), top_max), top_max, min(max(0, x2 - side), left_max), left_max


# uint8 [0, 255] -> float [-1, 1]
TO_UNIT = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])


def to_unit(x_uint8):
    return TO_UNIT(x_uint8).as_subclass(torch.Tensor)


def center_crops(imgs, size):
    """Center square crops: the reference for the memorization check. (N, 3, size, size) in [-1, 1]."""
    out = [TF.resize(TF.center_crop(img, min(img.shape[-2:])), [size, size], antialias=True) for img in imgs]
    return to_unit(torch.stack(out))


def save_grid(x, path, nrow=8, pad=2, upscale=2):
    """x: (N, 3, H, W) in [-1, 1] -> png grid, each image upscaled by a factor of upscale without smoothing."""
    x = (x.detach().float().clamp(-1, 1) + 1) / 2
    if upscale > 1:
        x = F.interpolate(x, scale_factor=upscale, mode="nearest")
    save_image(x.cpu(), path, nrow=nrow, padding=pad, pad_value=1.0)
