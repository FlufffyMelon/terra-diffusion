"""
Metrics: Inception features and KID, Frechet distance, and the pixel-space copy test that finds samples
which are near copies of training photos (ratio of the distances to the two nearest training photos).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from PIL import Image, ImageOps
from scipy import linalg

from .images import crop_offsets, min_crop_side, to_unit


def load_inception(dev):
    """Inception-v3 from torchvision, without the last layer: outputs 2048 features per image.
    The weights are downloaded on the first call into ~/.cache/torch; on nodes without internet download them beforehand."""
    import torchvision
    net = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
    net.fc = torch.nn.Identity()
    return net.to(dev).eval()


@torch.no_grad()
def inception_feats(net, x, batch=64):
    """x : (N, 3, S, S) in [-1, 1]  ->  (N, 2048) features in float64 on CPU."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(x), batch):
        b = (x[i:i + batch].float().clamp(-1, 1) + 1) / 2
        b = F.interpolate(b, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        out.append(net((b - mean) / std).double().cpu())
    return torch.cat(out)


def kid(f_real, f_gen, n_subsets=100, seed=0):
    """
    Kernel Inception Distance: unbiased estimate of MMD^2 with kernel k(a, b) = (a.b / d + 1)^3
    between features of real photos and samples. 0 = distributions match, lower = better.
    Computed over random subsets of equal size, returns the mean and the spread.
    """
    g = torch.Generator().manual_seed(seed)
    d = f_real.shape[1]
    m = min(len(f_real), len(f_gen))
    vals = []
    for _ in range(n_subsets):
        a = f_real[torch.randperm(len(f_real), generator=g)[:m]]
        b = f_gen[torch.randperm(len(f_gen), generator=g)[:m]]
        kaa = (a @ a.T / d + 1) ** 3
        kbb = (b @ b.T / d + 1) ** 3
        kab = (a @ b.T / d + 1) ** 3
        vals.append(((kaa.sum() - kaa.diagonal().sum()) / (m * (m - 1))
                     + (kbb.sum() - kbb.diagonal().sum()) / (m * (m - 1))
                     - 2 * kab.mean()).item())
    vals = np.array(vals)
    return vals.mean(), vals.std()


def fd(a, b):
    """Frechet distance between Gaussians of two feature sets (N, d)."""
    a, b = a.numpy(), b.numpy()
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    # tr sqrt(S1 S2) = tr sqrt(sqrt(S1) S2 sqrt(S1)): the second matrix is symmetric, its root via eigvalsh is stable
    e, v = linalg.eigh(s1)
    r1 = (v * np.sqrt(np.clip(e, 0, None))) @ v.T
    tr_covmean = np.sqrt(np.clip(linalg.eigvalsh(r1 @ s2 @ r1), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def build_refs(imgs, size, scales, npos, boxes=None, nsides=5):
    """
    All crop variants of each photo: scales x shifts x flip. Returns (M, 3, S, S) and the photo index.
    With boxes, instead of scales, nsides sides from the smallest allowed up to the short side of the photo, the box is always inside.
    """
    crops, owner = [], []
    for k, img in enumerate(imgs):
        _, H, W = img.shape
        short = min(H, W)
        if boxes is None:
            sides = [int(short * s) for s in scales]
        else:
            sides = sorted(set(np.linspace(min_crop_side(boxes[k], H, W, size), short, nsides).round().astype(int).tolist()))
        for side in sides:
            top_min, top_max, left_min, left_max = (0, H - side, 0, W - side) if boxes is None else crop_offsets(boxes[k], H, W, side)
            ys = sorted(set(np.linspace(top_min, top_max, npos).round().astype(int).tolist()))
            xs = sorted(set(np.linspace(left_min, left_max, npos).round().astype(int).tolist()))
            b = torch.stack([img[:, y:y + side, x:x + side] for y in ys for x in xs])        # uint8
            b = to_unit(TF.resize(b, [size, size], antialias=True))                          # as in training
            b = torch.cat([b, b.flip(-1)])
            crops.append(b)
            owner += [k] * len(b)
    return torch.cat(crops), torch.tensor(owner, dtype=torch.long)


@torch.no_grad()
def nearest(q, refs, owner, n_img, exclude_self=None, chunk=4096):
    """
    q    : (n, D) queries, refs : (M, D) crops, owner : (M,) photo index of each crop.
    Returns d1, d2 (to the two nearest DIFFERENT photos), the index of the nearest photo and the index of the best crop.
    exclude_self : (n,) index of the photo that must not count as a neighbor (for calibration on the photos themselves).
    """
    n = q.shape[0]
    qn = (q * q).sum(1, keepdim=True)
    per_img = torch.full((n, n_img), float("inf"), device=q.device)
    best = torch.full((n,), float("inf"), device=q.device)
    best_idx = torch.zeros(n, dtype=torch.long, device=q.device)
    for s in range(0, refs.shape[0], chunk):
        r = refs[s:s + chunk]
        o = owner[s:s + chunk]
        dd = (qn + (r * r).sum(1)[None, :] - 2.0 * (q @ r.T)).clamp_min(0)   # squared distances
        if exclude_self is not None:
            dd = dd.masked_fill(exclude_self[:, None] == o[None, :], float("inf"))
        per_img.scatter_reduce_(1, o[None, :].expand(n, -1), dd, reduce="amin")
        m, a = dd.min(1)
        upd = m < best
        best[upd] = m[upd]
        best_idx[upd] = a[upd] + s
    top = per_img.topk(2, dim=1, largest=False)
    d1, d2 = top.values.sqrt().T
    return d1, d2, top.indices[:, 0], best_idx


def load_photos(root, rows, short=256):
    """Photo with the short side reduced and the cat box in pixels: (3, H, W) uint8 and (x1, y1, x2, y2)."""
    imgs, boxes = [], []
    for r in rows:
        im = ImageOps.exif_transpose(Image.open(root / r["path"])).convert("RGB")
        k = short / min(im.size)
        t = TF.pil_to_tensor(im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.BICUBIC))
        _, H, W = t.shape
        b = r["box"]
        imgs.append(t)
        boxes.append((max(math.floor(b[0] * W), 0), max(math.floor(b[1] * H), 0), min(math.ceil(b[2] * W), W), min(math.ceil(b[3] * H), H)))
    return imgs, boxes


def copy_refs(root, train_rows, size=64, npos=5, nsides=5):
    """Crops of the Terra training photos by the training rule: (M, 3*size*size) on the GPU and the photo index of each crop."""
    imgs, boxes = load_photos(root, train_rows)
    refs, owner = build_refs(imgs, size, None, npos, boxes=boxes, nsides=nsides)
    return refs.flatten(1).cuda(), owner.cuda()


def random_crops(root, rows, size=64, seed=0):
    """One random crop per photo by the training rule, in [-1, 1]: calibration on the held-out photos."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for img, box in zip(*load_photos(root, rows)):
        _, H, W = img.shape
        side = torch.randint(min_crop_side(box, H, W, size), min(H, W) + 1, (1,), generator=g).item()
        t0, t1, l0, l1 = crop_offsets(box, H, W, side)
        y, x = torch.randint(t0, t1 + 1, (1,), generator=g).item(), torch.randint(l0, l1 + 1, (1,), generator=g).item()
        out.append(to_unit(TF.resize(img[:, y:y + side, x:x + side][None], [size, size], antialias=True))[0])
    return torch.stack(out)


def copy_ratio(img, refs, owner, n_img, size=64):
    """img : (B, 3, S, S) in [-1, 1] -> ratio = d1 / d2 and the index of the nearest photo."""
    q = F.interpolate(img.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    d1, d2, nn_img, _ = nearest(q.flatten(1).cuda(), refs, owner, n_img)
    return d1 / d2, nn_img
