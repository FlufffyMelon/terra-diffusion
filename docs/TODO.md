# Plan: from a large cat dataset to realistic photos of Terra

Goal: pretrain a latent diffusion model on other people's cats on square 512x512 frames, then experiment
with the 3:4 and 4:3 formats, then fine-tune on photos of Terra.

The data lives outside the project: `~/DiffusionLearn_data/` on the cluster. Mutagen does not sync it.
Overview of the sources: [research/cat_datasets.md](research/cat_datasets.md).

## Decisions made

- [x] Base: iNaturalist (captive, CC without ND) and Open Images (frames with human-annotated boxes), extras: pop-cats and Cat Individuals.
- [x] Detection: the YOLO26x + RT-DETR-x pair. Both found a cat with confidence >= 0.5, box IoU >= 0.7, the final box is their union.
- [x] Crop rule: the cat box is always entirely inside the crop, the crop does not require stretching the photo.
- [x] Cats at the edge of the frame are kept, close-ups are acceptable.
- [x] Min-SNR is off: on the project data it worsened KID and copying. Not tested for the latent model.
- [x] Pexels is not used: the site's terms forbid automated collection for ML.
- [x] The project is personal, but the author and license of each photo are kept.

## Formats

| Format | Pixels (H x W) | Latent (H x W) | Purpose |
|---|---|---|---|
| 1:1 | 512 x 512 | 64 x 64 | pretraining |
| 3:4 | 512 x 384 | 64 x 48 | experiments after pretraining |
| 4:3 | 384 x 512 | 48 x 64 | experiments after pretraining |

The latent sides must be divisible by 8 because of the three downsamplings in the U-Net, so 576 x 432 (latent 72 x 54) does not work.
A crop fits a format if `max(box_H, box_W * k, out_H) <= min(H, W * k)`, where `k = out_H / out_W`.

Subsets (measured 2026-11-02, on frames after detection, iNaturalist partial):

| Source | Total | 1:1 | 3:4 | 4:3 | At least one | Rectangle only | 1:1 but no rectangle |
|---|---|---|---|---|---|---|---|
| Terra | 394 | 284 | 325 | 231 | 389 | 105 | 0 |
| pop-cats | 9 454 | 2 712 | 3 516 | 2 046 | 5 186 | 2 474 | 434 |
| Cat Individuals | 12 207 | 7 939 | 7 310 | 8 540 | 11 894 | 3 955 | 75 |
| Open Images | 10 622 | 5 740 | 3 775 | 7 907 | 9 266 | 3 526 | 475 |
| iNaturalist, ~6% of the data | 4 558 | 2 758 | 3 219 | 2 483 | 4 209 | 1 451 | 49 |

Conclusion: square frames almost always fit at least one rectangle, but not both, so the three flags are stored separately.

## Test pretraining `ldm_v1`

A check of what images come out at 512x512 with no tricks: `v`, linear noise schedule, cosine LR, U-Net `base=192`,
attention at 32, 16, 8, all `fit_1x1` frames with Terra, 7 GPUs on `node211-17`, batch 64 per GPU, 26k steps, Lightning (`train_ldm.py`).
Latents are precomputed: the VAE encodes ~25 frames/s while the U-Net consumes ~100, so 8 random crops with a flip per frame instead of a crop on the fly.

DDP measurement on 7 GPUs: batch 32 gives 1.14 steps/s (255 frames/s), batch 64 gives 0.93 steps/s (417 frames/s). The step barely slows down as the batch grows,
because ~0.5 s goes to synchronizing 800 MB of gradients, so batch 64 is more efficient: in the same time 18% fewer steps but 1.6 times more frames.

Outcome of `ldm_v1`: at step 5 400 the loss jumped from 0.176 to 0.27 and returned only around step 16 800, when LR dropped below ~6e-5.
The cause, according to `scripts/eval/diag_ldm.py`: attention collapse. A single head over 384-768 channels without normalization of q and k, logits up to 18 000,
entropy 0.01 of log N at 16x16 already by step 5 000. EMA 5 000 and raw weights 23 000 give the same loss, i.e. 18k steps were wasted.
Also, all DDP processes ran with the same seed, and t, noise and dropout were identical on all GPUs.

## Pretraining `ldm_v2`

The same as `ldm_v1`, plus fixes: heads of 64 channels, RMSNorm on q and k, LR warmup of 2 000 steps up to 1e-4, a separate seed per process.
`metrics.csv` has the gradient norm before clipping and the worst attention block, `attn.csv` every block, raw weights every 1 000 steps in `raw/`.
6 GPUs on `node211-17`, batch 64 per GPU, 26k steps.

## 1. Data collection

- [x] Download Open Images: 12 839 frames.
- [x] Download and unpack pop-cats and Cat Individuals.
- [x] Finish downloading iNaturalist (`download_cats.py inat`, in `tmux`): 80 387 photos.
- [x] Run iNaturalist through both detectors: 80 293 processed, 94 photos were not accepted by the detectors and did not enter the dataset.

Outcome of `pretrain_v1`: 76 048 frames after detection, 58 673 after `filter_frames.py`, of which 39 259 in 1:1
(iNaturalist 31 017, Open Images 4 308, pop-cats 2 315, Cat Individuals 1 335, Terra 284).

## 2. Manifest and filter

A single file `manifest.jsonl`, one line per frame: source, path, sizes, final box, `fit_1x1`, `fit_3x4`, `fit_4x3`, license, author.

- [ ] Write `build_manifest.py`: gather the detections and apply the agreement conditions and box area >= 0.1.
- [ ] Open Images: use the human boxes, detectors only for checking.
- [ ] Compute `fit_*` for the three formats.
- [ ] Remove frames with several cats: currently 2-16% of those that passed.
- [ ] iNaturalist: exclude observations marked dead (345 of them, `term_id=17`, `term_value_id=19`).
- [ ] Remove black-and-white photos.
- [ ] Cat Individuals: limit the number of frames per individual and filter out pink cages, otherwise the model will learn this background.
- [ ] Remove duplicates and near-duplicates (perceptual hash).
- [ ] Review samples by eye for each source and for the rejected frames.
- [ ] Check whether expanding the box by a few percent reduces the share of cut-off cats. Currently it is about 12% on Open Images boxes.
- [ ] Consider a CLIP filter against drawings, screenshots, toys and collages, if they turn up in review.

## 3. Latent pipeline

- [ ] Compute the VAE scale factor on the final set. On Terra photos it is 0.231, not the standard 0.18215.
- [ ] Write the latent computation and cache. The crops are random, so either precompute a set of crops per frame or encode on the fly.
- [ ] Model preset: `in_ch=4`, `base=128`, attention at 32, 16, 8.
- [ ] Adapt `experiments/pixel64/memorization.py`: decode the latents, compute KID and `ratio<0.7` in pixels.
- [ ] A metric for cut-off cats and poses: run a detector on the samples, the share of boxes at the edge and of elongated boxes. KID does not see this.

## 4. Pretraining at 512x512

- [ ] Choose a variant: `eps` or `v` without Min-SNR. At 64 px `eps` gave the best balance, `v` copied less but the images were softer. Compare with a short run.
- [ ] Train on `fit_1x1` frames.

U-Net speed measurement on a 64x64x4 latent (RTX 4500 Ada, bf16, torch.compile, synthetic data, attention at 32, 16, 8, batch 64):

| base | Parameters | Steps/s | Memory | 300k steps |
|---|---|---|---|---|
| 128 | 90M | 2.78 | 8.1 GB | 30 h |
| 192 | 201M | 1.63 | 12.6 GB | 51 h |
| 256 | 358M | 1.11 | 17.8 GB | 75 h |

- [ ] Baseline variant: `base=192`, batch 64, 300k steps, which is ~19 million frames and ~60 GPU-hours with a margin for samples and data loading.
- [ ] Add DDP to `train.py`: on 4 GPUs we expect about 15-17 hours instead of 60, with the same or slightly more GPU-hours.
- [ ] VAE: candidate `stabilityai/sdxl-vae` (PSNR 0.4-0.7 dB higher than `sd-vae-ft-mse`), compare reconstructions by eye. Recompute the scale factor on the final set.
- [ ] If copying is high, add outdoor iNaturalist cats (~80 thousand observations).
- [ ] Evaluate KID, copying, samples.

Anatomy:

- [ ] Cluster label: cluster the DINOv2 features and train with classifier-free guidance (Self-Guided Diffusion). Choose the number of clusters before training starts.
- [ ] Autoguidance: guide the model with an early checkpoint or a smaller version, with no extra training.
- [ ] Replace Inception with DINOv2 in the quality metrics.
- [ ] Check `v` for composition: at 64 px it removed obvious defects.
- [ ] Select finished samples with a detector or discriminator.
- [ ] Reference: LoRA on SDXL, to see the limit of our own model.

## 5. Experiments with 3:4 and 4:3

The U-Net is convolutional, has no positional embeddings, and attention does not depend on the map size, so the weights of the square model
load without changes. We expect a short fine-tune to be enough, but this has not been verified.

- [ ] Check generation with the square model at 64 x 48 and 48 x 64 without fine-tuning.
- [ ] Fine-tune on `fit_3x4` and `fit_4x3` frames, with each batch from a single format.
- [ ] Bring rectangular frames back into `train.py` (`--buckets`): the work was started and postponed, the code was removed.
- [ ] Compare the share of elongated poses with the square model.

## 6. Fine-tuning on Terra

- [ ] Class label `cat / Terra`, 10% of batches without a label for classifier-free guidance, feed Terra photos more often.
- [ ] A short fine-tune with a small LR.
- [ ] In rectangles 389 of 394 frames are available for Terra instead of 284 in the square: this brings back the elongated poses.
- [ ] Metrics on the held-out 30 Terra photos.

## Ideas for later

- Shoot more of Terra: 30-50 frames of elongated poses, landscape orientation, with room around the cat.
- Feed frames with elongated boxes more often.
- Rotation by +-10-15 degrees for shots from above.
- LoRA on SDXL as a quality reference.
