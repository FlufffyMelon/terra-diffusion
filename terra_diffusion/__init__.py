"""
terra-diffusion: a latent diffusion model for photos of cats, fine-tuned on one cat, Terra.

  models      : U-Net with multi-head QK-normalised attention and optional class conditions
  diffusion   : VP-SDE, training loss, samplers
  data        : datasets of frames and precomputed latents
  guidance    : classifier-free guidance and autoguidance wrappers
  images      : photo loading, the crop rule that keeps the whole cat inside, image grids
  metrics     : KID, Frechet distance, copy detection
  checkpoints : loading trained models from a run folder
"""
