# Written Project Description

*Draft for the Devpost submission form. TikTok TechJam 2026 — Problem
Statement #5: Robust Detection of AI-Generated Images Under Real-World
Transformations. Solo submission.*

---

## How the solution addresses the problem statement

The problem statement asks for a detector whose accuracy holds after images
have been through the transformations a real distribution pipeline applies —
JPEG re-encoding, resizing, blur, noise, colour shifts, cropping.

The sharpest signals that an image was generated (periodic upsampling ripples
in the Fourier spectrum, a missing sensor-noise floor, over-smooth
micro-texture) all live in the high-frequency band, and that band is exactly
what compression and downscaling destroy first. A detector tuned only on
pristine images can score 99% and then collapse on the same images once they
have been posted.

This submission handles that with three ideas:

1. **Two evidence branches.** A frozen CLIP ViT-B/16 embedding (coarse but
   compression-robust) and a 129-dimensional hand-designed frequency /
   forensic feature vector (precise on clean images, fragile once the
   spectrum is smeared).

2. **A degradation-aware gate.** A small head estimates how damaged each image
   is — trained for free, because the augmentation pipeline applied the
   damage and therefore knows the answer — and that estimate sets the fusion
   weight between the two branches. A pristine PNG leans on frequency
   evidence; a q30 re-encode leans on semantics; nothing tells the model
   which case it is at test time.

3. **A consistency loss.** Training pairs each clean view with a degraded one
   and adds a symmetric-KL term that pushes the two predictions *together*.
   Ordinary augmentation asks the model to also get the damaged copy right;
   consistency asks for the *same* answer as the clean copy — a stronger
   constraint, and the exact property the robustness evaluation measures.

**Measured result.** Across a 16-cell transform grid (JPEG down to q30, blur
to σ=2.0, downscale to 0.25×, noise to σ=0.1, ±20% colour jitter, 80% crop),
clean-image AUC is 0.999 and no single cell falls below 0.993 AUC — a mean
AUC drop of +0.0017 under transformation. On a separate, never-trained-on
transfer benchmark (a WildFake subset) the model reaches 0.989 AUC on the
leak-free `laion_matched` configuration.

An ablation showed that on this training corpus the frozen CLIP branch alone
carries those numbers — the frequency branch and gate neither help nor hurt
here (see `results/ablation_table.md`). They are kept for the harder
distributions they were designed for, at near-zero cost.

## Development tools used

- **VS Code** — primary editor, with the terminal-based agent workflow used
  throughout development.
- **Google Colab** — a committed notebook (`notebooks/aigc_detector_colab.ipynb`)
  that clones the repo and runs the shipped checkpoint on a CPU runtime, for
  anyone to reproduce inference without a local setup.
- **Git / GitHub** — public repository, linear commit history.
- Hardware: a single consumer RTX 5060 Laptop GPU (8 GB, compute capability
  sm_120).

## Models and APIs used

- **OpenAI CLIP ViT-B/16** as a frozen semantic backbone, loaded via the
  `open_clip` library (`ViT-B-16-quickgelu`, `openai` pretrained weights). No
  fine-tuning of the backbone; only a ~564k-parameter head is trained.
- **Hugging Face Hub API** (`huggingface_hub`, `datasets`) — to stream the
  training data subset and the transfer-benchmark evaluation set.
- No hosted inference APIs, no private or custom pretrained weights. Total
  parameter count is 86.76M (563,724 trainable + 86,192,640 frozen), 23×
  under the 2-billion cap.

## Libraries and frameworks used

| Library | Role |
|---|---|
| PyTorch + torchvision | model, training loop, GPU inference |
| open_clip_torch | the frozen CLIP backbone |
| NumPy + SciPy | the hand-designed frequency features (radial FFT profile, DCT log-magnitudes, high-pass residual moments) — all on CPU |
| scikit-learn | metrics (ROC-AUC, calibration error, TPR@FPR) |
| Pillow | image loading and the six degradation transforms |
| Hugging Face `datasets` / `huggingface_hub` | dataset streaming and the transfer benchmark |
| FastAPI + Uvicorn | the local web demo (`app.py`); the graded CLI does not need these |

## Datasets and assets used

- **SID_Set** (`saberzl/SID_Set`, CC-BY-4.0) — the training base. Real photos
  from OpenImages V7, fully-synthetic (diffusion) images, and tampered
  photos. A balanced 10k-per-class subset is streamed and re-saved as PNG at
  ≤512 px. Tampered images are labelled **real** — a person still took them.
- **DALL·E 3 images** (`ProGamerGov/dalle-3-reddit-dataset`) — 2,681 images,
  to cover a generator family SID_Set lacks.
- **ProGAN images** (`frp94/progan_val`, the ForenSynths validation set) —
  3,380 images, plus 3,327 LSUN real photos from the same archive as a
  resolution-matched control.
- **WildFake subset** (`techjam-aigc/wildfake-eval-subset`) — the track's
  reference benchmark. Used for **evaluation only**; every training image is
  perceptual-hash checked against all ~40,000 of its images and dropped on a
  match, and nothing from it ever enters the training manifest.

All datasets are public and used under their stated licences. The full
training set is 42,220 images (26,159 real / 16,061 AI), with 5,072 held out
in a physically separate directory.
