# Devpost Submission — copy-paste reference

Every field of the TikTok TechJam 2026 submission form, in order.
Fields marked **`[FILL IN]`** need something only you can supply.

---

# Step 1 · Manage team

No fields. Add teammates by email, or share the invitation link.

---

# Step 2 · Project overview

## Project name *

```
Degradation-Aware AIGC Detector
```

*Alternatives if you want something punchier: "Survives the Feed",
"Still Fake After Compression", "Forensics After the Feed".*

## Project tagline *

```
Detecting AI-generated images after JPEG, resizing and blur have already destroyed the evidence.
```

## Thumbnail image

Upload **`assets/thumbnail.png`** — 1200×800, exactly the 3:2 ratio Devpost
asks for, 69 KB (well under the 5 MB cap).

---

# Step 3 · Project details

## Project story *

*Devpost renders Markdown. Paste everything between the rules below.*

---

## Inspiration

Every image a moderation system actually sees has already been through an
upload pipeline. It has been re-encoded as JPEG, resized for a thumbnail,
maybe blurred or cropped or run through a filter.

That matters more than it sounds. The sharpest evidence that an image was
generated — periodic upsampling ripples in the Fourier spectrum, a missing
sensor-noise floor, over-smooth micro-texture — all live in the
**high-frequency band**. And the high-frequency band is precisely what
compression and downscaling destroy first.

So a social platform's redistribution pipeline is, entirely by accident, an
almost optimal attack on frequency-domain forensics. A detector tuned on
pristine images can report 99% accuracy in a notebook and then collapse to
near-coin-flip on the same images once they have been posted once. We wanted
to build the version that does not do that.

## What it does

Give it an image, it returns a calibrated probability that the image is fully
AI-generated, plus a plain-language verdict. It is deliberately a **binary**
question: real versus fully synthetic.

One product decision worth stating up front: the training data also contains
*tampered* images — real photographs with an AI-edited region. We label those
**real**. A person still took the photograph, and "is this a real photo?" is
the question a viewer is actually asking.

The design goal is not peak accuracy on clean images — that is easy. It is
accuracy that holds after the image has been through a real pipeline.

## How we built it

Three ideas, stacked.

**1 · Two evidence branches.** A frozen CLIP ViT-B/16 embedding gives a
semantic read — coarse, but it barely moves under compression. Alongside it, a
129-dimensional hand-designed forensic vector (a 48-bin radial FFT log-power
profile plus slope and curvature, 64 per-position DCT log-magnitudes,
high-pass residual moments per channel, and cross-channel residual
correlations) gives a precise read on clean images that degrades fast. Every
one of those features is a sentence you can say out loud over a chart, which
matters when a detector's output is an accusation.

**2 · A degradation-aware gate.** Rather than blending the branches with fixed
weights, a small head *estimates how damaged the image is*, and that estimate
sets the fusion weights. A pristine PNG leans on frequency evidence; a q30
re-encode leans on semantics. Nothing tells the model which case it is at test
time. The estimator trains on free supervision — our augmentation pipeline
applied the damage, so it already knows the answer.

**3 · A consistency loss.** Training pairs each clean view with a degraded one
and adds a symmetric-KL term pushing the two predictions *together*. Ordinary
augmentation asks the model to also get the damaged copy right. Consistency
asks for the **same** answer as the clean copy — a strictly stronger
constraint, and exactly the property the robustness evaluation measures.

Only the head trains: **563,724 parameters** on top of an 86M frozen backbone,
23× under the 2-billion cap. Features are extracted once and cached, so a full
retrain after a data change takes seconds rather than hours — which is what
made it possible to iterate honestly on a single 8 GB laptop GPU.

## Challenges we ran into

### Four blind spots we could not close

The honest headline: we know exactly where this model fails, because we went
looking. All four are measured, reproducible, and still open.

**Hyperrealistic real photos trigger false alarms.** Studio-lit product shots,
polished travel photography and photographs *of* artwork get flagged. The
model over-indexes on the ultra-clean, low-noise look that high-end AI art
shares with professional photography — a **6.0% false-alarm rate** on genuine
photos in the benchmark's hardest real-image config, up from 1.7% before we
added DALL·E 3 data. Plain snapshots stay under 1%. This is the cost of the
capability we bought, and it is the limitation that most constrains where the
model can be deployed.

**Heavy noise works as a shield.** Intense compression or additive noise
buries the high-frequency fingerprint the forensic branch depends on, and a
degraded synthetic starts to look like a messy real snapshot. At σ=0.1 noise —
the worst cell in our grid — recall at a 1%-false-positive operating point
falls from **99.7% to 89.6%**. Ranking survives (cell AUC 0.993), but roughly
one AI image in ten slips past a strict threshold once it is grainy enough.
Adding noise is not a clever attack, and it partly works.

**GigaGAN slips through.** DALL·E 3 (93% recall) and Midjourney v5 (87%) are
caught reliably. **GigaGAN sits at a 4% detection rate.** We added ProGAN to
training specifically to fix this and it did *not* transfer — a 2018 category
GAN and a 2023 text-to-image GAN leave different traces. Architecture was not
the missing piece; the right training data is.

**Localised edits are invisible.** A real photo with a small AI-edited region
scores as ~100% real. Partly by design — a tampered photo was still taken by a
person — but it means inpainting, face swaps and object removal all pass
unflagged. Only whole-image generation is in scope.

### Three we did close

**The benchmark was gameable, and we nearly missed it.** In the spec-faithful
config of the reference evaluation set, every real image is exactly 200×200
and no generated image is. `img.size == (200, 200)` scores **AUC 1.000 with no
model at all**. Our evaluation script now measures that shortcut alongside our
own score in every run, so the two can never be confused, and we report the
leak-free config as our headline instead.

**We shipped a shortcut and had to roll it back.** An early attempt at the
GAN problem added BigGAN images — all 200×200, while every existing training
image was ~512 px. The model learned image size, not GAN artifacts: better on
one benchmark config while *regressing* another by 0.14 AUC and pushing
real-photo false positives from 0.28% to 6.4%. We reverted it entirely and
rebuilt the approach with resolution- and aspect-ratio-matched real controls
for every AI image added. The GigaGAN gap above is what remains after doing
it properly.

**A silent OOM masquerading as a network hang.** Dataset shard reads were
being OOM-killed with no traceback, which looked exactly like a stalled
download. The fix was streaming the parquet in batches (peak memory
2,850 MB → 942 MB). Separately the feature-extraction worker pool deadlocked
repeatedly on Windows; switching processes to threads solved it, since the
forensic work is SciPy FFT/DCT that releases the GIL anyway.

Two images we were handed for a targeted fix turned out to be byte-identical
to held-out evaluation images. Training on them would have leaked, so we left
them out — and they became a genuinely unbiased spot check instead.

## Accomplishments that we're proud of

**Robustness that actually holds.** Across 16 transform cells — JPEG down to
q30, blur to σ=2.0, downscale to 0.25×, noise to σ=0.1, ±20% colour jitter,
80% crop — clean AUC is **0.999** and **no single cell falls below 0.993**.
Mean AUC drop under transformation: **+0.0017**.

**It transfers.** On a reference benchmark drawn from a completely different
corpus, which the model has never trained on and which is perceptual-hash
excluded from our training data, it reaches **0.989 AUC** on the leak-free
configuration.

**We measured our own weaknesses instead of hiding them.** The ablation study
is a null result and we published it. The false-positive cost of our final
data pass is in the README, the limitations section, and the commit message.

## What we learned

**A benchmark number without its provenance is worth nothing.** The same model
scores 0.999 on one config of the same benchmark and 0.808 on another. Which
one you quote is a claim about your own honesty.

**Calibration degrades exactly where coverage does.** Expected calibration
error is 0.008 on clean held-out images and stays ≤0.026 across all sixteen
transform cells. On the transfer benchmark it holds up on the configs we
cover well (0.022–0.036) and blows out to **0.301** on `cross_generator` —
the one config dominated by a generator family we barely detect. Calibration
turned out not to be a separate problem from coverage; it is a symptom of it,
and a useful early warning that the model is out of its depth.

**Architecture is not always the lever.** Our ablation showed the frozen CLIP
branch alone matches the full model within ±0.001 AUC on this corpus. The
frequency branch, the gate, and the consistency loss are a wash here — because
after our data passes the training set became largely semantically separable,
and CLIP is already transform-robust. We kept the full architecture (it costs
≈564k parameters and never regresses anything) but we are not going to claim
credit it did not earn.

**Every capability has a price, and you should name it.** Adding DALL·E 3
images took its recall from 0.72 to 0.93 and carried Midjourney up with it —
and pushed false positives on polished real photography from 1.7% to 6.0%. We
took that trade deliberately, and we say so.

## What's next

One line per blind spot, in the order we would actually tackle them.

- **Modern GANs** *(blind spot 3)*. ProGAN went into training and did *not*
  transfer to GigaGAN. StyleGAN-3 / GigaGAN-class data — each image paired
  with resolution- and aspect-matched real controls, the way our second
  attempt was built — is the clearest next win.
- **Recover the false-positive headroom** *(blind spot 1)* with hard-negative
  mining specifically on studio, product and stock photography. The same
  targeted pass took false positives from 1.2% to 0.4% earlier in the project.
- **Train against noise as an attack** *(blind spot 2)*, not just incidental
  degradation — noise-injected synthetics, and a check on whether the gate
  learns to distrust the forensic branch harder at high σ.
- **A localisation head** *(blind spot 4)* beside the whole-image classifier,
  so inpainting and face swaps stop passing unflagged.
- **Per-domain calibration** — re-fit temperature on deployment traffic rather
  than shipping one global scalar.
- **An abstain option.** Extend the gate to emit a reliability estimate so the
  system can decline on images too degraded to judge, instead of guessing.
- **Unfreeze the top transformer blocks** and see whether the semantic branch
  can be pushed further.

---

## Built with *

*Devpost caps this at 25 tags.*

```
python · pytorch · torchvision · openai-clip · open-clip · hugging-face
huggingface-datasets · numpy · scipy · scikit-learn · pillow · matplotlib
fastapi · uvicorn · google-colab · jupyter · git · github · cuda · vs-code
computer-vision · deep-learning · image-forensics · transfer-learning
```

## Try it out links

```
https://github.com/mxhxmza/ai_img_detector
https://colab.research.google.com/github/mxhxmza/ai_img_detector/blob/main/notebooks/aigc_detector_colab.ipynb
```

## Image gallery

Upload in this order — the first three are in `assets/`, the last two in
`results/`:

| file | caption to paste |
|---|---|
| `assets/architecture.png` | Two evidence branches, mixed by a gate that estimates how damaged the image is. |
| `assets/robustness_chart.png` | 2,000 held-out images × 16 transform conditions. No cell falls below 0.993 AUC. |
| `assets/thumbnail.png` | Headline results at a glance. |
| `results/fp_grid.png` | False positives: real photos called AI — studio product shots, photos of artwork, unusual lighting. |
| `results/fn_grid.png` | False negatives: AI images called real — dominated by ProGAN objects and the most photorealistic diffusion outputs. |

## Video demo link *

```
[FILL IN]  — YouTube or Vimeo, public
```

**Suggested 2–3 minute run of show:**
1. *(0:00)* The problem — an AI image, then the same image after a JPEG
   re-encode and downscale. Naive detectors fall over on the second one.
2. *(0:25)* Architecture diagram — two branches, the gate, one sentence each.
3. *(0:50)* Live demo — run the Colab notebook, upload a mixed batch named
   `real_*` / `ai_*`, show the per-image verdicts, then the accuracy /
   precision / recall / F1 / confusion matrix it prints for the batch.
4. *(1:40)* The robustness chart — clean vs 16 transform cells.
5. *(2:05)* Honesty beat — the benchmark's 200×200 shortcut, and the
   false-positive cost we took on knowingly.
6. *(2:30)* What's next — modern GANs.

---

# Step 4 · Additional details

## File upload

Not required. If you want to attach something, a PDF export of the README
is the natural choice (the repo itself is the deliverable).

## Submitter type *

```
Team
```

## Countries of residence *

```
[FILL IN]  — all team members' countries
```

## Category *

```
Problem Statement #5 — Robust Detection of AI-Generated Images Under Real-World Transformations
```

## URL to your code repository *

```
https://github.com/mxhxmza/ai_img_detector
```

- **MIT License** — yes, `LICENSE` at the repository root.
- **Publicly available** — yes, public repository.

## Explanation of how you improved your project during the hackathon *

The project was built from nothing during the hackathon window. Development
went through four measured stages, each one evaluated before and after on both
a held-out set and an external transfer benchmark, so every claim below is a
number we can point at rather than an impression.

**Stage 1 — Baseline.** Two-branch architecture, degradation-aware gate,
consistency loss, trained on a balanced SID_Set subset. Strong
in-distribution, untested anywhere else.

**Stage 2 — Hard-negative pass.** Error analysis showed every residual false
positive had one shape: a *polished* real photograph — travel framing, shallow
depth of field, foreground bokeh — because the dataset's real class skews more
casual than its synthetic class. We mined the training reals the model scored
highest as AI, augmented them into crops, and appended them additively.
False positives fell **13 → 7** (1.2% → 0.4% of genuine photos); accuracy rose
0.994 → 0.996. A held-out slice of the augmented crops went from mean p(AI)
0.22 to 0.015, confirming it generalised rather than memorised.

**Stage 3 — Data scale-up.** Extended the subset from ~6.2k to 10k images per
class. Because our data pipeline is additive by construction — every image
that already had a train/test assignment keeps it — no image the previous
checkpoint had been *evaluated* on could drift into training. Transfer AUC
rose **0.910 → 0.930**; AI recall on the held-out set hit 1.000.

**Stage 4 — External generator coverage.** The transfer benchmark showed two
specific weaknesses: DALL·E 3 (0.72 recall) and GANs (0.04). Our training set
contained neither. We added 2,681 DALL·E 3 and 3,380 ProGAN images from
sources that are *not* the benchmark, each perceptual-hash checked against all
~40,000 benchmark images and dropped on a match. Critically, each was paired
with a matched real-image control — LSUN photos at the same 256 px for the
ProGAN images, square-cropped reals for the square DALL·E images — because our
own earlier failed attempt proved that without them the model learns
resolution and aspect ratio instead of content.

Result: transfer AUC **0.930 → 0.989**, DALL·E 3 recall **0.72 → 0.93**, and
Midjourney rose to 0.87 despite never appearing in training. Two held-out
images the model had previously scored as real — one DALL·E 3, one GigaGAN —
gave the unbiased check: the DALL·E one went **0.20 → 1.00**; the GigaGAN one
stayed missed.

**What it cost, stated plainly:** false positives on polished real photography
rose 1.7% → 6.0% (plain snapshots stay under 1%), in-distribution accuracy
slipped 0.9964 → 0.9929, and per-family robustness 0.9998 → 0.998. We took the
trade deliberately — DALL·E 3 and Midjourney are the generators a real user is
most likely to encounter — and it is documented in the README limitations, the
error analysis note, and the commit message.

We also ran and published an ablation that did **not** flatter us: on this
final corpus the frozen CLIP branch alone matches the full architecture within
±0.001 AUC. We kept the full design (near-zero cost, right structure for a
harder distribution) but did not claim credit it had not earned.

## Developer account ID *

```
[FILL IN]  — your TikTok for Developers account ID
```

## Feedback about the required developer tools

```
[FILL IN — optional]
```

Points you could raise, all encountered first-hand:

- The reference evaluation subset has a **severe artifact**: in the
  spec-faithful config, every real image is exactly 200×200 and no generated
  image is, so image dimensions alone score AUC 1.000. If the final test set
  shares this property, the leaderboard will rank resolution detectors rather
  than AIGC detectors. Worth flagging to the organisers.
- Clearer guidance on whether *tampered* images (real photo, AI-edited region)
  count as AI-generated would help — it is a genuine product decision and
  different reasonable readings give materially different numbers.

---

# Pre-submit checklist

- [x] Public GitHub repo — https://github.com/mxhxmza/ai_img_detector
- [x] MIT `LICENSE` at repo root
- [x] Run script: `python predict.py --image-dir DIR --out preds.json` →
      `[{image_path, pred}]`
- [x] README: overview · setup · reproduce · limitations · contributions
- [x] Robustness evaluation summary — `results/robustness_table.md`
- [x] Error analysis note — `results/error_analysis.md`
- [x] Thumbnail 3:2 — `assets/thumbnail.png`
- [x] Gallery images — `assets/` + `results/`
- [ ] **Demo video uploaded to YouTube (public) and linked**
- [ ] **Teammates added on Devpost**
- [ ] **Countries of residence**
- [ ] **Developer account ID**
