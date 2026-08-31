# Error Analysis Note

Split `test` | 5072 images | threshold 0.5

- False positives (real called AI): **20** (0.6% of real images)
- False negatives (AI called real): **16** (0.8% of AI images)

## Error rate by image kind

| Kind | total | false pos | false neg | error rate |
|---|---|---|---|---|
| synthetic | 1200 | 0 | 3 | 0.2% |
| real | 1200 | 11 | 0 | 0.9% |
| tampered | 1200 | 4 | 0 | 0.3% |
| ext_progan | 406 | 0 | 7 | 1.7% |
| lsun | 399 | 5 | 0 | 1.3% |
| ext_dalle3 | 322 | 0 | 6 | 1.9% |
| real_square | 318 | 0 | 0 | 0.0% |
| hard_real | 27 | 0 | 0 | 0.0% |

![false positives](fp_grid.png)

![false negatives](fn_grid.png)

## Interpretation

**False positives — representative cases** (real photos the model called AI):

- *Studio and product photography* — a blue sports car on a seamless
  background (p=0.92), a red pickup against a cloudless sky (p=0.97), a
  styled kitchen stock shot (p=1.00). These carry the clean, even, low-noise
  look the model has learned to read as generation.
- *Photos of artwork* — a framed portrait painting (p=0.97) and a sculpted
  garden-scene cake (p=0.99, labelled tampered). The subject is synthetic
  imagery, just made by hand; the forensic branch cannot draw that line.
- *Unusual light* — a figure against a glowing green column (p=0.98), a
  backlit clock face (p=0.92), a monochrome street (p=0.75). Colour and
  noise statistics fall outside the training distribution of "real".

The two tampered-image false positives are both dramatic real photographs,
not obvious localised edits — the AI-edited region is not what triggered them.

**False negatives — representative cases** (AI images the model called real):

- *ProGAN objects dominate* — 7 of 16, all LSUN-category single objects at
  256 px (a dog, three motorbikes, a car, a potted plant, a bottle). ProGAN
  is in the training set, yet these soft, plausible object crops still slip
  through: a 2018 category GAN and the modern text-to-image generators the
  model handles well leave different traces.
- *The most photorealistic diffusion outputs* — a beach scene with a sunflower
  sculpture (p=0.03), Napoleon rendered convincingly astride a crocodile
  (p=0.11), a plain tulip still life (p=0.08). No compositional "tell", clean
  rendering, and the frequency signature is faint.

**The cost asymmetry.** At threshold 0.5 the model favours catching AI: on the
held-out set it misses 0.8% of synthetic images and wrongly flags 0.6% of real
ones. On the harder WildFake transfer benchmark the real-photo false-positive
rate on polished web imagery rises to ~6% (plain snapshots stay under 1%).
Calling a real photograph synthetic is an accusation against a person, so the
intended deployment is a human-review queue, and the threshold should be
raised wherever a false accusation costs more than a missed synthetic — the
`@best` column of the WildFake table shows most of the balanced accuracy
survives that shift.