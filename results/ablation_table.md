# Ablation

What each architecture piece contributes, built up one at a time on the same
cached features (`python -m src.train --tag <variant> <flags>`). All four
variants train the same ~564k-parameter head budget; the flags only change
which inputs and objectives are active.

| Variant | Flags | Components |
|---|---|---|
| `baseline` | `--no-forensic --no-gate --no-consistency` | frozen CLIP embedding → head |
| `aug` | `--no-forensic --no-gate` | + degraded views in the training data |
| `freq` | `--no-gate` | + 129-d frequency / forensic branch |
| `full` | *(none)* | + degradation-aware gate + consistency loss |

## In-distribution held-out set

`auc_aug` is measured on degraded views of the same held-out images, so it is
a robustness proxy.

| Variant | clean AUC | degraded-view AUC | clean−degraded gap | ECE |
|---|---|---|---|---|
| `baseline` | 0.9995 | 0.9984 | +0.0011 | 0.0065 |
| `aug` | 0.9997 | 0.9985 | +0.0013 | 0.0061 |
| `freq` | 0.9996 | 0.9985 | +0.0011 | 0.0068 |
| `full` | 0.9997 | 0.9986 | +0.0011 | 0.0063 |

## Transfer — WildFake `laion_matched` (2,500-image subsample, never trained on)

| Variant | AUC | balanced acc @0.5 | @best thr. | F1 | ECE | FP on real photos |
|---|---|---|---|---|---|---|
| `baseline` | 0.9890 | 0.9500 | 0.9532 | 0.9505 | 0.033 | 5.9% |
| `full` | 0.9886 | 0.9464 | 0.9496 | 0.9468 | 0.038 | 6.1% |

## Reading this

**The four variants are statistically indistinguishable** — every AUC sits
within ±0.001 of every other, on both the in-distribution set and the
transfer benchmark. The frozen CLIP embedding alone reaches 0.989 transfer
AUC; the frequency branch, the gate, and the consistency loss neither help
nor hurt on this data mix.

That is an honest null result, and it has a cause: after the external DALL·E 3
and ProGAN pass, the training corpus is largely **semantically** separable,
and CLIP ViT-B/16 embeddings are already robust to the six transforms. The
design rationale — a frequency-forensic fallback for the cases semantics
cannot catch — is sound, but this particular corpus does not exercise it.

The full architecture is kept in the shipped checkpoint anyway: it costs
almost nothing (≈564k added parameters, a few ms per image), it never
regresses the metrics, and it is the right structure for a harder or more
adversarial distribution than the one measured here. The place it would earn
its keep — heavily compressed images of generators whose semantic signature
is weak — is exactly the place a benchmark drawn from clean-ish web images
does not probe.
