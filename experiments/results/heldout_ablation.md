# P3-2: gate-signal ablation on held-out cells

`sep` marks a signal whose mean on NIAH-MK3 lies strictly below its mean on every dilution-prone task, which is the property the gate thresholds. AUC is per-input separability of MK3 from the dilution-prone pool.

| cell | status | wrapper | $D$ on MK3 | min dil. | margin | margin excl. multivalue | AUC($D$) | signals with sep. |
|---|---|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B 4K | fitting | stock | 0.0448 | 0.1114 | +0.0667 | +0.0667 | 1.000 | 1/6 (head-agreement) |
| Qwen2.5-14B 4K | held out | v1 | 0.0455 | 0.0487 | +0.0032 | +0.0436 | 0.899 | 1/6 (head-agreement) |
| Llama-3.1-8B 4K | held out | v1 | 0.0710 | 0.0607 | -0.0103 | +0.0073 | 0.741 | 0/6 |
| Mistral-7B 16K | held out | v3 | 0.0544 | 0.0756 | +0.0212 | +0.0393 | 0.978 | 1/6 (head-agreement) |

**What survives.** $D$ is the only one of six statistics that achieves task-level separation anywhere, and it does so on 2 of the 3 held-out cells. Every alternative fails on every cell.

**What does not.** The AUC of $1.000$ is a property of the fitting cell. Held out it is 0.899, 0.741, 0.978, so the paper should report $1.000$ as a fitting-cell figure.

**Llama-3.1-8B 4K inverts the ordering**: MK3 sits at 0.0710, above the nearest dilution-prone task at 0.0607 (margin -0.0103), and no signal separates. That nearest task is niah_multivalue; excluding it, the margin is +0.0073. sec:limitations already excludes niah_multivalue from the task-level ordering claim on Llama-architecture models, so the claim holds on 3 of 3 held-out cells under the stated scope and 2 of 3 without it.
