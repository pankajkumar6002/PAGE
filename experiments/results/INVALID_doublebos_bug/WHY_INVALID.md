# Quarantined: double-BOS tokenization, 2026-08-08

`adakv_4k_mistral7b.jsonl` (E8 Mistral cell). Do not analyse.

**Bug.** The runner called `tokenizer(prompt, return_tensors="pt")` without
`add_special_tokens=False`. The canonical runner uses it
(`gated_eviction.py:538`). Mistral's and Llama's chat templates already emit a
BOS token, so the tokenizer prepended a second one.

**Effect on this cell.** Every input carried T+1 tokens versus the released log.
`D` drifted by up to **1.13e-02**, and **18 of 400 gate calls flipped** (4.5%)
against tau = 0.07.

**Why it matters here specifically.** This is the cell where pre-registered P1
failed (7 of 8 budgets, excess growing to +0.10). That finding cannot be
trusted while the gate decisions underneath it are wrong on 4.5% of inputs.

**Scope.** Model-dependent:
* Qwen2.5-1.5B, Qwen2.5-3B -- T offset 0 on all 400 inputs. Their templates do
  not emit BOS, so those E8 cells are CLEAN and were not re-run.
* Mistral-7B -- T offset +1 on all 400. INVALID, re-run.
* Llama-3.1-8B -- same +1 offset; caught here before any E10 data was kept.

**Fix.** `adakv_matrix.py` now passes `add_special_tokens=False`. Verified: T
and D reproduce the released Mistral log **exactly** (dT=0, dD=0.00e+00 on 6/6
inputs).

**How it was found.** Only by adding the Llama control (E10). The Qwen cells
reproduced exactly and hid it; Mistral's drift was small enough to pass the
accuracy guards. It surfaced when Llama's T came out +1 against its released
log on every input.
