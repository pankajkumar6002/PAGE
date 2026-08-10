# 2x2 control: code path x hardware

| host | code path | mean $D$ (MK3) | AUC | rows |
|---|---|---:|---:|---:|
| jagannath (A100) | ORIGINAL | 0.0448 | 1.000 | 213 |
| jagannath (A100) | WRAPPER | 0.0300 | 1.000 | 350 |
| sateri (Ada) | ORIGINAL | 0.0412 | 1.000 | 350 |
| sateri (Ada) | WRAPPER | 0.0299 | 1.000 | 350 |
| paper (A100, released) | ORIGINAL | 0.0450 | 1.000 | -- |

**Code-path effect** (hardware held fixed)
- jagannath: original vs wrapper: **0.0148**
- sateri: original vs wrapper: **0.0113**

**Hardware effect** (code path held fixed)
- original: jagannath vs sateri: **0.0035**
- wrapper: jagannath vs sateri: **0.0001**

max code-path effect **0.0148**, max hardware effect **0.0035**

**VERDICT: the wrapper changes the statistic.** Discard the Qwen-14B, Llama-8B and Mistral-16K cells measured through it.
