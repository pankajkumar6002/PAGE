# Pre-registration: B2b, multiple choice with retrieval during decoding

Written on 2026-11-07, after seeing partial results of B2 and before any B2b run. This is an added arm, not a replacement. B2's letter-only runs still finish and are reported.

## Why this exists

In B2's letter-only prompt on QuALITY with Qwen2.5-3B, predictions at b = 1/16 match the full-cache prediction on 99.4% of the first 176 inputs, although the kept fraction is 0.062 as intended. A one-token answer is decided by what prefill already wrote into the kept recent tokens, so post-prefill eviction cannot harm it much. That prompt therefore cannot test whether D predicts harm. It was a flaw in my design, and it could not have been seen without running it.

## Change

Same datasets, inputs, models, budgets and eviction as B2. Only the prompt tail and the decode length change:

> First quote the sentence from the text that best supports the answer. Then give the letter of the correct option on a final line in the form Answer: X. Keep it under 60 words.

- max_new_tokens = 128.
- Scoring: the last match of `Answer:` followed by a letter A to D, with optional parentheses. No match counts as wrong and as a parse failure, and the rate is reported per budget.

## Metrics

As B2: selection against a random gate at the same open rate at b = 1/16 and 1/8, bootstrapped by document, plus AUC(D) and AUC(a_early) against harm, for fixed tau and the z-gate.

## Predictions, recorded before running

- **P1.** Harm is clearly above zero at b = 1/16, unlike the letter-only arm.
- **P2.** Selection's interval includes zero on most (dataset, model) pairs.
- **P3.** AUC(D) against harm stays below 0.65 on most pairs.
- **P4.** At fixed tau the gate opens on few inputs on Qwen2.5-3B, as in the letter-only arm, so the z-gate is the informative one there.

## Reporting

Both arms are reported, and any prediction that fails is stated. If quoting inflates parse failures so much that accuracy is uninformative, that is reported rather than re-tuned.
