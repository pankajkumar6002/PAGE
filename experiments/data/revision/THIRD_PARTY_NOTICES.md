# Third-party code and data used by the revision experiments

No third-party data file is redistributed here. `fetch.sh` downloads every data file from its pinned
public source and checks it against `b8_raw/SHA256SUMS` and `leval/SHA256SUMS`.

Two runners contain short verbatim excerpts of third-party code, reproduced under the licences below.

## Lost in the Middle (Liu et al., TACL 2024)

- Source: github.com/nelson-liu/lost-in-the-middle, commit `29b8a6d042ce29abccee3db1a73171a107d7e6af`.
- Licence: MIT. Full text with the copyright notice: `licenses/lost-in-the-middle_MIT.txt`.
- Used in `experiments/scripts/revision/b8_neartie.py`: `normalize_answer` and `best_subspan_em` from
  `src/lost_in_the_middle/metrics.py`, verbatim. The QA prompt template `qa.prompt` is read from the
  downloaded file.
- Data fetched, not redistributed: `qa_data/30_total_documents/nq-open-30_total_documents_gold_at_{0,4,9,14,19,24,29}.jsonl.gz`
  (derived from Natural Questions).

## BABILong (Kuratov et al., NeurIPS 2024 Datasets and Benchmarks)

- Code source: github.com/booydar/babilong, commit `7a6efee29f5cac03c3c410e6799c80fd2ffe3610`.
- Licence: Apache License 2.0. Full text: `licenses/babilong_Apache-2.0.txt`.
- Used in `experiments/scripts/revision/b8_neartie.py`: `TASK_TEMPLATE`, `USER_TEMPLATE`, `DEFAULT_TEMPLATE`,
  `get_formatted_input` and the qa2 and qa3 entries of `DEFAULT_PROMPTS` (as `BABI_PROMPTS`) from
  `babilong/prompts.py`, and `preprocess_output`, `compare_answers` and the qa2/qa3 label set (as `BABI_LABELS`)
  from `babilong/metrics.py`. Modification: only the qa2 and qa3 parts are kept.
- Data fetched, not redistributed: huggingface.co/datasets/RMT-team/babilong, revision
  `ee0d588794c7ac098062ee0d247c733d62e94fe2`, `data/{qa2,qa3}/{4k,8k}.json`.

## L-Eval (An et al., ACL 2024)

- Data: huggingface.co/datasets/L4NLP/LEval, revision in `leval/COMMIT`, files `LEval/Exam/{tpo,quality,coursera,topic_retrieval_longchat,gsm100,sci_fi}.jsonl`.
  Licence: GPL-3.0. Fetched, not redistributed.
- `experiments/scripts/revision/b7_leval.py` uses L-Eval's task system prompts (from `Baselines/LEval_config.py`,
  github.com/OpenLMLab/LEval at `cd34b050`, GPL-3.0) and reimplements its exam scoring
  (`Evaluation/auto_eval.py`, `em.py`) so that predictions are scored as L-Eval scores them.
- `leval/SHA256SUMS` lists every L-Eval file as recorded before the runs. Only the six exam tasks above are fetched.

## Datasets loaded from the Hugging Face Hub at run time

`experiments/scripts/revision/b2_gating.py` (Table 47) loads these with `datasets.load_dataset`. None is stored
in this repository.

- QuALITY (Pang et al., NAACL 2022): `emozilla/quality`, validation split.
- LongBench-v2 (Bai et al., 2024): `zai-org/LongBench-v2`, Apache-2.0.
- LongBench (Bai et al., ACL 2024): `Xnhyacinth/LongBench`, MIT.
