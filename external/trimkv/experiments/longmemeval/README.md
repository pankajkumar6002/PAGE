
# Notes from TRIMKV's authors

We observed that running SnapKV with this R-KV codebase can lead to unexpected behavior. We later traced this to a bug in the DynamicCache implementation used by R-KV. For our experiments, we therefore switched to the SnapKV implementation from MInference. The results reported in our paper for these experiments are based on that implementation and should be correct.

If you want to use MInference with Qwen3, you’ll need to update the MInference package to support Qwen3’s attention implementation. The simplest change is to add q_norm and k_norm into the attention computation in this file:
https://github.com/microsoft/MInference/blob/fc1c63f595a0143e47cf6af4bbb729bc31198ee9/minference/modules/forward.py#L61

Check whether the model has q_norm and k_norm attributes, and if so, apply them to q and k before computing the attention scores.

We currently do not have bandwidth to rerun all experiments to ensure R-KV works flawlessly with SnapKV, so the R-KV + SnapKV combination is provided as is. If you plan to use SnapKV with R-KV, please keep this limitation in mind.

In our LongBench evaluation, we fixed the bug by correcting the `get_seq_length()` function in R-KV’s DynamicCache implementation. You may apply a similar fix in your setup.

### Running the Code

We provide an example script in the scripts folder to help you get started.