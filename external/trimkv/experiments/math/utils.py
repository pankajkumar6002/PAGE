import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DynamicCache

def estimate_max_batch_size(model, tokenizer, cache_creator, seq_len, min_seq_len=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # seq_len += 2048  # Add some buffer to the sequence length
    max_seq_len = max(seq_len + 512, min_seq_len)  # Ensure minimum sequence length is 1024
    max_seq_len = min(max_seq_len, tokenizer.model_max_length)

    best_batch_size = 64

    with torch.no_grad():
        # Prepare dummy input
        dummy_input_ids = torch.ones((1, max_seq_len), dtype=torch.long).to(device)
        while best_batch_size > 1:
            print(f"Testing batch size: {best_batch_size}")
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                input_ids = dummy_input_ids.repeat(best_batch_size, 1)
                attention_mask = torch.ones_like(input_ids)

                # Forward pass
                for _ in range(3):
                    past_key_values = cache_creator(model)
                    model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values)
                
                print(f"Batch size {best_batch_size} succeeded, peak memory: {torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB")
                torch.cuda.empty_cache()
                break  # If successful, exit the loop
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"Batch size {best_batch_size} failed due to OOM, reducing batch size.")
                best_batch_size //= 2
            except Exception as e:
                print(f"Unhandled exception: {e}")
                raise e

    print(f"Best batch size found: {best_batch_size}")
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory

    # Return in MiB as well for convenience
    peak_memory_mb = peak_memory_bytes / (1024 ** 2)
    total_memory_mb = total_memory_bytes / (1024 ** 2)

    # test if the model can handle the estimated batch size and benchmark it
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        # Measure peak memory during forward pass with estimated batch size
        with torch.no_grad():
            input_ids = torch.ones((best_batch_size, max_seq_len), dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids)
            past_key_values = cache_creator(model, max_seq_len=max_seq_len)
            model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values)

        est_peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        est_peak_memory_mb = est_peak_memory_bytes / (1024 ** 2)
        print(f"Estimated max batch size: {best_batch_size}, Peak memory: {est_peak_memory_mb:.2f} MB")
    except RuntimeError as e:
        print(f"RuntimeError during max batch size estimation: {e}")
        best_batch_size = (best_batch_size // 2) if best_batch_size > 1 else 1

    return {
        "sequence_length": max_seq_len,
        "peak_memory_MB": round(peak_memory_mb, 2),
        "total_gpu_memory_MB": round(total_memory_mb, 2),
        "estimated_max_batch_size": best_batch_size,
    }

