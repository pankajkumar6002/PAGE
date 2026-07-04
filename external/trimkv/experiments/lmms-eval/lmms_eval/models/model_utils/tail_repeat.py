# # TailRepeat loop-stop for Hugging Face generation
# # Works for greedy, sampling, and beam search.
from typing import Optional
import torch
from transformers import StoppingCriteria, StoppingCriteriaList


MASK = (1 << 64) - 1
BASE = 11400714819323198485  # 64-bit odd constant (Knuth's multiplicative hash)

class TailRepeatHashCriteria(StoppingCriteria):
    """
    Stop when the last p tokens repeat contiguously `repeats` times for some p in [1, pmax].
    Example: ... A B C A B C A B C  -> repeats=3, period p=3

    Note: StoppingCriteria stops the *entire* batch when any sequence triggers.
    If you need per-sequence stopping, run 1 sequence per batch or gate with EOS via a logits processor.
    """

    def __init__(self, repeats: int = 3, pmax: int = 16, eos_token_id: Optional[int] = None):
        assert repeats >= 2, "repeats must be >= 2"
        assert pmax >= 1, "pmax must be >= 1"
        self.pmin = 3
        self.repeats = repeats
        self.pmax = pmax
        self.eos_token_id = eos_token_id
        self.ended = None

    def _has_tail_loop(self, toks) -> bool:
        L = len(toks)
        max_p = min(self.pmax, L // self.repeats)
        if max_p == 0:
            return False

        # Only the trailing window can matter: size = repeats * pmax
        need = self.repeats * self.pmax
        tail = toks[-need:] if L > need else toks
        tail = tail.tolist()
        n = len(tail)

        # Precompute 64-bit rolling-hash prefix and powers over the tail
        pref = [0] * (n + 1)
        powB = [1] * (n + 1)
        for i in range(1, n + 1):
            powB[i] = (powB[i - 1] * BASE) & MASK
            # +1 to avoid zero contributing nothing
            pref[i] = ((pref[i - 1] * BASE) + (tail[i - 1] + 1)) & MASK

        def H(l: int, r: int) -> int:
            # hash of tail[l:r]
            return (pref[r] - (pref[l] * powB[r - l] & MASK)) & MASK

        # Try small periods first; most loops have tiny p (1..10)
        for p in range(self.pmin, min(self.pmax, n // self.repeats) + 1):
            end = n
            h_last = H(end - p, end)
            ok = True
            # Compare last block against the previous (repeats-1) blocks
            for k in range(2, self.repeats + 1 + max(0, 6 - p)):
                a = end - k * p
                b = end - (k - 1) * p
                if H(a, b) != h_last:
                    ok = False
                    break
            if ok:
                return True
        return False

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> bool:
        # `input_ids` shape: (batch, seq_len) or (batch*num_beams, seq_len)
        # Convert each row to python ints and check.
        # (Fast enough in practice since we examine only the trailing window.)
        if self.ended is None:
            self.ended = [False] * input_ids.shape[0]

        for row_idx in range(input_ids.shape[0]):
            if self.ended[row_idx]:
                continue
            loop = self._has_tail_loop(input_ids[row_idx])
            eos_hit = (input_ids[row_idx, -1].item() == self.eos_token_id)
            if loop:
                print(f"TailRepeatHashCriteria: Stopping due to tail repeat loop in row {row_idx}")

            self.ended[row_idx] = loop or eos_hit

        if all(self.ended):
            return True

        return False

class TailRepeatCriteria(StoppingCriteria):
    """
    Stop when the last p tokens repeat contiguously `repeats` times for some p in [1, pmax].
    Example: ... A B C A B C A B C  -> repeats=3, period p=3

    Note: StoppingCriteria stops the *entire* batch when any sequence triggers.
    This class tracks per-row `ended` and only returns True when all rows have ended.
    For per-sequence early-stop, run 1 sequence per batch or gate via a logits processor.
    """

    def __init__(self, repeats: int = 3, pmax: int = 16, eos_token_id: Optional[int] = None):
        assert repeats >= 2, "repeats must be >= 2"
        assert pmax >= 1, "pmax must be >= 1"
        self.repeats = int(repeats)
        self.pmax = int(pmax)
        self.eos_token_id = eos_token_id
        self.ended = None  # lazily sized to batch on first call

    @torch.no_grad()
    def _has_tail_loop_tensor(self, toks: torch.LongTensor) -> bool:
        # toks: 1D tensor on CPU or GPU, dtype long
        L = toks.size(0)
        max_p = min(self.pmax, L // self.repeats)
        if max_p == 0:
            return False

        # Only inspect a small trailing window: at most repeats * pmax tokens.
        n_tail = min(L, self.repeats * self.pmax)
        tail = toks[-n_tail:]  # still a tensor view

        # Try small periods first; loops are usually tiny (p in 1..10).
        # Each check uses a single tensor equality over a (repeats x p) view.
        for p in range(1, max_p + 1):
            seg_len = self.repeats * p
            seg = tail[-seg_len:]  # length seg_len, view
            # Shape into [repeats, p] without copies; unfold is robust to strides.
            blocks = seg.unfold(0, p, p)  # (repeats, p)
            last = blocks[-1]             # (p,)
            # Broadcast compare; .all().item() gives Python bool without sync surprises.
            if (blocks == last).all().item():
                return True
        return False

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,          # (batch or batch*num_beams, seq_len)
        scores: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> bool:
        batch = input_ids.size(0)
        if self.ended is None or len(self.ended) != batch:
            self.ended = [False] * batch

        # Quick EOS gate (vectorized) to avoid per-row work when possible.
        if self.eos_token_id is not None:
            eos_hit = (input_ids[:, -1] == self.eos_token_id)
            # Update ended flags; keep existing True values.
            for i in range(batch):
                if eos_hit[i].item():
                    self.ended[i] = True

        # Check only rows that haven't ended yet.
        # This avoids re-checking finished sequences at every step.
        for row_idx in range(batch):
            if not self.ended[row_idx]:
                self.ended[row_idx] = self._has_tail_loop_tensor(input_ids[row_idx])

        # Stop the whole batch only when all rows have ended,
        # matching the intended semantics from the original implementation.
        return all(self.ended)
