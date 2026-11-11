#!/usr/bin/env python3
"""Table 45: runs the pre-registered analyze_b8.py unchanged and adds one to each two-sided bootstrap tail count before its Holm adjustment. Usage: python3 -I analyze_b8_corrected.py [results_dir] [logs_dir]"""
from pathlib import Path
import runpy


def corrected_p(original_p, replicates):
    tail_count = round(original_p * replicates / 2)
    return min(1.0, 2 * (tail_count + 1) / (replicates + 1))


def main():
    frozen = runpy.run_path(str(Path(__file__).with_name("analyze_b8.py")))
    original_boot = frozen["boot"]

    def boot(*args, **kwargs):
        estimate, interval, p, opened = original_boot(*args, **kwargs)
        return estimate, interval, corrected_p(p, frozen["REPS"]), opened

    original_boot.__globals__["boot"] = boot
    print("Post-hoc reporting correction (2026-11-11): two-sided bootstrap p = "
          "min(1, 2 * (min(n_nonpositive, n_nonnegative) + 1) / (R + 1)), "
          "followed by Holm adjustment. The preregistered script is unchanged. "
          "Point estimates and confidence intervals are unchanged.\n")
    frozen["main"]()


if __name__ == "__main__":
    main()
