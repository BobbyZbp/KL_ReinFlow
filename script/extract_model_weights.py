"""
Extract model weights from full checkpoints (which include replay buffers).

Full checkpoints are 300-400MB due to replay buffers. This script extracts just
the 'model' state_dict (~4.5MB) into separate files for lightweight analysis.

Usage:
    conda run -n reinflow python script/extract_model_weights.py --run A2
    conda run -n reinflow python script/extract_model_weights.py --run B
    conda run -n reinflow python script/extract_model_weights.py --run A2 --iters 0,20000,40000,60000,80000

Output: log/gym/finetune/{run}/checkpoint/model_only/state_{iter}.pt
"""
import argparse
import os
import sys
import gc

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN_MAP = {
    "A2": "perstep_runA2_baseline",
    "B": "ablation_runB_reward_penalty",
    "5b": "perstep_run5b_exact_highkl",
    "7": "perstep_run7_exact_ema",
    "6": "perstep_run6_hutchinson",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--iters", type=str, default=None,
                        help="Comma-separated iters (default: all available)")
    args = parser.parse_args()

    run_dir = RUN_MAP[args.run]
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        f"log/gym/finetune/{run_dir}/checkpoint"
    )
    out_dir = os.path.join(base, "model_only")
    os.makedirs(out_dir, exist_ok=True)

    if args.iters is None:
        available = sorted([
            int(f.replace("state_", "").replace(".pt", ""))
            for f in os.listdir(base)
            if f.startswith("state_") and f.endswith(".pt") and f != "latest.pt"
        ])
    else:
        available = [int(x) for x in args.iters.split(",")]

    print(f"Run: {args.run} ({run_dir})")
    print(f"Extracting {len(available)} checkpoints to {out_dir}/")

    for it in available:
        src = os.path.join(base, f"state_{it}.pt")
        dst = os.path.join(out_dir, f"state_{it}.pt")

        if os.path.exists(dst):
            print(f"  iter {it}: already extracted, skipping")
            continue

        if not os.path.exists(src):
            print(f"  iter {it}: source not found, skipping")
            continue

        src_size = os.path.getsize(src) / 1e6
        print(f"  iter {it}: loading ({src_size:.0f} MB)...", end=" ", flush=True)

        data = torch.load(src, map_location="cpu", weights_only=False)
        model_state = data["model"]
        # Save only model weights + iteration number
        torch.save({"itr": data["itr"], "model": model_state}, dst)
        del data
        gc.collect()

        dst_size = os.path.getsize(dst) / 1e6
        print(f"saved ({dst_size:.1f} MB)")

    print(f"\nDone. Use --device cpu with diagnostic scripts (they'll look for model_only/ first).")


if __name__ == "__main__":
    main()
