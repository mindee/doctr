# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Formula recognition latency benchmark"""

import argparse
import time

import numpy as np
import torch

from doctr.models import formula


@torch.inference_mode()
def main(args):
    device = torch.device("cuda:0" if args.gpu else "cpu")

    model = (
        formula
        .__dict__[args.arch](
            pretrained=args.pretrained,
            pretrained_backbone=False,
            input_shape=(3, *args.size),
        )
        .eval()
        .to(device=device)
    )
    # The decoding time grows with the number of tokens: decode a fixed number of them (an untrained model rarely
    # stops on its own, a trained one stops at the end of each formula)
    model.max_decode_length = args.tokens

    # Input
    img_tensor = torch.rand((args.batch_size, 3, *args.size)).to(device=device)

    # Warmup
    for _ in range(10):
        _ = model(img_tensor)

    timings = []

    # Evaluation runs
    for _ in range(args.it):
        start_ts = time.perf_counter()
        _ = model(img_tensor)
        if args.gpu:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start_ts)

    _timings = np.array(timings)
    print(
        f"{args.arch} ({args.it} runs on {tuple(args.size)} inputs in batches of {args.batch_size}, "
        f"up to {args.tokens} tokens)"
    )
    print(f"mean {1000 * _timings.mean():.2f}ms, std {1000 * _timings.std():.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="docTR latency benchmark for formula recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("arch", type=str, help="Architecture to use")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="The batch_size")
    parser.add_argument(
        "--size", type=int, nargs=2, default=(256, 768), metavar=("H", "W"), help="The image input size (H, W)"
    )
    parser.add_argument("--tokens", type=int, default=128, help="Maximum number of tokens to decode")
    parser.add_argument("--gpu", dest="gpu", help="Should the benchmark be performed on GPU", action="store_true")
    parser.add_argument("--it", type=int, default=100, help="Number of iterations to run")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    args = parser.parse_args()

    main(args)
