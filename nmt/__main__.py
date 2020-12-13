import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=""
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Set true to use GPU"
    )

    parser.add_argument(
        '--train',
        action="store_true",
        default=True,
        help="Set true to run training with given config"
    )

    parser.add_argument(
        "--train-src",
        type=str,
        help="Training data source location",
        default=None
    )

    parser.add_argument(
        "--train-tgt",
        type=str,
        help="Training data target location",
        default=None
    )

    parser.add_argument(
        "--dev-src",
        type=str,
        help="Validation/dev data source location",
        default=None
    )

    parser.add_argument(
        "--dev-tgt",
        type=str,
        help="Validation/dev data source location",
        default=None
    )

    parser.add_argument(
        "--max-epoch",
        type=int,
        help="Maximum epochs for training",
        default=30
    )

    parser.add_argument(
        "--vocab",
        type=str,
        help="Vocab location",
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed value for random states: default=42",
        default=42
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size: default=32",
        default=32
    )

    parser.add_argument(
        "--clip-grad",
        type=float,
        default=5.0,
        help="Value for gradient clipping: default=5.0"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate value: Default=0.001"
    )

    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.5,
        help="Learning rate decay: Default=0.5"
    )

    parser.add_argument(
        "--uniform-init",
        type=float,
        default=0.1,
        help="Uniform initialization value: Default=0.1"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Model save location"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="dropout probability value: Default=0.3"
    )

    parser.add_argument(
        "--use-chardecoder",
        action="store_true",
        default=False,
        help="If set true, NMT model will use char decoder"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    print(args)
