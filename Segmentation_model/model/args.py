from typing import List
import argparse
from dataclasses import dataclass
from pathlib import Path


def parse_args(args: List[str]):

    parser = argparse.ArgumentParser(
        "Run model training", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--data_folder",
        type=Path,
        required=True,
        help="Path to the dataset folder",
    )

    parser.add_argument(
        "--output_folder",
        type=Path,
        required=True,
        help="Path to the logs and saved checkpoints",
    )

    return parser.parse_args(args)
