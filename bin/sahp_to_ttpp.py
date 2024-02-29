#!/usr/bin/env python

# Converts SAHP training data to GNTPP format.

import argparse
import pickle
import torch
import numpy as np

from pathlib import Path

here = Path(__file__).parent


def convert_data(sahp_root, dataset, out_root):
    sub = ["train", "dev", "test"]
    total_number_items = 0
    event_types = set()

    for s in sub:
        path = f"{sahp_root}/data/{dataset}/{s}_manifold_format.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        event_types = event_types.union(set(np.unique(np.concatenate(data["types"]))))

    event_type_num = len(event_types)

    for s in sub:
        path = f"{sahp_root}/data/{dataset}/{s}_manifold_format.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        data["intervals"] = data["timeintervals"]
        del data["timeintervals"]
        data["t_max"] = np.concatenate(data["timestamps"]).max()
        data["event_type_num"] = event_type_num
        out_path = Path(
            f"{out_root}/{dataset}/{'val' if s == 'dev' else s}_manifold_format.pkl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    datasets = [
        "mimic",
        "retweet",
        "simulated",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("sahp")
    parser.add_argument(
        "datasets", default="all", nargs="+", choices=datasets + ["all"]
    )

    args = parser.parse_args()
    args.sahp = Path(args.sahp)
    assert args.sahp.exists()
    if "all" in args.datasets:
        args.datasets = datasets

    for d in args.datasets:
        print(f"Converting {d}.")
        converted = convert_data(args.sahp, d, Path("./data/"))

    print("Done.")
