import argparse
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import math

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

from pararel.eval_on_fact_recall_set.generate_plots import (
    create_plots
)

def main(args):
    kind = "mlp"
    data = pd.read_json(args.query_file, lines=args.query_file.endswith(".jsonl"))
    print(f"The data contains {len(data)} entries. Generating results for the first 1000...")
    print()
    data = data.iloc[:1000]
    count = len(data)
    
    create_plots(data, kind, count, args.arch, args.archname, args.savefolder)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--query_file",
        required=True,
        type=str,
        help="File with combined queries to process, with corresponding CT results folders.",
    )
    argparser.add_argument(
        "--savefolder",
        required=True,
        type=str,
        help="Folder to save plot results to.",
    )
    argparser.add_argument(
        "--arch",
        required=True,
        type=str,
    )
    argparser.add_argument(
        "--archname",
        required=True,
        type=str,
    )
    args = argparser.parse_args()

    print(args)
    main(args)