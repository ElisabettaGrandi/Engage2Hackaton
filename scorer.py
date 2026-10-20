"""
Scorer for the AWARE task-prioritisation challenge: top-3 hit rate.

For every sample the submission lists up to three task ids (row_index values), best first, separated by spaces.
A sample scores 1 if the labelled task is among the first three distinct ids, else 0. The score is the mean over
all samples (each sample counts equally).

Kaggle metric notebook: paste this file; Kaggle calls score(solution, submission, row_id_column_name="sample_id").
    solution:   sample_id, target, Usage    (target = row_index of the labelled task)
    submission: sample_id, prediction       (e.g. "12 3 7")

Local use:
    python scorer.py solution.csv submission.csv
    python scorer.py --labelled validation.csv submission.csv     (solution taken from a file with target_index)
"""
import argparse
import sys

import numpy as np
import pandas as pd

K = 3


class ParticipantVisibleError(Exception):
    pass


def _ids(value):
    """space-separated ids -> list of ints, duplicates removed, order kept"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out = []
    for tok in str(value).split():
        try:
            i = int(float(tok))
        except ValueError:
            raise ParticipantVisibleError(f"prediction contains a non-numeric id: {tok!r}") from None
        if i not in out:
            out.append(i)
    return out


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = "sample_id") -> float:
    if "prediction" not in submission.columns:
        raise ParticipantVisibleError(f"submission must have the columns {row_id_column_name}, prediction")
    if submission[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError(f"duplicate {row_id_column_name} values in the submission")
    sub = submission.set_index(row_id_column_name)["prediction"]
    missing = set(solution[row_id_column_name]) - set(sub.index)
    if missing:
        raise ParticipantVisibleError(f"{len(missing)} {row_id_column_name} values are missing from the submission, "
                                      f"e.g. {sorted(missing)[:5]}")
    hits = [int(int(t) in _ids(sub.loc[rid])[:K])
            for rid, t in zip(solution[row_id_column_name], solution["target"])]
    return float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solution", nargs="?", help="solution.csv (sample_id, target[, Usage])")
    ap.add_argument("submission", help="submission.csv (sample_id, prediction)")
    ap.add_argument("--labelled", help="instead of a solution file: a data file with target_index (e.g. validation.csv)")
    args = ap.parse_args()
    if args.labelled:
        lab = pd.read_csv(args.labelled, usecols=["sample_id", "row_index", "target_index"])
        sol = lab[lab["target_index"] == 1].rename(columns={"row_index": "target"})[["sample_id", "target"]]
    elif args.solution:
        sol = pd.read_csv(args.solution)
    else:
        sys.exit("give a solution file or --labelled <file>")
    sub = pd.read_csv(args.submission, dtype={"prediction": str})
    try:
        print(f"top-3 hit rate: {score(sol, sub, 'sample_id'):.4f}  ({len(sol)} samples)")
        if "Usage" in sol.columns:
            for usage, part in sol.groupby("Usage"):
                print(f"  {usage:8s} {score(part, sub[sub.sample_id.isin(part.sample_id)], 'sample_id'):.4f}  "
                      f"({len(part)} samples)")
    except ParticipantVisibleError as e:
        sys.exit(f"invalid submission: {e}")


if __name__ == "__main__":
    main()
