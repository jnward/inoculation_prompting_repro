"""Shared plot utilities for gradient routing experiments."""

import glob
import json
import os
import zipfile


def read_eval_log(path):
    """Read .eval ZIP header.json -> flat {scorer/metric: value} dict.

    If *path* is a directory, pick the latest .eval file inside it.
    """
    if os.path.isdir(path):
        candidates = sorted(glob.glob(os.path.join(path, "*.eval")))
        if not candidates:
            return {}
        path = candidates[-1]

    try:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("header.json") as f:
                log_data = json.load(f)
    except (zipfile.BadZipFile, json.JSONDecodeError, FileNotFoundError, KeyError):
        return {}

    metrics = {}
    for score_group in log_data.get("results", {}).get("scores", []):
        scorer_name = score_group.get("name", "")
        for metric_name, metric_obj in score_group.get("metrics", {}).items():
            value = metric_obj.get("value")
            if value is not None:
                metrics[f"{scorer_name}/{metric_name}"] = value
    return metrics


def pareto_frontier(points):
    """Given list of (x, y, ...) tuples, return indices on the Pareto frontier.

    We want to maximize x and minimize y. A point dominates another if it has
    higher x AND lower y. The frontier consists of all non-dominated points.
    Sort by x ascending, then sweep keeping running min of y.
    """
    indexed = sorted(enumerate(points), key=lambda t: t[1][0])
    frontier = []
    min_y = float("inf")
    # Walk from highest x to lowest to find non-dominated points
    for idx, pt in reversed(indexed):
        if pt[1] <= min_y:
            min_y = pt[1]
            frontier.append(idx)
    frontier.reverse()
    return frontier
