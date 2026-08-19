#!/usr/bin/env python3
"""Stable entry point for the locked untouched-test evaluation."""

import run_sembench_untouched_test as evaluation


_run_one = evaluation.experiment.run_one


def run_one_with_locked_margin(train, validation, test, seed, pairwise_weight, args):
    args.margin = 1.0
    return _run_one(train, validation, test, seed, pairwise_weight, args)


if __name__ == "__main__":
    evaluation.experiment.run_one = run_one_with_locked_margin
    evaluation.main()
