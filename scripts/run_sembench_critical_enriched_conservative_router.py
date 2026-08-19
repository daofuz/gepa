#!/usr/bin/env python3
"""Critical-enriched outcome router with conservative validation threshold."""

import run_sembench_outcome_pairwise_router as experiment
from run_sembench_critical_enriched_outcome_router import make_critical_enriched_split
from run_sembench_expanded_conservative_router import select_conservative_threshold


if __name__ == "__main__":
    experiment.make_split = make_critical_enriched_split
    experiment.select_threshold = select_conservative_threshold
    experiment.main()
