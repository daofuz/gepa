"""Run SemBench Movie sentiment filtering through a nano-first cascade.

Example:
    python examples/cascaded_sembench_movie.py \
        --reviews ../sembench/files/movie/data/sf_2000/Reviews.csv \
        --checkpoint ../outputs/sembench_movie_sentiment_cascade_4tok_seed707.pt
"""

import argparse
from dataclasses import asdict

import pandas as pd

import lotus
from lotus.models import (
    CascadedLM,
    CPUSentimentDisagreementRouter,
    LM,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews",
        required=True,
        help="Path to SemBench Movie Reviews.csv",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a sentiment-disagreement checkpoint",
    )
    parser.add_argument("--cheap-model", default="gpt-5-nano")
    parser.add_argument("--strong-model", default="gpt-5-mini")
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-model-download", action="store_true")
    args = parser.parse_args()

    router = CPUSentimentDisagreementRouter(
        args.checkpoint,
        decision_threshold=args.decision_threshold,
        batch_size=args.batch_size,
        local_files_only=not args.allow_model_download,
    )
    model = CascadedLM(
        cheap_lm=LM(model=args.cheap_model),
        strong_lm=LM(model=args.strong_model),
        router=router,
    )
    lotus.settings.configure(lm=model)

    reviews = pd.read_csv(args.reviews).head(args.limit)
    if "reviewText" not in reviews or "reviewId" not in reviews:
        raise ValueError(
            "Reviews CSV must contain reviewText and reviewId columns"
        )

    positive = reviews.sem_filter(
        "The movie review has positive sentiment."
    )
    print(positive[["reviewId"]].head(5).to_string(index=False))
    print(asdict(model.cascade_stats))


if __name__ == "__main__":
    main()
