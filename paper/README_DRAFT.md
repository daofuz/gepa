# ScoutRoute VLDB draft

This directory contains an English VLDB draft assembled from the experiments currently recorded in the repository. It uses the official `acmart` / `sigconf` template and is organized as an **Experiment, Analysis & Benchmark** submission.

## Entry point and structure

- Main file: `main.tex`
- Bibliography: `main.bib`
- Sections: `sections/`

The draft includes an abstract, introduction, Prompt Optimization Overview, a methodology section targeted at roughly 1.5 pages, an experiment section targeted at roughly 5 pages, related work, limitations, and conclusion. The experiment section intentionally leaves future work visible and never describes post-hoc fresh-set analysis as untouched evaluation.

## Build

Recommended:

```text
latexmk -pdf main.tex
```

Fallback:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The build was verified locally with MiKTeX 25.12 (`latexmk` requires Perl; the fallback sequence works without it). The current build output is `main.pdf`: ten letter-size pages, with resolved citations/references and no overfull boxes in the final log.

## Required before submission

1. Replace the placeholder author, affiliation, and email.
2. Add the artifact URL and provide a complete reproducibility package as required for the EAB track.
3. Recheck the rendered section lengths after future edits; the current ten-page draft is intentionally shorter than a complete submission because several experiments remain open.
4. Add at least one cross-operator or cross-dataset replication, confidence intervals or paired tests, and end-to-end latency and realized token cost.
5. Preserve the distinction among locked fresh, historical test, and post-hoc fresh analyses.
6. Recheck the rules for the target cycle. This draft follows the [VLDB 2027 submission guidelines](https://www.vldb.org/2027/submission-guidelines.html) and [formatting guidelines](https://www.vldb.org/2027/formatting-guidelines.html).

## Scope note

Movie uses cached GPT-5 Nano/Mini outcomes. Computer Science QA uses GPT-4.1 Nano/Mini outcomes. The two primary methodological contributions are observable active label acquisition and a cost-derived route score. The primary router is a lightweight frozen-encoder soft prompt with 3,074--13,826 trainable parameters; discrete GEPA is an independent, larger comparison rather than a stage of that method. The acquisition policy runs both candidate models over an unlabeled pool, prioritizes their prediction disagreements while retaining agreement coverage, and requests human gold labels only after selection. A correct selected route receives `1 / expected_route_cost`; an incorrect route receives zero. Existing Movie rows labeled regret CE and fixed-utility GEPA are retained as legacy diagnostics and require a matched cost-score rerun. The direct pre-router never observes Nano's generated answer. The answer-aware cascade is exploratory and cannot be compared with direct pre-routing without accounting for its mandatory Nano call.
