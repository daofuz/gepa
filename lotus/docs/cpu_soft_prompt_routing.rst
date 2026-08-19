CPU soft-prompt routing
=======================

``RoutedLM`` inserts a local router between Lotus semantic operators and their
language-model calls. Each rendered row is routed independently, while Lotus's
existing prompt construction and output parsing remain unchanged.

Installation
------------

Install Lotus and the optional router dependencies::

    pip install -e .
    pip install -r requirements-softprompt-router.txt

Configuration
-------------

The router checkpoint must contain ``model_state_dict`` and the training ``args``
mapping produced by ``train_softprompt_router.py``. Class 0 is the cheap route and
class 1 is the strong route. If the checkpoint contains ``routes`` metadata,
``CPUSoftPromptRouter`` uses it to recover the class ordering. It also uses the
top-level ``selected_decision_threshold`` written by threshold tuning::

    import lotus
    from lotus.models import CPUSoftPromptRouter, LM, RoutedLM

    router = CPUSoftPromptRouter(
        checkpoint_path="path/to/softprompt-router.pt",
        decision_threshold=0.45,
        batch_size=8,
        cpu_threads=8,
    )

    routed_lm = RoutedLM(
        cheap_lm=LM(model="gpt-4.1-nano"),
        strong_lm=LM(model="gpt-4.1-mini"),
        router=router,
    )
    lotus.settings.configure(lm=routed_lm)

    result = df.sem_filter("{text} is relevant to the query")
    print(routed_lm.routing_stats)
    print(routed_lm.last_routing_decisions)

Checkpoints may declare ``router_text_format`` metadata when training used a
specific rendering of the Lotus row. ``raw_messages_v1`` remains the default;
``sembench_movie_review_v1`` extracts the filter context and recreates the
Movie training prompt before tokenization.

All semantic operators that call the configured ``lm`` use the router. This
includes ``sem_filter`` and ``sem_map`` and applies to future operators such as
``sem_classify`` when they use the same LM interface.

Semantic SQL / SemBench-style workloads
---------------------------------------

Semantic SQL predicates map naturally to Lotus operators. For example, the
SemBench Movie Q1 predicate ``WHERE AI.IF(review is clearly positive)`` is the
following routed ``sem_filter``::

    reviews = pd.read_csv("Reviews.csv")
    positive = reviews.sem_filter("The movie review has positive sentiment.")
    result = positive[["reviewId"]].head(5)

The exported Movie checkpoint learned routing outcomes for ``gpt-5-nano`` and
``gpt-5-mini``. Use those models as the cheap and strong branches respectively;
changing the child models changes the routing target and requires new
validation or retraining.

See ``examples/routed_sembench_movie.py`` for a command-line
example that loads the SemBench CSV, configures both LMs, runs the semantic
predicate, and prints routing rates.

Nano-first cascade
------------------

``CascadedLM`` calls the cheap model for every row, passes its True/False or
POSITIVE/NEGATIVE answer to ``CPUSentimentDisagreementRouter``, and retries only
high-disagreement rows with the strong model. ``cascade_stats`` reports both the
accepted cheap answers and strong retries. This is a two-stage policy, so total
LM calls equal the input row count plus the number of escalations.

See ``examples/cascaded_sembench_movie.py`` for the Movie cascade and
``examples/routed_sembench_movie.py`` for direct pre-call routing.


The default ``non_text_policy="strong"`` sends image, audio, and other
non-text inputs directly to the strong model. A text-trained router should not
make cheap/strong decisions for modalities it has never seen. Use
``non_text_policy="route_text"`` only with a router validated for that behavior.

Failure behavior
----------------

By default, a router exception sends the affected batch to the strong model. Set
``router_error_policy="raise"`` on ``RoutedLM`` or ``CascadedLM`` to surface router failures.

The strong LM supplies token counting, context limits, and model-specific output
postprocessing for the wrapper. The cheap and strong models should therefore use
compatible prompt and output formats.

Training-domain warning
-----------------------

A router only generalizes to workloads represented by its training data. A
checkpoint trained on multiple-choice computer-science questions is suitable for
runtime integration tests, but should be retrained or validated before routing
arbitrary Lotus filter or classification workloads.

Likewise, a checkpoint trained with the cheap model's answer included in the
router input is not a direct pre-call router and must not be used with
``RoutedLM``. Train the checkpoint on the rendered task/row text alone, matching
``messages_to_router_text``.
