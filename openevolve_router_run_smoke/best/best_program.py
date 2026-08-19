# EVOLVE-BLOCK-START
ROUTER_PROMPT = """You are a cost-aware model router for MMLU-Pro computer science questions.

Choose exactly one model:
- nano: cheaper and preferred whenever it is likely to preserve accuracy.
- mini: more capable and more expensive; choose it only when the question is likely to expose nano's weakness.

Routing rules:
1. Default to nano for straightforward definitions, standard facts, simple code tracing, and basic algorithm questions.
2. Choose mini for questions with multi-step calculation, tricky formal logic, subtle architecture/protocol distinctions, dense reasoning, or hidden traps where nano may confidently pick the wrong option.
3. If both models are likely to answer correctly, choose nano to save cost.
4. If neither model is likely to answer correctly, choose mini under this objective; the router should avoid under-escalating difficult failure-prone cases.

Examples:
- Basic definition or common concept -> nano
- Simple loop count or direct AP-style CS concept -> nano
- Cache simulation, matrix/rank trick, unification, or precise protocol edge case -> mini
- Ambiguous wording with a subtle NOT/EXCEPT distinction -> mini

Respond with exactly one token: nano or mini."""


def route_easy_case(question, options, category="", source=""):
    """Return 'nano' for obvious easy cases; return 'defer' for the LLM router."""
    text = (question + "\n" + "\n".join(options)).lower()

    hard_markers = [
        "not",
        "except",
        "none of the above",
        "not enough information",
        "which of the following",
        "i.",
        "ii.",
        "iii.",
        "cache",
        "matrix",
        "rank",
        "unification",
        "boolean",
        "recurs",
        "capacity",
        "entropy",
        "half-life",
        "bayesian",
        "vertex cover",
        "maximum clique",
        "maximum flow",
        "chord",
        "dht",
        "nmap",
    ]
    if any(marker in text for marker in hard_markers):
        return "defer"

    easy_markers = [
        "what is",
        "what are",
        "define",
        "definition",
        "stands for",
        "abbreviation",
        "simple loop",
        "repeat",
        "standard fact",
    ]
    if len(question) < 180 and any(marker in text for marker in easy_markers):
        return "nano"

    return "defer"
# EVOLVE-BLOCK-END


def get_router_prompt():
    return ROUTER_PROMPT
