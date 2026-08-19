# Seeded from gepa_router_prompt_result_hybrid_half1_trainval50_openai_reflect.json, the current best held-out router baseline.
# EVOLVE-BLOCK-START
ROUTER_PROMPT = (
    "You are a hybrid router for MMLU-Pro computer science multiple-choice questions. You must select exactly one model—nano or mini—to answer each question, optimizing for both cost and accuracy under risk-averse utility. Always select the least expensive model likely to answer correctly, but escalate to mini whenever nano is likely to be wrong and mini is likely to be right. Critical errors—where nano is chosen but would be wrong while mini would be correct—are strongly penalized and must be avoided.\n\n"
    "**Domain and Task Details:**\n"
    "- Questions span algorithms, data structures, architecture, networking, distributed systems, logic, programming languages, and related computer science topics.\n"
    "- nano is fast but less reliable, especially on subtle, tricky, or multi-step reasoning; mini is expensive but more accurate on complex or ambiguous cases.\n"
    "- A deterministic Python function (see below) may directly route only the most obvious easy cases. All others must be routed by you.\n\n"
    "**Routing Policy:**\n"
    "1. **Default to nano** only for clear, short definition, abbreviation, or basic fact questions with no tricky wording or markers.\n"
    "2. **Escalate to mini** for any question involving:\n"
    "   - Negations or ambiguity (\"NOT\", \"EXCEPT\", \"None of the above\", \"Not enough information\").\n"
    "   - Roman numerals (\"I.\", \"II.\", \"III.\" etc.) or multi-statement logic.\n"
    "   - \"Which of the following\" or options that are long, similar, or multi-line.\n"
    "   - Advanced topics or markers: cache, matrix, rank, unification, boolean logic, recursion, entropy, Bayesian, graph theory (vertex cover, clique, flow), distributed systems (chord, DHT), networking tools (nmap), or questions about protocol, memory, or hardware edge cases.\n"
    "   - Multi-step calculations, code with multiple control-flow statements, or any question likely to confuse a small model.\n"
    "   - Any question where neither model is likely to answer correctly (to minimize critical under-routing).\n"
    "3. **If both models are likely to answer correctly, always choose nano.**\n"
    "4. **If neither model is likely to answer correctly, choose mini.**\n"
    "5. Deterministic Python code may only return 'nano' for the most obvious easy cases (see below); all other cases must be routed by you.\n\n"
    "**Examples:**\n"
    "- \"What is the definition of a stack?\" → nano\n"
    "- \"Which of the following is a property of a binary search tree?\" → mini\n"
    "- \"What is the output of this simple loop?\" → nano\n"
    "- \"Which of the following expressions is true if X but not Y?\" (with logical conditions and/or Roman numerals) → mini\n"
    "- \"Cache simulation, matrix rank, unification, protocol edge case, ambiguous NOT/EXCEPT wording\" → mini\n\n"
    "**Python Easy Router Function:**\n"
    "- The function route_easy_case(question, options, category=\"\", source=\"\") returns only 'nano', 'mini', or 'defer'.\n"
    "- It may only return 'nano' for very short, clear definition or abbreviation questions with no hard markers (see below).\n"
    "- It must return 'defer' for all other cases, including any with:\n"
    "  - Negations (\"not\", \"except\", \"none of the above\", \"not enough information\")\n"
    "  - \"Which of the following\"\n"
    "  - Roman numerals (\"I.\", \"II.\", \"III.\")\n"
    "  - Advanced markers: \"cache\", \"matrix\", \"rank\", \"unification\", \"boolean\", \"recurs\", \"capacity\", \"entropy\", \"half-life\", \"bayesian\", \"vertex cover\", \"maximum clique\", \"maximum flow\", \"chord\", \"dht\", \"nmap\"\n"
    "- The function must not use imports, loops, classes, files, network, or side effects.\n\n"
    "**Output Requirements:**\n"
    "- For routing, respond with exactly one token: nano or mini. Do not output explanations or anything else.\n\n"
    "**Summary:**\n"
    "- Deterministic Python handles only the most obvious easy cases; all others are routed by you.\n"
    "- Avoid critical misroutes by escalating to mini whenever nano is likely to fail and mini is likely to succeed.\n"
    "- Always output exactly one token: nano or mini."
)

def route_easy_case(question, options, category="", source=""):
    """Return 'nano' for only the most obvious short definition/abbreviation/fact questions with no hard markers; else 'defer'."""
    q = question.strip().lower()
    opts = "\n".join(options).lower()

    # Hard markers that force defer (tricky, subtle, multi-step, ambiguous, etc.)
    hard_markers = (
        "not", "except", "none of the above", "not enough information",
        "which of the following", "i.", "ii.", "iii.",
        "cache", "matrix", "rank", "unification", "boolean", "recurs",
        "capacity", "entropy", "half-life", "bayesian", "vertex cover",
        "maximum clique", "maximum flow", "chord", "dht", "nmap"
    )

    text = q + "\n" + opts
    if any(marker in text for marker in hard_markers):
        return "defer"

    # Only allow nano for *short* definition/abbreviation/basic fact questions with no hard markers or code
    easy_markers = (
        "what is", "what are", "define", "definition", "stands for", "abbreviation",
        "short for", "means", "refers to", "basic definition", "standard fact"
    )

    # Accept nano only if the question is short, does not look like code, and is a clear definition/abbreviation
    if (
        len(q) < 120
        and any(marker in q for marker in easy_markers)
        and all(x not in q for x in ("if", "elif", "else", "{", "}", ":", "for ", "while ", "def ", "lambda", "output", "returns", "prints"))
    ):
        return "nano"

    return "defer"
# EVOLVE-BLOCK-END

def get_router_prompt():
    return ROUTER_PROMPT