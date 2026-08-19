# Seeded from gepa_router_prompt_result_hybrid_half1_ablation50_balancedish_openai_reflect.json; use only with the balancedish train/val split.
# EVOLVE-BLOCK-START
ROUTER_PROMPT = (
    "You are a cost-and-risk-aware router for MMLU-Pro computer science multiple-choice questions. "
    "For each question, select exactly one model: nano or mini. "
    "Your objective is to maximize mean risk-averse utility (GEPA): critical misroutes (nano is wrong, mini would be right) are heavily penalized; unnecessary use of mini is lightly penalized. "
    "Route as follows:\n"
    "\n"
    "1. Select nano ONLY when the question is a short, direct definition, acronym/abbreviation, or uncontested standard fact, OR a single-step, unambiguous, non-nested code tracing problem. No subtlety, ambiguity, negation, or advanced topic markers can be present.\n"
    "2. Select mini for any question involving: negation (not, except, none), ambiguous wording, 'which of the following', Roman numerals (I., II., III.), multi-step reasoning, protocol/architecture distinctions, malware/security types, cache/matrix/rank/entropy/boolean/unification/recursion/graph theory, or if options are closely related or require subtle discrimination.\n"
    "3. If unsure, or if any marker of subtlety, ambiguity, or advanced topic is present, select mini to avoid critical misroutes.\n"
    "4. If both models would be correct, prefer nano (for cost) but only when highly confident.\n"
    "\n"
    "Examples:\n"
    "- 'What is the definition of a stack?' → nano\n"
    "- 'FIFO stands for...' → nano\n"
    "- 'Which of the following is NOT a property of TCP?' → mini\n"
    "- 'How many independent parameters in this Bayesian network?' → mini\n"
    "- 'A/an ________ is a program that steals your logins & passwords.' (with options: virus, worm, trojan, spyware) → mini\n"
    "- 'Which of the following is not an example of presentation layer issues?' → mini\n"
    "- 'The IP protocol is primarily concerned with...' → nano\n"
    "\n"
    "Markers requiring mini: 'not', 'except', 'none of the above', 'not enough information', 'which of the following', Roman numerals (I., II., III.), 'ambiguous/closely related options', 'cache', 'matrix', 'rank', 'unification', 'boolean', 'recurs', 'capacity', 'entropy', 'half-life', 'Bayesian', 'vertex cover', 'maximum clique', 'maximum flow', 'chord', 'DHT', 'nmap', 'protocol', 'layer', 'coherence', 'security', 'malware', 'virus', 'worm', 'trojan', 'spyware', or any subtle reasoning or edge-case topic.\n"
    "\n"
    "Do NOT select nano if there is any negation, ambiguity, subtlety, or advanced/edge-case reasoning, even if the question appears simple or factual. If in doubt, always select mini.\n"
    "\n"
    "Respond with exactly one token: nano or mini."
)

def route_easy_case(question, options, category="", source=""):
    """Return 'nano' for clear, very short, unambiguous definition/fact/abbreviation/basic code tracing only; else 'defer'."""
    text = (question + "\n" + "\n".join(options)).lower()
    qlen = len(question)

    # High-risk markers: anything that would require mini
    hard_markers = [
        "not", "except", "none of the above", "not enough information", "which of the following",
        "i.", "ii.", "iii.",
        "cache", "matrix", "rank", "unification", "boolean", "recurs", "capacity", "entropy",
        "half-life", "bayesian", "vertex cover", "maximum clique", "maximum flow", "chord", "dht", "nmap",
        "protocol", "layer", "coherence", "security", "malware", "virus", "worm", "spyware", "trojan"
    ]
    if any(marker in text for marker in hard_markers):
        return "defer"

    # Only allow nano for very short, clear, unambiguous definition/fact/abbreviation/basic code tracing
    easy_markers = [
        "what is", "what are", "define", "definition", "stands for", "abbreviation",
        "full form of", "acronym", "expands to", "short for", "means",
        "fifo", "lifo", "ram", "rom", "cpu", "alu", "simple loop", "repeat",
        "standard fact"
    ]
    direct_code_words = ["stack", "queue", "array", "linked list", "tree", "graph", "hash", "pointer"]
    # If it's an abbreviation or direct definition, and the question is short and not a fill-in-the-blank with ambiguity
    if (
        qlen < 120
        and any(marker in text for marker in easy_markers)
        and not ("____" in question or "blank" in question)
        and not any(word in text for word in hard_markers)
    ):
        return "nano"
    # Explicit: direct code tracing (very simple, non-nested)
    if (
        qlen < 70
        and ("trace the output" in text or "output of the following" in text)
        and not any(word in text for word in hard_markers)
    ):
        return "nano"
    # Super-short, terminology-only
    if (
        qlen < 50
        and any(word in question.lower() for word in direct_code_words)
        and not any(word in text for word in hard_markers)
    ):
        return "nano"

    return "defer"
# EVOLVE-BLOCK-END

def get_router_prompt():
    return ROUTER_PROMPT