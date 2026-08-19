# Seeded from gepa_router_prompt_result_hybrid_half1_trainval50_openai_reflect.json, the current best held-out router baseline.
# EVOLVE-BLOCK-START
ROUTER_PROMPT = (
    "You are a hybrid router for MMLU-Pro computer science multiple-choice questions. "
    "Your task is to select exactly one model—nano or mini—to answer each question, optimizing for both cost and accuracy. "
    "Select nano only for the most obvious, safe, and simple questions. Escalate to mini for any question where nano is likely to fail but mini is likely to succeed. "
    "Critical misroutes—where nano is selected but is wrong while mini would be correct—are strongly penalized and must be avoided. "
    "If both are likely to succeed, select nano. If neither are likely to succeed, select mini. "
    "You may use deterministic Python code for the easiest obvious cases, otherwise you must decide. "
    "\n\n"
    "**Examples:**\n"
    "- 'What is the definition of a stack?' → nano\n"
    "- 'Which of the following is a property of a binary search tree?' → nano\n"
    "- 'What is the output of this simple loop?' → nano\n"
    "- 'Which of the following expressions is true if X but not Y?' (with logical conditions and/or Roman numerals) → mini\n"
    "- 'Cache simulation, matrix rank, unification, protocol edge case, or ambiguous NOT/EXCEPT wording' → mini\n"
    "\n"
    "**Routing Rules:**\n"
    "Default to nano for very short, clear definition/abbreviation/fact questions with no hard markers. "
    "Escalate to mini for anything involving:\n"
    "- Negations (not, except, none of the above, not enough information)\n"
    "- 'Which of the following'\n"
    "- Roman numerals (I., II., III.)\n"
    "- Advanced CS topics: cache, matrix, rank, unification, boolean, recursion, capacity, entropy, half-life, bayesian, vertex cover, clique, flow, chord, DHT, nmap\n"
    "- Complex logic, code with multiple statements, tricky edge cases, or ambiguous/long questions.\n"
    "- Any question where nano is likely to confidently answer incorrectly but mini is likely to succeed.\n"
    "\n"
    "**Python Routing Function:**\n"
    "The deterministic Python function route_easy_case(question, options, category=\"\", source=\"\") ONLY returns 'nano' for short, obvious, definition/abbreviation questions with no hard markers as above. In all other cases, return 'defer'.\n"
    "\n"
    "**Output:**\n"
    "Respond with exactly one token: nano or mini. Do not explain your answer or output anything else.\n"
    "\n"
    "**Summary:**\n"
    "- Use nano only for the safest, most obvious cases.\n"
    "- Escalate to mini for anything nontrivial, ambiguous, or risky.\n"
    "- Always output exactly one token: nano or mini."
)

def route_easy_case(question, options, category="", source=""):
    """Return 'nano' for obvious easy cases; return 'defer' for the router LLM."""
    q = question.lower()
    o = "\n".join(options).lower()
    text = q + "\n" + o

    # Escalate on any hard markers or complexity
    hard_markers = [
        "not", "except", "none of the above", "not enough information",
        "which of the following", "i.", "ii.", "iii.",
        "cache", "matrix", "rank", "unification", "boolean", "recurs",
        "capacity", "entropy", "half-life", "bayesian", "vertex cover",
        "maximum clique", "maximum flow", "clique", "flow", "chord", "dht", "nmap",
        "roman numeral", "subset", "superset", "recursion", "protocol", "deadlock", "thread", "schedule", "probability",
        "probabilit", "distribution", "bayes", "expectation", "variance", "graph", "edge", "tree traversal", "depth first", "breadth first"
    ]
    if any(marker in text for marker in hard_markers):
        return "defer"

    # Only allow nano for very short questions with definition/abbreviation markers
    easy_markers = [
        "what is", "what are", "define", "definition", "stands for",
        "abbreviation", "short for", "meaning of", "means", "simple loop", "repeat", "standard fact"
    ]
    # Avoid routing to nano for anything that looks like a code sample or has complex symbols
    code_like = [";", "{", "}", "def ", "class ", "lambda", "=", "==", "+=", "-=", "*=", "/=", "->", "import ", "print(", "return", "input("]
    if any(code in q for code in code_like) and not any(e in q for e in ["simple loop", "repeat"]):
        return "defer"

    # Only allow very short questions (strict threshold) to nano
    if len(question) < 120 and any(marker in text for marker in easy_markers):
        return "nano"

    return "defer"
# EVOLVE-BLOCK-END

def get_router_prompt():
    return ROUTER_PROMPT