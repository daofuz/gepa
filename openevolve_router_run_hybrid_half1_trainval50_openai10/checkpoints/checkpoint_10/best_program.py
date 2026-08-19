# Seeded from gepa_router_prompt_result_hybrid_half1_trainval50_openai_reflect.json, the current best held-out router baseline.
# EVOLVE-BLOCK-START
ROUTER_PROMPT = (
    "You are a hybrid router for MMLU-Pro computer science multiple-choice questions. "
    "Your job is to select exactly one model—nano or mini—to answer each question, maximizing the GEPA mean_score (risk_averse_utility). "
    "Select nano only for the most obvious, safe, and simple questions: short, clear definitions or standard facts. "
    "Escalate to mini for any question where nano is likely to fail but mini is likely to succeed, especially if the question is tricky, ambiguous, or involves advanced topics. "
    "Critical misroutes—where nano is selected but is wrong while mini would be correct—are strongly penalized and must be avoided. "
    "If both models are likely to succeed, select nano. If neither are likely to succeed, select mini. "
    "You may use deterministic Python code for the easiest, most obvious cases; otherwise, you must decide.\n\n"
    "**Examples:**\n"
    "- 'What is the definition of a stack?' → nano\n"
    "- 'What does FIFO stand for?' → nano\n"
    "- 'What is the time complexity of binary search?' → nano\n"
    "- 'Which of the following is a property of a binary search tree?' → nano\n"
    "- 'What is the output of this simple loop?' → nano\n"
    "- 'Which of the following expressions is true if X but not Y?' (with logical conditions and/or Roman numerals) → mini\n"
    "- 'Cache simulation, matrix rank, unification, protocol edge case, ambiguous NOT/EXCEPT wording, or multi-step code logic' → mini\n"
    "\n"
    "**Routing Rules:**\n"
    "Select nano ONLY for very short, clear definition, abbreviation, or straightforward fact questions with no complexity or tricky wording. "
    "Escalate to mini for:\n"
    "- Negations or exceptions (not, except, none of the above, not enough information, false, does not)\n"
    "- 'Which of the following' or similar phrases\n"
    "- Roman numerals (I., II., III., etc.)\n"
    "- Advanced CS topics: cache, matrix, rank, unification, boolean, recursion, protocol, concurrency, memory, deadlock, probability, entropy, capacity, flow, clique, DHT, nmap, scheduling, graph, tree traversal\n"
    "- Complex logic, multi-statement code, edge cases, ambiguous, or long questions\n"
    "- Any question where nano is likely to confidently answer incorrectly but mini is likely to succeed\n"
    "\n"
    "**Python Routing Function:**\n"
    "The deterministic Python function route_easy_case(question, options, category=\"\", source=\"\") ONLY returns 'nano' for very short, obvious, clear-cut definition/abbreviation/fact questions with no hard marker as above. In all other cases, return 'defer'.\n"
    "\n"
    "**Output:**\n"
    "Respond with exactly one token: nano or mini. Do not explain your answer or output anything else.\n"
    "\n"
    "**Summary:**\n"
    "- Use nano only for the safest, most obvious, short, and unambiguous cases.\n"
    "- Escalate to mini for anything nontrivial, ambiguous, risky, or potentially tricky.\n"
    "- Always output exactly one token: nano or mini."
)

def route_easy_case(question, options, category="", source=""):
    """Return 'nano' for obvious easy cases; return 'defer' for the router LLM."""
    q = question.lower()
    o = "\n".join(options).lower()
    text = q + "\n" + o

    # Escalate to mini on any hard markers, complexity, or ambiguity
    hard_markers = [
        "not", "except", "none of the above", "not enough information", "false", "does not",
        "which of the following", "select the", "choose the", "all of the following", "none of these",
        "i.", "ii.", "iii.", "iv.", "v.",
        "cache", "matrix", "rank", "unification", "boolean", "recurs", "recursion", "recursive",
        "capacity", "entropy", "half-life", "bayesian", "vertex cover", "maximum clique", "maximum flow",
        "clique", "flow", "chord", "dht", "nmap", "protocol", "deadlock", "thread", "schedule", "scheduling",
        "probability", "probabilit", "distribution", "bayes", "expectation", "variance", "random", "markov",
        "graph", "edge", "vertex", "tree traversal", "depth first", "breadth first", "heap", "bfs", "dfs",
        "memory", "pointer", "address", "stack overflow", "segmentation fault", "race condition", "mutex", "lock"
    ]
    if any(marker in text for marker in hard_markers):
        return "defer"

    # Only allow nano for very short, simple definition/abbreviation/fact questions
    easy_markers = [
        "what is", "what are", "define", "definition", "stands for",
        "abbreviation", "short for", "meaning of", "means", "is the result of", "is true for", "simple loop", "standard fact"
    ]
    # Avoid nano for code samples or anything with complex symbols, unless it's a specifically simple loop or "repeat"
    code_like = [
        ";", "{", "}", "def ", "class ", "lambda", "=", "==", "+=", "-=", "*=", "/=", "->",
        "import ", "print(", "return", "input(", "while ", "for ", "if ", "else", "elif", "try:", "except"
    ]
    if any(code in q for code in code_like) and not ("simple loop" in q or "repeat" in q):
        return "defer"

    # Only allow very short, direct questions (strict threshold) to nano
    if len(question) < 90 and any(marker in text for marker in easy_markers):
        return "nano"

    # Edge: also allow nano for the most common acronym/abbreviation questions
    acronym_phrases = [
        "stand for", "stands for", "abbreviation of", "short for"
    ]
    if len(question) < 80 and any(phrase in q for phrase in acronym_phrases):
        return "nano"

    return "defer"
# EVOLVE-BLOCK-END

def get_router_prompt():
    return ROUTER_PROMPT