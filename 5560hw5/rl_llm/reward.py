import re

def compute_reward(text: str) -> float:
    has_answer = bool(re.search(r"answer:", text, re.IGNORECASE))
    has_reason = bool(re.search(r"reason:", text, re.IGNORECASE))
    has_conf   = bool(re.search(r"confidence:", text, re.IGNORECASE))

    count = sum([has_answer, has_reason, has_conf])

    if count == 3:
        return 1.0
    else:
        return -0.5