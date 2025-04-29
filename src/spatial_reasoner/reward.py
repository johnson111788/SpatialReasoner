"""Reward functions for GRPO training."""
import os
from datetime import datetime
import string
import copy as cp
import re


def can_infer_option(answer, choices):
    
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def build_choices(choices):
    ret = {}
    for option, choice in zip(['A', 'B', 'C', 'D'], choices):
        if choice is not None:
            ret[option] = choice
    return ret

def detect_not_matching(answer, choices):
    for option in choices:
        if answer.startswith(option + '.'):
            other_options = {k: v for k, v in choices.items() if k != option}
            return any(v in answer for v in other_options.values())
    return False


def accuracy_reward(completions, question, answer, question_index, A, B, C, D, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, q, sol, ind, a, b, c, d in zip(contents, question, answer, question_index, A, B, C, D):

        choices = build_choices([a, b, c, d])
        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        extracted_content = content_match.group(1).strip() if content_match else content.strip()
        option_not_matched = detect_not_matching(extracted_content, choices)
        if option_not_matched:
            return 0.0
        
        ret = can_infer(extracted_content, choices)
        reward = 1.0 if ret == sol else 0.0
        
        if reward == 0.0:
            if len(set(choices.values())) == 1:
                reward = 1.0
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Index: {ind}\n")
                f.write(f"Question: {q}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")

        rewards.append(reward)
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
