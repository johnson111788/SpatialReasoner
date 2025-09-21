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

def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,|Therefore,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]

# Precompile regex patterns
phrases_dict = {
    "location": re.compile(r"3D location of .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)", flags=re.IGNORECASE), # The 3D location of the airplane can be estimated as (1.50, 1.01, 22.49)
    "vector": re.compile(r"vector from .*? to .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)", flags=re.IGNORECASE), # vector/distance of the airplane from the camera. Prompt modified

    "dist_from": re.compile(r"distance from .*? to the camera .*? \d+\.\d+", flags=re.IGNORECASE),
    "dist_between": re.compile(r"distance between .*? \d+\.\d+", flags=re.IGNORECASE),
    
    "cosine": re.compile(r"cosine similarity between .*? -?\d+\.\d+", flags=re.IGNORECASE),
    "angle": re.compile(r"angle between .*? \d+", flags=re.IGNORECASE),

    "front": re.compile(r"front direction of .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)", flags=re.IGNORECASE),
    "left": re.compile(r"left direction of .*? \(-?\d+\.\d+, -?\d+\.\d+, -?\d+\.\d+\)", flags=re.IGNORECASE),
}

# Required patterns per category (each group = interchangeable patterns)
REQUIRED_PHRASES = {
    'height_higher':  [[phrases_dict["location"]], [phrases_dict["location"]]],
    
    'location_above':            [[phrases_dict["location"]], [phrases_dict["location"]]], 
    'location_closer_to_camera': [[phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["dist_from"]], [phrases_dict["dist_from"]]],
    'location_next_to':          [[phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["dist_between"]]],
    
    'orientation_in_front_of': [[phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["vector"]], 
                                [phrases_dict["front"]], [phrases_dict["cosine"],phrases_dict["angle"]]],
    'orientation_on_the_left': [[phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["vector"]],
                                [phrases_dict["left"]], [phrases_dict["cosine"],phrases_dict["angle"]]],
    'orientation_viewpoint': [[phrases_dict["location"]], [phrases_dict["vector"]],
                                [phrases_dict["front"]], [phrases_dict["cosine"],phrases_dict["angle"]], 
                                [phrases_dict["left"]], [phrases_dict["cosine"],phrases_dict["angle"]]],
    
    'multi_object_closer_to': [[phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["dist_between"]], [phrases_dict["dist_between"]]],
    'multi_object_parallel': [[phrases_dict["front"]], [phrases_dict["front"]], [phrases_dict["cosine"],phrases_dict["angle"]]],
    'multi_object_same_direction': [[phrases_dict["front"]], [phrases_dict["front"]], [phrases_dict["cosine"],phrases_dict["angle"]]],
    'multi_object_facing': [[phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["front"]], 
                            [phrases_dict["vector"]], [phrases_dict["cosine"],phrases_dict["angle"]], 
                            [phrases_dict["vector"]], [phrases_dict["cosine"],phrases_dict["angle"]]],
    'multi_object_viewpoint_towards_object': [[phrases_dict["location"]], [phrases_dict["location"]], [phrases_dict["vector"]],
                                                [phrases_dict["front"]], [phrases_dict["cosine"],phrases_dict["angle"]], 
                                                [phrases_dict["left"]], [phrases_dict["cosine"],phrases_dict["angle"]]],
}

ALL_GROUPS = set(group_key(group) for groups in REQUIRED_PHRASES.values() for group in groups)
def process_reward(completions, category, **kwargs):
    """
    Reward function using accuracy = (TP + TN) / (TP + TN + FP + FN),
    correctly handling multiple required and extra matched phrases.
    Same phrase FP included
    """
    rewards = []

    for completion, cat in zip(completions, category):
        content = completion[0]["content"]
        required_groups = REQUIRED_PHRASES[cat]
        required_counter = Counter(group_key(g) for g in required_groups)

        # Step 1: count all matches per pattern
        pattern_match_counts = {id(p): len(p.findall(content)) for p in phrases_dict.values()}

        # Step 2: compute matches per group (sum all interchangeable matches)
        actual_group_counts = {}
        for group in ALL_GROUPS:
            group_patterns = [p for pid in group for p in phrases_dict.values() if id(p) == pid]
            individual_counts = [pattern_match_counts[id(p)] for p in group_patterns]
            count = sum(individual_counts)

            # special rule: if group is interchangeable (more than one pattern),
            # and both patterns are present, subtract 1 to prevent double counting
            if len(group_patterns) > 1 and sum(c > 0 for c in individual_counts) > 1:
                count -= 1

            actual_group_counts[group] = count

        # Step 3: compute TP, FP, FN
        TP = sum(min(actual_group_counts[g], required_counter[g]) for g in required_counter)
        FN = sum(max(required_counter[g] - actual_group_counts.get(g, 0), 0) for g in required_counter)
        FP = sum(
            max(actual_group_counts[g] - required_counter.get(g, 0), 0)
            for g in actual_group_counts
        )
        TN = sum(
            1 for g in ALL_GROUPS if g not in required_counter and actual_group_counts.get(g, 0) == 0
        )

        denom = TP + TN + FP + FN
        reward = (TP + TN) / denom if denom > 0 else 1.0
        rewards.append(reward)
    return rewards