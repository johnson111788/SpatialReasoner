from spatial_reasoner.prompt import *

def make_conversation_oi(example):
    
    question_text = example["question"]
    options = [f"{opt}. {example[opt]}" for opt in ["A", "B", "C", "D"] if example[opt]]
    options_text = "\n".join(options)
    question = f"Question: {question_text}\nOptions:\n{options_text}\nPlease select the correct answer from the options above."

    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
    }


def make_conversation_oi_3DRwd(example):
    
    question_text = example["question"]
    options = [f"{opt}. {example[opt]}" for opt in ["A", "B", "C", "D"] if example[opt]]
    options_text = "\n".join(options)
    question = f"Question: {question_text}\nOptions:\n{options_text}\nPlease estimate the 3D location, vector between objects, front or left direction, angle, distance to the camera, or distance between objects according to the task requirements and reason the correct answer from the options above."

    return {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
    }