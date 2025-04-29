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