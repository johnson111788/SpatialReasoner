import os, base64, tqdm, argparse, torch, pandas as pd, string

from PIL import Image
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from torch.multiprocessing import Process, set_start_method, Manager

from spatial_reasoner.prompt import *

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 1. get evaluation configuration <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_eval_config():
    parser = argparse.ArgumentParser(description="Inference script for GeoQA evaluation.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the model checkpoint (e.g., qwen2vl model or a fine-tuned model).")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for inference. Reduce if GPU OOM (default: 50).")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save inference result (e.g., JSON file).")
    parser.add_argument("--prompt_path", required=True, type=str, help="Path to the prompts JSONL file for GeoQA evaluation.")
    all_gpu = ",".join(map(str, range(torch.cuda.device_count())))
    parser.add_argument("--gpu_ids", default=all_gpu, help="comma-separated list of GPU IDs to use")
    parser.add_argument("--skip_system_prompt", action="store_true", help="Skip the system prompt for SFT.")
    args = parser.parse_args()
    return args


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>> 2. load testset <<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def prepare_test_messages(testset_path, skip_system_prompt=False):

    def make_conversation_image(example):

        options = [f"{opt}. {example[opt]}" for opt in string.ascii_uppercase if opt in example and not pd.isna(example[opt])]
        question_text = example["question"]
        options_text = "\n".join(options)

        question = f"Question: {question_text}\nOptions:\n{options_text}\nPlease select the correct answer from the options above."
        conv = [] if skip_system_prompt else [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

        return conv+[
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]

    dataset = pd.read_csv(testset_path, sep='\t')
    dataset["prompt"] = dataset.apply(make_conversation_image, axis=1)
    
    return dataset, list(dataset["prompt"]), list(dataset["image"])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>> 3. use several GPUs to accelerate inference at testset <<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def init_model(model_path, gpu_id):
    """init a model(args.model_path) on a specific gpu"""
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{gpu_id}",
    )

    # default processer
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left", use_fast=True)
    return model, processor

def answer_a_batch_question_qwen(batch_messages, model, processor):
    """ let qwen answer a batch of questions """
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages[0]]

    # process image
    def decode_image(image_string):
        image_data = base64.b64decode(image_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image

    image_inputs = [decode_image(image_string) for image_string in batch_messages[1]]


    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generate_kwargs = dict(
            max_new_tokens=512,
            top_p=0.001,
            top_k=1,
            temperature=0.01,
            repetition_penalty=1.0,
        )

    generated_ids = model.generate(**inputs, **generate_kwargs, use_cache=True) # do_sample=False
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    batch_output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return batch_output_text

def infer_on_single_gpu(model_path, device_id, chunk_of_tested_messages, image_paths_chunk, batch_size, results=None):
    """init model on this single gpu and let it answer asign chunk of questions"""
    model, processor = init_model(model_path, device_id)

    ### split batch
    responses = []
    batch_messages_list = [[chunk_of_tested_messages[start: start + batch_size], image_paths_chunk[start: start + batch_size]]
               for start in range(0, len(chunk_of_tested_messages), batch_size)]
    for batch_messages in tqdm.auto.tqdm(batch_messages_list, desc=f"GPU {device_id} progress", position=device_id, leave=False):
        batch_output_text = answer_a_batch_question_qwen(batch_messages, model, processor)

        responses.extend(batch_output_text)

    results[device_id] = responses
    return


def multi_gpu_inference(prompts, image_paths, gpu_ids, model_path, batch_size):
    """ let each gpu (along with a model) answer a chunk of questions """
    set_start_method("spawn", force=True)
    manager = Manager()
    gpu_id2result = manager.dict()

    gpu_ids = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
    num_gpus = len(gpu_ids)

    chunk_size = len(prompts) // num_gpus
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(prompts)
        chunk = prompts[start_idx: end_idx]
        image_paths_chunk = image_paths[start_idx: end_idx]
        process = Process(target=infer_on_single_gpu, args=(model_path, gpu_id, chunk, image_paths_chunk, batch_size, gpu_id2result))
        process.start()
        processes.append(process)

    # for process in tqdm.auto.tqdm(processes, desc="Inference progress", position=num_gpus, leave=True):
    for process in processes:
        process.join()

    all_predicts = []
    for gpu_id in gpu_ids:
        all_predicts.extend(gpu_id2result[gpu_id])

    return all_predicts


if __name__ == "__main__":
    args = get_eval_config()
    testset_data, tested_messages, image_paths = prepare_test_messages(testset_path=args.prompt_path, skip_system_prompt=args.skip_system_prompt)
    all_predicts = multi_gpu_inference(tested_messages, image_paths, args.gpu_ids, args.model_path, args.batch_size)

    testset_data['prediction'] = all_predicts
    test_df = testset_data.drop(columns=['prompt'])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with pd.ExcelWriter(args.output_path, engine='xlsxwriter') as writer:
        test_df.to_excel(writer, sheet_name='test', index=False)

