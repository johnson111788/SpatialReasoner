# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from PIL import Image
import torch
import datasets
import transformers
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed

from spatial_reasoner.utils.callbacks import get_callbacks, EarlyStoppingCallback
from spatial_reasoner.configs import SFTConfig
from spatial_reasoner.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ################
    # Load datasets
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    processor.image_processor.max_pixels = training_args.max_pixels
    processor.image_processor.min_pixels = training_args.min_pixels

    def collate_fn(examples):
        samples = []
        for example in examples:
            if example['question']:
                if example["A"]:
                    options = [f"{opt}. {example[opt]}" for opt in ["A", "B", "C", "D"] if example[opt]]
                    question_text = example["question"]
                    options_text = "\n".join(options)
                    question = f"Question: {question_text}\nOptions:\n{options_text}\nPlease select the correct answer from the options above."
                else:
                    question = example["question"]
                
                image_path = os.path.join(training_args.data_dir, example["image_filename"])
                image = Image.open(image_path).convert("RGB")
                converted_sample = [
                        {"role": "user", "content": [
                            {"type": "image", 'image': image},
                            {"type": "text", "text": question}]
                            },
                        {"role": "assistant", "content": [
                            {"type": "text", "text": example['answer_cot']}]},
                    ]
                samples.append(converted_sample)
            else:
                converted_sample = []
                for turn in example['conversations']:
                    role = 'user' if turn['from'] == 'human' else 'assistant'
                    if '<image>' in turn['value']:
                        image_path = os.path.join(training_args.llava_dir, example["image_path"])
                        image = Image.open(image_path).convert("RGB")
                        converted_sample.append({"role": role, "content": [
                            {"type": "image", 'image': image},
                            {"type": "text", "text": turn['value'].lstrip('<image>\n').rstrip('\n<image>')}
                        ]})
                    else:
                        converted_sample.append({"role": role, "content": [
                            {"type": "text", "text": turn['value']}
                        ]})
                samples.append(converted_sample)
        

        batch = processor.apply_chat_template(samples, tokenize=True, return_dict=True, return_tensors="pt", padding=True)
        
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2_5_VLProcessor):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset['val'] if training_args.eval_strategy != "no" else None,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
        processing_class=processor.tokenizer,
        callbacks=get_callbacks(training_args, model_args)+[EarlyStoppingCallback(stop_step=training_args.stop_steps)],
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "model_name": model_args.model_name_or_path,
        "dataset_name": script_args.dataset_name,
        "tags": ["SpatialReasoner"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if model_args.use_peft:
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=True)
    main(script_args, training_args, model_args)
