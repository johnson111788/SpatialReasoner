model_name=Qwen/Qwen2.5-VL-7B-Instruct
dataset_name=ccvl/SpatialReasoner-Basic3D-QA

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=spatial-reasoning-r1
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_RUN_NAME=$(basename $model_name)-SFT-$(basename $dataset_name)-$(date +%Y-%m-%d-%H-%M-%S)

wandb login $WANDB_API_KEY

export HF_TOKEN=YOUR_HF_TOKEN

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=4 src/spatial_reasoner/sft.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/sft/config_demo.yaml \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path $model_name \
    --dataset_name $dataset_name \
    --run_name $WANDB_RUN_NAME \
    --stop_steps 3000

# Download preprocessor_config.json and chat_template.json
# https://github.com/huggingface/transformers/issues/29790#issuecomment-2016755078
wget -P checkpoints/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/preprocessor_config.json
wget -P checkpoints/${WANDB_RUN_NAME} https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/chat_template.json
