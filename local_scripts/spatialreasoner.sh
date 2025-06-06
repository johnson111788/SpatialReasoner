model_name=ccvl/SpatialReasoner-SFT
dataset_name=ccvl/OpenImages_3DSR_mar16_filtered1200

export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_PROJECT=spatial-reasoning-r1
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_RUN_NAME=$(basename $model_name)-GRPO-$(basename $dataset_name)-$(date +%Y-%m-%d-%H-%M-%S)

wandb login $WANDB_API_KEY

export HF_TOKEN=YOUR_HF_TOKEN

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=3 src/spatial_reasoner/grpo.py \
    --config recipes/Qwen2.5-VL-7B-Instruct/grpo/config_demo.yaml \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path $model_name \
    --dataset_name $dataset_name \
    --run_name $WANDB_RUN_NAME \
    --hub_model_id $WANDB_RUN_NAME \
    --stop_steps 6000