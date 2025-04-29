batch_size=36
gpu_ids=0,1,2,3
result_name=SpatialReasoner

datasets=("3DSRBench" "CV-Bench-3D")
prompt_paths=("./data/3dsrbench_v1_vlmevalkit_circular.tsv" "CV-Bench-3D.tsv")

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    prompt_path=${prompt_paths[$i]}

    echo $result_name $dataset 
    model_path=YOUR_SpatialReasoner_MODEL_NAME/PATH
    output_path=YOUR_OUTPUT_PATH/${result_name}_${dataset}.xlsx

    python src/eval/infer.py \
        --model_path ${model_path} \
        --batch_size ${batch_size} \
        --output_path ${output_path} \
        --prompt_path ${prompt_path} \
        --gpu_ids ${gpu_ids}
done