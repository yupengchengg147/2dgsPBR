#!/bin/bash

# Define the combinations of models and iter_intervals
export TORCH_HOME=/root/autodl-tmp/torch_chpt

models=("ficus")
iter_intervals=(1 25 100 500 3000)

# Loop through each combination of model and iter_interval
for model in "${models[@]}"; do
    for iter_interval in "${iter_intervals[@]}"; do
        
        # Define the output path based on the model and iter_interval
        output_dir="./output/${model}_${iter_interval}"

        # Run the pbr_train.py script
        python pbr_train.py \
            -s "../data/nerf_synthetic/${model}/" \
            -m "${output_dir}" \
            --eval \
            --warmup_iterations 1 \
            --metallic \
            --lambda_dist 100 \
            --lambda_normal 0.01 \
            --iter_interval "${iter_interval}"

        # Define the checkpoint path
        checkpoint="${output_dir}/chkpnt45000.pth"

        # Run the pbr_render.py script
        python pbr_render.py \
            -s "../data/nerf_synthetic/${model}/" \
            -m "${output_dir}" \
            --eval \
            --checkpoint "${checkpoint}" \
            --metallic

    done
done