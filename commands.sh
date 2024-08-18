

#!/bin/bash


# 定义你要替换的项目列表
models=("drums" "chair" "mic" "materials")

# 循环遍历每一个模型名称
for model in "${models[@]}"
do
    # 执行训练脚本
    python pbr_train.py -s "../data/nerf_synthetic/${model}/" -m "./all_test/${model}_test01" --lambda_dist 1000.0 --eval

    # 执行渲染脚本
    python pbr_render.py -s "../data/nerf_synthetic/${model}/" -m "./all_test/${model}_test01" --checkpoint "./all_test/${model}_test01/chkpnt30000.pth" --eval

    python metrics.py -m "./all_test/${model}_test01"
done

models=("teapot" "toaster")

# 循环遍历每一个模型名称
for model in "${models[@]}"
do
    # 执行训练脚本
    python pbr_train.py -s "../data/ref_synthetic/${model}/" -m "./all_test/${model}_test01" --lambda_dist 1000.0 --eval

    # 执行渲染脚本
    python pbr_render.py -s "../data/ref_synthetic/${model}/" -m "./all_test/${model}_test01" --checkpoint "./all_test/${model}_test01/chkpnt30000.pth" --eval

    python metrics.py -m "./all_test/${model}_test01"
done

models=("gardenspheres" "toycar")

# 循环遍历每一个模型名称
for model in "${models[@]}"
do
    # 执行训练脚本
    python pbr_train.py -s "../data/ref_real/${model}/" -m "./all_test/${model}_test01" --lambda_dist 100.0 --eval -i images_4 -r 1 

    # 执行渲染脚本
    python pbr_render.py -s "../data/ref_real/${model}/" -m "./all_test/${model}_test01" --checkpoint "./all_test/${model}_test01/chkpnt30000.pth" --eval -i images_4 -r 1 

    python metrics.py -m "./all_test/${model}_test01"
done

models=("bicycle" "garden")
for model in "${models[@]}"
do
    # 执行训练脚本
    python pbr_train.py -s "../data/mip360/${model}/" -m "./all_test/${model}_test01" --lambda_dist 100.0 --eval -i images_4 -r 1 

    # 执行渲染脚本
    python pbr_render.py -s "../data/mip360/${model}/" -m "./all_test/${model}_test01" --checkpoint "./all_test/${model}_test01/chkpnt30000.pth" --eval -i images_4 -r 1 

    python metrics.py -m "./all_test/${model}_test01"
done

