result name rules:
1. foward_shading with GSIR Parameterization: fw_ir/{}_fwir



logs:

1. helmet, with {fw, GSIR-para, metallic, gamma, 1500 warmup iterations}
python pbr_train.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir --eval --warmup_iterations 1500 --metallic --gamma
normal_loss 有问题，一直稳定不变了 
python pbr_render.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir --checkpoint ./all_test/fw_ir/helmet_fwir/chkpnt30000.pth --eval --metallic --gamma

test-psnr: 25.2399654

2. 重新跑一遍，取消rend_normal和surf_normal的normalization

python pbr_train.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_1 --eval --warmup_iterations 1500 --metallic --gamma --white_background

python pbr_render.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_1 --checkpoint ./all_test/fw_ir/helmet_fwir_1/chkpnt30000.pth --eval --metallic --gamma --white_background

test-psnr: 24.4964848

3. 排除whitebackgroud

python pbr_train.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_2 --eval --warmup_iterations 1500 --metallic --gamma --white_background

python pbr_render.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_2 --checkpoint ./all_test/fw_ir/helmet_fwir_2/chkpnt30000.pth --eval --metallic --gamma --white_background

test-psnr: 24.9810104

4. no warmup

python pbr_train.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_3 --eval --warmup_iterations 15 --metallic --gamma --white_background

python pbr_render.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_3 --checkpoint ./all_test/fw_ir/helmet_fwir_3/chkpnt30000.pth --eval --metallic --gamma --white_background

test-psnr: 24.6840591

5. no warmup + black-background

python pbr_train.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_4 --eval --warmup_iterations 15 --metallic --gamma

python pbr_render.py -s ../data/ref_synthetic/helmet/ -m ./all_test/fw_ir/helmet_fwir_4 --checkpoint ./all_test/fw_ir/helmet_fwir_4/chkpnt30000.pth --eval --metallic --gamma

test-psnr: 23.5593853

5. teapot with {fw, GSIR-para, metallic False, gamma, no warmup, white-bg}

python pbr_train.py -s ../data/ref_synthetic/teapot/ -m ./all_test/fw_ir/teapot_fwir --eval --warmup_iterations 15 --gamma --white_background

python pbr_render.py -s ../data/ref_synthetic/teapot/ -m ./all_test/fw_ir/teapot_fwir --checkpoint ./all_test/fw_ir/teapot_fwir/chkpnt30000.pth --eval --gamma --white_background

test-psnr: 
41.7631073
