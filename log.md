to see if combining deffered shading and forward shading can help with inverse rendering.

for now v0.0
we read ground truth mask, so for deffered shading, we can only shade ROI and we only care about loss within this ROI.

no 3dgs warmup, all from random initialization.

background set to green, to eliminate background's potential impact.

iteratively: 
l1 (gt image  - forward shading img) + l1 (gt image  - deffered shading img)

no distortion loss for now, because opacities' concentration may lead to more severe concave problem?

normal loss from 5000 iteration: for we still want normal and depth to be consistent.
densification also from 500 to 15000. 

densification and normal loss both serve as geometry regularization.

parameterizations keep the same with GSIR.

chpt iterations: 15000, 30000, 45000
densify from 500 to 30000, every 200
--warmup_iterations 1

1. train for 45000 iterations
python pbr_train.py -s ../data/ref_synthetic/helmet/ -m ./output/test_train --eval --warmup_iterations 1 --metallic

python pbr_render.py -s ../data/ref_synthetic/helmet/ -m ./output/test_train --eval --checkpoint ./output/test_train/chkpnt45000.pth --metallic

helmet test_psnr: 23.47

---

2. try out the same but with distortion loss, to see how much distortion loss can affect.
distortion loss from 3000 to the end, lambda distortion use 100

python pbr_train.py -s ../data/ref_synthetic/helmet/ -m ./output/test_train_dist --eval --warmup_iterations 1 --metallic --lambda_dist 100

helmet test_psnr: 23.49

3. try out other cases: {"toaster" "teapot" "ficus"}
a. helmet

python pbr_train.py -s ../data/ref_synthetic/toaster/ -m ./output/test_toaster --eval --warmup_iterations 1 --metallic --lambda_dist 100 --lambda_normal 0.01

python pbr_render.py -s ../data/ref_synthetic/toaster/ -m ./output/test_toaster --eval --checkpoint ./output/test_toaster/chkpnt45000.pth --metallic

helmet test_psnr: 18.74

b. teapot

python pbr_train.py -s ../data/ref_synthetic/teapot/ -m ./output/teapot_100 --eval --warmup_iterations 1 --metallic --lambda_dist 100 --lambda_normal 0.01 --iter_interval 100

python pbr_render.py -s ../data/ref_synthetic/teapot/ -m ./output/test_teapot --eval --checkpoint ./output/test_teapot/chkpnt45000.pth --metallic




