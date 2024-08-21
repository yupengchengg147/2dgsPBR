1. teapot, 5000 warm-up iterations, rend_norm.all() based mask, wo metallic, 
no normal loss, no distortion loss (for want to observe pbr_shading's gradient wrt normal and xyz)

    python pbr_deff_train.py -s ../data/ref_synthetic/teapot -m ./all_test/df_ir/teapot_dfir --warmup_iterations 5000 --eval --gamma --white_background

    python pbr_deff_render.py -s ../data/ref_synthetic/teapot -m ./all_test/df_ir/teapot_dfir --checkpoint ./all_test/df_ir/teapot_dfir/chkpnt30000.pth --eval 

todo: 
try wo warmup;
try w normal loss, distortion loss



2. helmet, 5000 warm-up iterations, rend_norm.all() based mask, with metallic, 
no normal loss, no distortion loss (for want to observe pbr_shading's gradient wrt normal and xyz)

    python pbr_deff_train.py -s ../data/ref_synthetic/helmet -m ./all_test/df_ir/helmet_dfir --warmup_iterations 5000 --eval --gamma --white_background --metallic

    python pbr_deff_render.py -s ../data/ref_synthetic/helmet -m ./all_test/df_ir/helmet_dfir --checkpoint ./all_test/df_ir/helmet_dfir/chkpnt30000.pth --eval --white_background --metallic


3. toaster, 5000 warm-up iterations, rend_norm.all() based mask, with metallic, 

python pbr_deff_train.py -s ../data/ref_synthetic/toaster -m ./all_test/df_ir/toaster_dfir --warmup_iterations 5000 --eval --gamma --white_background --metallic

python pbr_deff_render.py -s ../data/ref_synthetic/toaster -m ./all_test/df_ir/toaster_dfir --checkpoint ./all_test/df_ir/toaster_dfir/chkpnt30000.pth --eval --white_background --metallic
