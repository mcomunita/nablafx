CUDA_VISIBLE_DEVICES=0 \
python scripts/main.py fit \
-c cfg/data/data-param_multidrive-ffuzz_trainval.yaml \
-c cfg/model/gb/gb-param_fuzz/model_gb-param_fuzz-rnl_peq.sc+g.sc+off.dc+phinv+rnl+g.sc+peq.sc+lp.s.yaml \
-c cfg/trainer/trainer_gb.yaml