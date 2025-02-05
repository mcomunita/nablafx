CUDA_VISIBLE_DEVICES=0 \
python scripts/main.py fit \
-c cfg/data/data-param_multidrive-ffuzz_trainval.yaml \
-c cfg/model/s4-param/model_bb-param_s4-tvf-b8-s32-c16.yaml \
-c cfg/trainer/trainer_bb.yaml