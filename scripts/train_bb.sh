CUDA_VISIBLE_DEVICES=0 \
python scripts/main.py fit \
-c cfg-new/data/data_ampeg-optocomp_trainval.yaml \
-c cfg-new/model/s4/model_bb_s4-b4-s4-c16.yaml \
-c cfg-new/trainer/trainer_bb.yaml