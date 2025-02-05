python scripts/main.py test \
--config logs-scratch-02/multidrive-ffuzz/S4-TTF/bb_S4-TTF-B8-S32-C16_lr.01_td5_fd5/config.yaml \
--ckpt_path "logs-scratch-02/multidrive-ffuzz/S4-TTF/bb_S4-TTF-B8-S32-C16_lr.01_td5_fd5/nnlinafx-PARAM/p362csrv/checkpoints/last.ckpt" \
--trainer.logger.entity team-mcomunita-qmul \
--trainer.logger.project nnlinafx-PARAM \
--trainer.logger.save_dir logs-scratch-02/multidrive-ffuzz/TEST/bb_S4-TTF-B8-S32-C16_lr.01_td5_fd5--last \
--trainer.logger.name bb_S4-TTF-B8-S32-C16_lr.01_td5_fd5--last \
--trainer.logger.group Multidrive-FFuzz_TEST \
--trainer.logger.tags "['Multidrive-FFuzz', 'TEST']" \
--trainer.accelerator gpu \
--trainer.strategy auto \
--trainer.devices=1 \
--config cfg/data/data-param_multidrive-ffuzz_test.yaml
# --data.sample_length 480000
# -c cfg/model/gb-fuzz-peq+gain+off+rnl+peq.yaml \
# --trainer.logger.id "jhfofqfn" \