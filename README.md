<div align="center">
  
<img width="600px" src="assets/nablafx_1.png">
<img width="600px" src="assets/nablafx_2.png">

</div>

python 3.9.7

conda env create -f requirements_temp-for-rnl.yaml

move weights/rationals_config.json to:
/homes/mc309/.conda/envs/nablafx/lib/python3.9/site-packages/rational/rationals_config.json

conda env update --file environment.yml

mkdir data
mkdir logs

cd data
ln -s /import/c4dm-datasets-ext/TONETWIST-AFX-DATASET/

save_dir: logs/multidrive-ffuzz/S4-TVF/bb_S4-TVF-B8-S32-C16_lr.01_td.5_fd.5 # dir needs to already exist



python -m venv .venv
source .venv/bin/activate
pip install -r requirements_temp-for-rnl.txt
pip install --upgrade -r requirements.txt
