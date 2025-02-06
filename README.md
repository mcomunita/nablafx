<div align="center">
<img width="250px" src="assets/nablafx_1.png">
<br><br>
  
# NablAFx

**differentiable black-box and gray-box audio effects modeling framework**

[Paper](https://arxiv.org/abs/2310.15247) | [Webpage](https://mcomunita.github.io/syncfusion-webpage/)

[Marco Comunità](https://mcomunita.github.io/)<sup>1</sup>, [Riccardo F. Gramaccioni](https://www.linkedin.com/in/riccardo-fosco-gramaccioni/)<sup>2</sup>, [Emilian Postolache](https://emilianpostolache.com/)<sup>2</sup><br>[Emanuele Rodolà,](https://www.linkedin.com/in/erodola/)<sup>2</sup>, [Danilo Comminiello](https://www.linkedin.com/in/danilocomminiello/)<sup>2</sup>, [Joshua D. Reiss](http://www.eecs.qmul.ac.uk/~josh/)<sup>1</sup>

<sup>1</sup> Centre for Digital Music, Queen Mary University of London, UK<br><sup>2</sup> Sapienza University of Rome, Italy

</div>

## Abstract
Sound design involves creatively selecting, recording, and editing sound effects for various media like cinema, video games, and virtual/augmented reality. One of the most time-consuming steps when designing sound is synchronizing audio with video. In some cases, environmental recordings from video shoots are available, which can aid in the process. However, in video games and animations, no reference audio exists, requiring manual annotation of event timings from the video. We propose a system to extract repetitive actions onsets from a video, which are then used - in conjunction with audio or textual embeddings - to condition a diffusion model trained to generate a new synchronized sound effects audio track. In this way, we leave complete creative control to the sound designer while removing the burden of synchronization with video. Furthermore, editing the onset track or changing the conditioning embedding requires much less effort than editing the audio track itself, simplifying the sonification process. We provide sound examples, source code, and pretrained models to faciliate reproducibility


```BibTex
@inproceedings{comunita2024syncfusion,
  title={Syncfusion: Multimodal Onset-Synchronized Video-to-Audio Foley Synthesis},
  author={Comunit{\`a}, Marco and Gramaccioni, Riccardo F and Postolache, Emilian and Rodol{\`a}, Emanuele and Comminiello, Danilo and Reiss, Joshua D},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={936--940},
  year={2024},
  organization={IEEE}
}
```

---

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





<!-- ![syncfusion](img/syncfusion-image.png){width=200} -->
<img width="700px" src="img/syncfusion-image.png">

</div>

## Setup

Install the requirements (use Python version < 3.10).
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

:warning:
CLAP might give errors with `transformers` versions other than `4.30.2`.

Afterwards, copy `.env.tmp` as `.env` and replace with your own variables (example values are random):

```
DIR_LOGS=/logs/diffusion
DIR_DATA=/data

# Required if using wandb logger
WANDB_PROJECT=audioproject
WANDB_ENTITY=johndoe
WANDB_API_KEY=a21dzbqlybbzccqla4txa21dzbqlybbzccqla4tx
```

---

## Dataset
You can find the GREATEST HITS dataset page at [https://andrewowens.com/vis/](https://andrewowens.com/vis/), where you can download the [high-res](https://web.eecs.umich.edu/~ahowens/vis/vis-data.zip) or [low-res](https://web.eecs.umich.edu/~ahowens/vis/vis-data-256.zip) videos and annotations.

---

## Pre-processing for Onset Model
To prepare the dataset for training you have to pre-process the videos and annotations, as well as prepare the data split.

### Video Pre-processing
To extract the video frames and audio from videos run (setting the arguments as necessary)
```
python script/gh_preprocess_videos.py
```

### Annotations
To extract the annotations run (setting the arguments as necessary):
```
python script/gh_preprocess_annotations.py
```

### Data Splits
To prepare the data splits run (setting the arguments as necessary):
```
python script/gh_preprocess_split.py
```

The scripts (training, testing) for the onset model expect the pre-processed files to be placed in `data/greatest-hits/mic-mp4-processed`.
Create the directories and place the files inside or use a symbolic link `ln -s path/to/processed/folder`

---

## Pre-processing and CLAP checkpoint for Diffusion Model

Pre-processed video frames, audio and annotations are organized into shards for training and validation (we use webdataset to train the diffusion model):

- train_shard_1/2/3.tar
- val_shard_1.tar

To test the diffusion model using ground truth onset annotations you have the test shard:

- test_shard_1.tar

To test the diffusion model using annotations generated by the onset model (w/ or w/out augmentation) you have the test shards:

- test_onset_preds.tar
- test_onset_augment_preds.tar

All data is available here:

[https://zenodo.org/records/12634671](https://zenodo.org/records/12634671)

The scripts (training, evaluation) for diffusion expect the shards to be placed in `data/greatest-hits/webdataset`.
Create the directories and place the shards inside or use a symbolic link `ln -s path/to/shards/folder`

Additionally, the diffusion model requires the CLAP checkpoint [630k-audioset-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt) to be placed in
`checkpoints` folder. Download the checkpoint, create the folder `checkpoints` and place it inside or use a symbolic link `ln -s path/to/clap-checkpoint/folder`

---

## Training

### Onset Model

To train the onset model WITHOUT data augmentation run:
```
CUDA_VISIBLE_DEVICES=0 sh script/train_onset_model_gh.sh
```
The training is configured using Lightning CLI with the following files:
```
cfg/data/data-onset-greatesthit.yaml
cfg/model/model-onset.yaml
cfg/trainer/trainer-onset.yaml
```
Check the files and change the arguments as necessary.

---

To train the onset model WITH data augmentation run:
```
CUDA_VISIBLE_DEVICES=0 sh script/train_onset_model_gh_augment.sh
```
The training is configured using Lightning CLI with the following files:
```
cfg/data/data-onset-greatesthit-augment.yaml
cfg/model/model-onset.yaml
cfg/trainer/trainer-onset-augment.yaml
```
Check the files and change the arguments as necessary.

---

### Diffusion Model

To train the diffusion model run:

```
CUDA_VISIBLE_DEVICES=0 sh script/train_diffusion_model_gh.sh
```
The training is configured using Hydra with the following files:
```
exp/model/diffusion.yaml
exp/train_diffusion_gh.yaml
```
Check the files and change the arguments as necessary.

---

## Checkpoints

You can find the checkpoints for both, Onset and Diffusion models on Zenodo:
[https://zenodo.org/records/12634630](https://zenodo.org/records/12634630).
Such checkpoints are required for reproducing the results in the paper and should
be placed in the `checkpoints` directory.

---

## Testing and Evaluation

### Onset Model
To test the onset model (i.e., BCE loss, Average Precision, Binary Accuracy and Number of Onsets Accuracy) run:
```
CUDA_VISIBLE_DEVICES=0 sh script/test_onset_model.sh
```
changing the necessary arguments.

This corresponds to Table 1 in the paper.

---

### Diffusion Model

First, check that `epoch=784-valid_loss=0.008.ckpt` is present in `checkpoints` folder and
`test_shard_1.tar`, `test_onset_preds.tar`, `test_onset_augment_preds.tar` in `data/greatest-hits/webdataset`.

Following, prepare the GT data for FAD experiments by running:

- `CUDA_VISIBLE_DEVICES=0 sh script/run_prepare_gh_gt.sh` (GT data for diffusion only experiments)
- `CUDA_VISIBLE_DEVICES=0 sh script/run_prepare_gh_gt_pred.sh` (GT data for diffusion + predicted onsets experiments)

The scripts create the GT data in `output/experiments/gh-gt`, `output/experiments/gh-gt-pred` and `output/experiments/gh-gt-pred-augment`.

You can now run:

- `CUDA_VISIBLE_DEVICES=0 sh script/run_evaluate_gh_gen.sh` (evaluates FAD for diffusion only conditioning with audio (random onsets); Table 2)
- `CUDA_VISIBLE_DEVICES=0 sh script/run_evaluate_gh_gen_text.sh` (evaluates FAD for diffusion only conditioning with text (random onsets); Table 2)
- `CUDA_VISIBLE_DEVICES=0 sh script/run_evaluate_gh_gen_pred.sh` (evaluates FAD for diffusion conditioning with predicted onsets and audio; Table 3)
- `CUDA_VISIBLE_DEVICES=0 sh script/run_evaluate_gh_gen_pred_augment.sh` (evaluates FAD for diffusion conditioning with predicted onsets obtained via augmented model and audio; Table 3)

:warning:
You might need to reduce the batch size in the `exp` files, depending on your available GPU memory. Results may vary because of this. Experiments in paper performed with bs=10.

To compute the onset metrics for the diffusion model (i.e., Average Precision, Binary Accuracy and Number of Onsets Accuracy) run:

- `sh script/evaluate_onset.sh` (evaluates metrics from generated audio using GT onsets and audio conditioning; Table 2)
- `sh script/evaluate_onset_text.sh` (evaluates metrics from generated audio using GT onsets and text conditioning; Table 2)
- `sh script/evaluate_onset_pred.sh` (evaluates metrics from generated audio using predicted onsets and audio conditioning; Table 3)
- `sh script/evaluate_onset_pred_augment.sh` (evaluates metrics from generated audio using predicted onsets via augmented model and audio conditioning; Table 3)

---

## Credits

[https://github.com/archinetai/audio-diffusion-pytorch-trainer](https://github.com/archinetai/audio-diffusion-pytorch-trainer)\
[https://github.com/XYPB/CondFoleyGen](https://github.com/XYPB/CondFoleyGen)\
[https://andrewowens.com/vis/](https://andrewowens.com/vis/)
