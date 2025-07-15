import os
from nablafx.data import DryWetFilesPluginDataModule

datamodule = DryWetFilesPluginDataModule(
    root_dir_dry="/import/c4dm-datasets-ext/NNLIN-AFX-DATASET/fuzz-2500ms/chunks/dry",
    root_dir_wet="/import/c4dm-datasets-ext/NNLIN-AFX-DATASET/fuzz-2500ms/chunks/wet",
    params_idxs_to_use=None,
    data_to_use=1.0,
    sample_length=-1,
    preload=True,
    batch_size=16,
    num_workers=4,
)

datamodule.setup(stage="fit")
datamodule.setup(stage="test")

train_dataset = datamodule.train_dataset
val_dataset = datamodule.val_dataset
test_dataset = datamodule.test_dataset

for i in range(len(train_dataset)):
    x, y = train_dataset[i]
    assert x.shape == y.shape

print("train dataset checked!")

for i in range(len(val_dataset)):
    x, y = val_dataset[i]
    assert x.shape == y.shape

print("val dataset checked!")

for i in range(len(test_dataset)):
    x, y = test_dataset[i]
    assert x.shape == y.shape

print("test dataset checked!")
