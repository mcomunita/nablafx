import os
from nablafx.data import PluginDataset

dataset = PluginDataset(
    root_dir=os.path.join("/import/c4dm-datasets-ext/NNLIN-AFX-DATASET/fuzz-2500ms/chunks", "train"),
    params_file=None,
    params_to_use=None,
    data_to_use=1.0,
    sample_length=-1,
    preload=True,
)

for i in range(len(dataset)):
    x, y = dataset[i]

    assert x.shape == y.shape

print("dataset checked!")
