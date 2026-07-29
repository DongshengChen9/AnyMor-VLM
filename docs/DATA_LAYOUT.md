# Data and Checkpoint Layout

This code repository includes the two derived assets required to reproduce the
reported multimodal and morphometrics-based results:

- `captions/captions_6cls.npz`: generated captions paired with UrbanForm sample IDs.
- `datasets/batch_morphology_59_features_standardized.npz`: standardized 59-dimensional
  morphometrics features paired with the same sample IDs.

These files are stored at the repository root. Do not download duplicate copies
from the data repository.

## Additional files required for full reproduction

To reproduce all paper results, download the companion data repository 
(https://doi.org/10.5281/zenodo.18326961) and copy its contents into this 
code repository as follows:

1. Copy the data repository's `datasets/` directory to `AnyMor-VLM/datasets`.
   It contains the UrbanForm images and the class-split JSON files.
2. Copy the data repository's `saved_models/` directory to `AnyMor-VLM/saved_models/`.
   It contains the released model checkpoints used by the reproduction scripts.

The resulting layout should be:

```text
AnyMor-VLM/
+-- captions/captions_6cls.npz        # existed
+-- datasets/
|   +-- urbanForm/                    # copied from the data repository
|       +-- class_split_info_run00.json
|       +-- ...                       # UrbanForm ImageFolder class directories
|   +-- batch_morphology_59_features_standardized.npz/                    # existed
+-- saved_models/                     # copied from data-repository saved_models/
|   +-- ...                           # released checkpoint files
+-- docs/
+-- ...
```

The image folders under `datasets/urbanForm/` must be readable by
`torchvision.datasets.ImageFolder`. The sample IDs in the two NPZ files must
match the IDs emitted by the UrbanForm dataset loader.

## Paths used by the scripts

When a script requires captions, pass the root-level caption asset explicitly:

```bash
--captions_path captions/captions_6cls.npz
```

For the morphometrics baseline, use:

```bash
--morphology_npz datasets/batch_morphology_59_features_standardized.npz
```

The released checkpoints are expected under `saved_models/`, for example:

```bash
--TES_model_dir saved_models/tes_222_run00.pt
```
