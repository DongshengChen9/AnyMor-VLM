# Running AnyMor-VLM

This guide describes the default AnyMor-VLM workflow and the additional files
required to reproduce the results reported in the paper.

## Default AnyMor-VLM workflow

The default workflow has two stages:

1. Generate figure-ground-map captions with the OpenAI API.
2. Train and evaluate the default AnyMor-VLM configuration.

Before the first stage, open
`experiments/run_anymor_vlm/run_map_descriptioner.bat` and set
`OPENAI_API_KEY` to a valid API key. You may also adjust the model and sampling
parameters at the top of that file.

Then run the two Windows batch files in order:

```
experiments\run_anymor_vlm\run_map_descriptioner
experiments\run_anymor_vlm\run_anymor_default
```

The first script writes generated captions. The second script runs the default
Step 1 and Step 2 AnyMor-VLM pipeline.

## Reproducing the paper results

To reproduce the paper results, download the companion data repository first.
From its `data/` directory, copy the following resources into this code
repository:

| Source in the data repository | Destination in this code repository | Purpose |
| --- | --- | --- |
| `data/urbanForm/` | `datasets/urbanForm/` | UrbanForm images and class-split files |
| `data/saved_models/*` | `saved_models/` | All released model checkpoints |

The required layout is:

```
AnyMor-VLM/
+-- datasets/
|   +-- urbanForm/                    # copied from urbanForm in data repository
|   +-- batch_morphology_59_features_standardized.npz
+-- saved_models/
|   +-- ...                           # all checkpoints from saved_models in data repository
+-- experiments/
|   +-- run_anymor_vlm/
|       +-- run_map_descriptioner.bat
|       +-- run_anymor_default.bat
|   +-- comparison/
|   +-- ablation/
+-- captions/captions_6cls.npz
```

The code repository already includes `captions_6cls.npz` and
`batch_morphology_59_features_standardized.npz`. These files are used for the
caption-based and morphometrics-based experiments and do not need to be downloaded
again from the data repository.

For the repeated paper-result runs, use the scripts under
`experiments\run_anymor_vlm\replicate_AnyMor_10times`, `experiments\ablation\replicate_ablation_10times`, and
`experiments\comparison\*Specific Model BAT*` after the dataset and checkpoints are in the layout
above.
