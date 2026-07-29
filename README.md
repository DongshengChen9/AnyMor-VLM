# AnyMor-VLM: GCD-inspired urban morphology classification (figure-ground maps) 

This repository contains the code for the journal article: **Classifying arbitrary urban morphotypes: A typology-aligned, open-world approach**
To keep the publication repository lightweight and access-controlled where necessary, images, captions, and trained checkpoints are provided 
separately through the companion **data repository** (https://doi.org/10.5281/zenodo.18326961)
        

## Repository structure

```text
experiments/
  run_anymor_vlm/                 # full two-stage AnyMor model
  ablation/               # modality/component ablations
  comparison/
datasets/                     # place datasets here
dataset_class_name/       # class-name and description resources
docs/                     # data contract and experiment instructions
captions/                 # place the generated map descriptions here
```

## Installation

Python 3.11.5 and a CUDA-enabled PyTorch installation are recommended.

Please check requirements.txt for detailed requirements. 

## Install the companion data

See [the data layout guide](docs/DATA_LAYOUT.md) for the required directory tree and environment-variable alternatives. 

## Run experiments

Concrete commands and the mapping from scripts to paper experiments are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Outputs and reproducibility

Generated logs and checkpoints are written to `outputs/` by default. Set `ANYMOR_OUTPUT_ROOT` in config.py to redirect them. Record the exact
command, random seed,data-repository commit, and checkpoint hash for each reported result.

## Citation

Add the paper citation here after the manuscript receives its final DOI or preprint identifier.
