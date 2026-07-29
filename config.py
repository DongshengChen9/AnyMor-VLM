"""Central paths for the UrbanForm and AllForm AnyMor experiments."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ANYMOR_DATA_ROOT", PROJECT_ROOT / "datasets"))

# Dataset roots expected after installing the separate data repository.
urbanform_root = os.environ.get("ANYMOR_URBANFORM_ROOT", str(DATA_ROOT / "urbanForm"))
allform_root = os.environ.get("ANYMOR_ALLFORM_ROOT", str(DATA_ROOT / "allForm"))

# Data-repository assets used by multimodal and morphology baselines.
caption_root = str(PROJECT_ROOT / "captions")


# Code-repository assets and generated experiment outputs.
get_dir = str(PROJECT_ROOT)
exp_root = os.environ.get("ANYMOR_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))
