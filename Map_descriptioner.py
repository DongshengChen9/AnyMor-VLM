"""Generate GPT descriptions for figure-ground maps with anonymous API uploads."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

CSV_FIELDS = (
    "image",
    "unique_id",
    "label",
    "caption",
)

def make_anonymous_filename(image_bytes: bytes) -> str:
    """Create a deterministic filename that cannot reveal the source filename."""
    image_id = hashlib.sha256(image_bytes).hexdigest()
    return f"map_{image_id[:16]}.png"


def prepare_anonymous_image(image_path: str) -> tuple[str, dict[str, object]]:
    """Return a metadata-free PNG payload and safe metadata for an API request."""
    with Image.open(image_path) as source_image:
        source_image.load()
        has_alpha = "A" in source_image.getbands()
        clean_image = source_image.convert("RGBA" if has_alpha else "RGB")
        width, height = clean_image.size

    image_buffer = io.BytesIO()
    clean_image.save(image_buffer, format="PNG")
    clean_bytes = image_buffer.getvalue()
    anonymous_image_id = hashlib.sha256(clean_bytes).hexdigest()

    metadata = {
        "anonymous_filename": make_anonymous_filename(clean_bytes),
        "anonymous_image_id": anonymous_image_id,
        "width": width,
        "height": height,
        "mime_type": "image/png",
        "metadata_removed": True,
    }
    encoded_image = base64.b64encode(clean_bytes).decode("ascii")
    return encoded_image, metadata


def save_or_append_npz(npz_path: Path, records: list[dict[str, object]]) -> None:
    """Append records and save only unique IDs and captions in the NPZ file."""
    unique_ids: list[object] = []
    captions: list[object] = []
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as npz_data:
            if "unique_id" in npz_data.files:
                unique_ids = list(npz_data["unique_id"])
            elif "ids" in npz_data.files:
                unique_ids = list(npz_data["ids"])
            else:
                raise ValueError(f"{npz_path} does not contain unique IDs.")

            if "caption" in npz_data.files:
                captions = list(npz_data["caption"])
            elif "captions" in npz_data.files:
                captions = list(npz_data["captions"])
            else:
                raise ValueError(f"{npz_path} does not contain captions.")

    for record in records:
        unique_ids.append(record["unique_id"])
        captions.append(record["caption"])

    np.savez(
        npz_path,
        unique_id=np.asarray(unique_ids),
        caption=np.asarray(captions, dtype=str),
    )
    print(f"NPZ updated with {len(records)} entries (total {len(unique_ids)}).")


def _validate_csv_schema(csv_path: Path) -> bool:
    """Return whether a header is needed and reject incompatible existing files."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        header = next(csv.reader(csv_file), None)
    if header != list(CSV_FIELDS):
        raise ValueError(
            f"{csv_path} has an incompatible header. Use a new save directory or "
            "rename the old captions.csv before generating new captions."
        )
    return False


def generate_and_save_captions_chatgpt(
    dataset,
    save_dir: str | Path,
    client: Any,
    model: str = "gpt-4o-mini",
    prompt_prefix: str = "Describe this image in detail.",
    max_tokens: int = 77,
    temperature: float = 0.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: int = 42,
    max_images: int | None = None,
    start_idx: int = 0,
    image_detail: str = "low",
) -> tuple[Path, Path]:
    """Describe dataset images and store source paths with anonymous upload metadata."""
    if not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0 and 1")
    if not -2.0 <= frequency_penalty <= 2.0:
        raise ValueError("frequency_penalty must be between -2 and 2")
    if not -2.0 <= presence_penalty <= 2.0:
        raise ValueError("presence_penalty must be between -2 and 2")
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative")
    if max_images is not None and max_images < 0:
        raise ValueError("max_images must be non-negative")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / "captions.csv"
    npz_path = save_dir / "captions.npz"
    write_header = _validate_csv_schema(csv_path)

    end_idx = len(dataset)
    if max_images is not None:
        end_idx = min(end_idx, start_idx + max_images)

    records = []
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for dataset_index in tqdm(
            range(start_idx, end_idx), desc="Generating captions"
        ):
            image_path, label = dataset.imgs[dataset_index]
            unique_id = dataset.uq_idxs[dataset_index]
            encoded_image, anonymous_metadata = prepare_anonymous_image(image_path)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_prefix},
                            {
                                "type": "text",
                                "text": "Anonymous image metadata: "
                                + json.dumps(anonymous_metadata),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}",
                                    "detail": image_detail,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
            )
            caption = response.choices[0].message.content.strip()
            record = {
                "image": image_path,
                "unique_id": unique_id,
                "label": label,
                "caption": caption,
            }
            writer.writerow(record)
            csv_file.flush()
            records.append(record)
            print(f"[{dataset_index}] {caption}")

    save_or_append_npz(npz_path, records)
    return csv_path, npz_path

def get_class_split_info(args):
      
    # loading class split file if exist
    if not args.random_class_split:
        assert os.path.exists(args.class_split_file)
        
        with open(args.class_split_file, 'r') as f:
            split_info = json.load(f)
        args.train_classes = split_info['train_classes']
        args.unlabeled_classes = split_info['unlabeled_classes']
        args.train_labelled_classes = split_info['train_labelled_classes']
        args.train_unlabelled_classes = split_info['train_unlabelled_classes']

        return args

    # random generate class split
    np.random.seed(args.seed)

    all_classes = np.arange(args.num_classes4test)
    np.random.shuffle(all_classes)
    
    num_train_classes = int(args.num_classes4test/3*2)
    num_train_labeled_classes = int(num_train_classes/2)

    train_classes = np.sort(all_classes[:num_train_classes])
    unlabeled_classes = np.sort(all_classes[num_train_classes:])

    labelled_classes = np.sort(np.random.choice(train_classes,size=num_train_labeled_classes,replace=False))
    unlabelled_train_classes = np.sort(np.setdiff1d(train_classes,labelled_classes))
    unlabeled_classes = sorted(unlabeled_classes.tolist()+unlabelled_train_classes.tolist())

    split_info = {
        "train_classes":
            train_classes.tolist(),
        "unlabeled_classes":
            unlabeled_classes,
        "train_labelled_classes":
            labelled_classes.tolist(),
        "train_unlabelled_classes":
            unlabelled_train_classes.tolist()
    }
    
    base_name = "class_split_info_run"
    extension = ".json"
    counter = 1
    while True:
        class_split_path = f"{base_name}{counter}{extension}"
        class_split_path = os.path.join('data/', class_split_path)
        if not os.path.exists(class_split_path):
            break  
        counter += 1
        
    with open(class_split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
        
    args.train_classes = split_info['train_classes']
    args.unlabeled_classes = split_info['unlabeled_classes']
    args.train_labelled_classes = split_info['train_labelled_classes']
    args.train_unlabelled_classes = split_info['train_unlabelled_classes']
    print(args.train_classes)
    return args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe figure-ground maps using anonymous OpenAI API uploads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", choices=("urbanform", "allform"), default="urbanform")
    parser.add_argument("--class_split_file", default="data/class_split_info_run00.json")
    parser.add_argument("--prop_train_labels", type=float, default=0.8)
    parser.add_argument("--transform", default="imagenet")
    parser.add_argument("--interpolation", type=int, default=3)
    parser.add_argument("--crop_pct", type=float, default=0.875)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--model", default="gpt-4o-2024-08-06")
    parser.add_argument("--max_tokens", type=int, default=77)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--image_detail", choices=("low", "high", "auto"), default="low")
    parser.add_argument('--random_class_split', action='store_true', default=False, help='if true, randomly split Labeled-Unlabeled-Novel equally. if false, load --class_split_file')
    return parser.parse_args()


def main() -> None:
    from data.augmentations import get_transform
    from data.get_datasets import get_class_splits, get_datasets
    from openai import OpenAI

    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")
    args.image_size = 224
    args = get_class_split_info(args)    
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    
    _, _, _, _, _, full_dataset = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )

    prompt = (
        "You are an expert urban morphologist. This image is a figure-ground "
        "diagram in the study of urban form, where white represents buildings "
        "and streets. Describe morphology, size, geometry, density, overall "
        "spatial characteristics, and staining pattern in one or two short sentences."
    )
    
    #prompt = "buildings and streets with pattern of" # for BLIP

    csv_path, npz_path = generate_and_save_captions_chatgpt(
        dataset=full_dataset,
        save_dir=args.save_dir,
        client=OpenAI(),
        model=args.model,
        prompt_prefix=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
        max_images=args.max_images,
        start_idx=args.start_idx,
        image_detail=args.image_detail,
    )
    print(f"Saved captions to {csv_path} and {npz_path}.")


if __name__ == "__main__":
    main()
