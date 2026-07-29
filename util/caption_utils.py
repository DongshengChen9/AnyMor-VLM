from pathlib import Path
from collections import Counter
from typing import Any, List, Literal

import numpy as np


LookupMode = Literal["id", "index", "auto"]


class CaptionResolver:
    """
    Resolve captions using either:
      - an ID stored in the npz `ids` array;
      - a positional index into the `captions` array.

    lookup_mode:
      - "id": uq_item must be an ID.
      - "index": uq_item must be a positional index.
      - "auto": first try ID lookup, then positional index.
    """

    def __init__(
        self,
        npz_path: str,
        lookup_mode: LookupMode = "id",
    ) -> None:
        self.npz_path = Path(npz_path).expanduser().resolve()
        self.lookup_mode = lookup_mode

        if lookup_mode not in {"id", "index", "auto"}:
            raise ValueError(
                f"Invalid lookup_mode: {lookup_mode!r}. "
                "Expected 'id', 'index', or 'auto'."
            )

        if not self.npz_path.is_file():
            raise FileNotFoundError(
                f"Caption file not found: {self.npz_path}"
            )

        with np.load(
            self.npz_path,
            allow_pickle=True,
        ) as npz:
            required_keys = {"ids", "captions"}
            missing_keys = required_keys.difference(npz.files)

            if missing_keys:
                raise KeyError(
                    f"Caption file {self.npz_path} is missing keys: "
                    f"{sorted(missing_keys)}. "
                    f"Available keys: {npz.files}"
                )

            self.all_ids = np.array(npz["ids"], copy=True)
            self.all_captions = np.array(
                npz["captions"],
                dtype=object,
                copy=True,
            )

            self.all_paths = (
                np.array(npz["images"], copy=True)
                if "images" in npz.files
                else None
            )

        if len(self.all_ids) != len(self.all_captions):
            raise ValueError(
                "The lengths of `ids` and `captions` do not match: "
                f"{len(self.all_ids)} vs "
                f"{len(self.all_captions)}."
            )

        normalized_ids = [
            self._normalize_id(item)
            for item in self.all_ids
        ]

        duplicate_ids = [
            item
            for item, count in Counter(normalized_ids).items()
            if count > 1
        ]

        if duplicate_ids:
            preview = duplicate_ids[:10]
            raise ValueError(
                "Duplicate caption IDs were found. "
                f"Examples: {preview}"
            )

        self.id_to_pos = {
            item: pos
            for pos, item in enumerate(normalized_ids)
        }

    @staticmethod
    def _normalize_id(value: Any) -> Any:
        """Convert NumPy scalar and bytes IDs to stable Python values."""

        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    "Expected a scalar ID, but received an array "
                    f"with shape {value.shape}."
                )
            value = value.item()

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = value.decode(
                    "latin1",
                    errors="replace",
                )

        return value

    @staticmethod
    def _caption_to_string(caption: Any) -> str:
        if caption is None:
            return ""

        if isinstance(caption, bytes):
            try:
                return caption.decode("utf-8")
            except UnicodeDecodeError:
                return caption.decode(
                    "latin1",
                    errors="replace",
                )

        if isinstance(caption, np.str_):
            return str(caption)

        if isinstance(caption, np.ndarray):
            if caption.ndim == 0:
                return CaptionResolver._caption_to_string(
                    caption.item()
                )

            return " ".join(
                CaptionResolver._caption_to_string(item)
                for item in caption.tolist()
            )

        if isinstance(caption, (list, tuple)):
            return " ".join(
                CaptionResolver._caption_to_string(item)
                for item in caption
            )

        return str(caption)

    def _resolve_position(self, uq_item: Any) -> int:
        key = self._normalize_id(uq_item)

        if self.lookup_mode in {"id", "auto"}:
            if key in self.id_to_pos:
                return self.id_to_pos[key]

        if self.lookup_mode in {"index", "auto"}:
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                if self.lookup_mode == "index":
                    raise KeyError(
                        f"Caption index is not an integer: {key!r}"
                    ) from exc
            else:
                if 0 <= index < len(self.all_captions):
                    return index

        raise KeyError(
            f"Caption could not be resolved for {key!r}. "
            f"lookup_mode={self.lookup_mode!r}, "
            f"source={str(self.npz_path)!r}."
        )

    def get(self, uq_item: Any) -> str:
        position = self._resolve_position(uq_item)
        caption = self.all_captions[position]
        return self._caption_to_string(caption)

    def get_batch(self, uq_items: Any) -> List[str]:
        if hasattr(uq_items, "tolist"):
            uq_items = uq_items.tolist()

        return [
            self.get(item)
            for item in uq_items
        ]

    def __len__(self) -> int:
        return len(self.all_captions)