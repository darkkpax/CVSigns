from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RTSD COCO-style annotations to YOLO detect format."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("train"),
        help="Path to RTSD source directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train") / "yolo_rtsd",
        help="Path to YOLO output directory.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place images into YOLO folder.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    dst.write_bytes(src.read_bytes())


def convert_split(
    split_name: str,
    anno_path: Path,
    source_images_root: Path,
    output_root: Path,
    category_to_index: dict[int, int],
    link_mode: str,
) -> dict[str, int]:
    data = load_json(anno_path)
    images = {item["id"]: item for item in data["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    split_images_dir = output_root / "images" / split_name
    split_labels_dir = output_root / "labels" / split_name
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_labels_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    box_count = 0
    empty_count = 0
    list_file = output_root / f"{split_name}.txt"

    with list_file.open("w", encoding="utf-8") as list_fp:
        for image_id, image_info in images.items():
            rel_name = Path(image_info["file_name"]).name
            src_image = source_images_root / rel_name
            if not src_image.exists():
                raise FileNotFoundError(f"Missing source image: {src_image}")

            dst_image = split_images_dir / rel_name
            safe_link_or_copy(src_image, dst_image, link_mode)

            width = float(image_info["width"])
            height = float(image_info["height"])
            label_lines: list[str] = []

            for ann in anns_by_image.get(image_id, []):
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                norm_w = w / width
                norm_h = h / height
                class_id = category_to_index[ann["category_id"]]
                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                )

            dst_label = split_labels_dir / f"{dst_image.stem}.txt"
            dst_label.write_text("\n".join(label_lines), encoding="utf-8")

            if not label_lines:
                empty_count += 1
            box_count += len(label_lines)
            image_count += 1
            list_fp.write(str(dst_image.resolve()) + "\n")

    return {
        "images": image_count,
        "boxes": box_count,
        "empty_images": empty_count,
    }


def write_dataset_yaml(output_root: Path, class_names: list[str]) -> None:
    yaml_lines = [
        f"path: {output_root.resolve()}",
        f"train: {(output_root / 'train.txt').resolve()}",
        f"val: {(output_root / 'val.txt').resolve()}",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    (output_root / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def write_summary(output_root: Path, class_names: list[str], train_stats: dict, val_stats: dict) -> None:
    summary = {
        "classes": len(class_names),
        "class_names": class_names,
        "train": train_stats,
        "val": val_stats,
    }
    (output_root / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()

    train_anno = source_root / "train_anno.json"
    val_anno = source_root / "val_anno.json"
    source_images_root = source_root / "rtsd-frames" / "rtsd-frames"

    train_data = load_json(train_anno)
    categories = sorted(train_data["categories"], key=lambda item: item["id"])
    category_to_index = {item["id"]: index for index, item in enumerate(categories)}
    class_names = [item["name"] for item in categories]

    train_stats = convert_split(
        split_name="train",
        anno_path=train_anno,
        source_images_root=source_images_root,
        output_root=output_root,
        category_to_index=category_to_index,
        link_mode=args.link_mode,
    )
    val_stats = convert_split(
        split_name="val",
        anno_path=val_anno,
        source_images_root=source_images_root,
        output_root=output_root,
        category_to_index=category_to_index,
        link_mode=args.link_mode,
    )

    write_dataset_yaml(output_root, class_names)
    write_summary(output_root, class_names, train_stats, val_stats)

    print(f"YOLO dataset prepared at: {output_root}")
    print(f"Classes: {len(class_names)}")
    print(f"Train images: {train_stats['images']}, boxes: {train_stats['boxes']}")
    print(f"Val images: {val_stats['images']}, boxes: {val_stats['boxes']}")
