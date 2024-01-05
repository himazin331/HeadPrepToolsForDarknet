import os
import re
import shutil

from loguru import logger
from tqdm import tqdm

ANNOTATIONS_FILES: list[str] = [
    "custom-data/_dataset/brainwash/brainwash_train.idl",
    "custom-data/_dataset/brainwash/brainwash_test.idl",
    "custom-data/_dataset/brainwash/brainwash_val.idl",
]
TARGET_DIR: str = "custom-data/_dataset/brainwash/brainwash_11_24_2014_images"
ENABLE_RENAME: bool = False


def rename_file():
    parent_dir: str = TARGET_DIR.split("/")[-1]
    for fn in tqdm(os.listdir(TARGET_DIR)):
        try:
            shutil.move(
                os.path.join(TARGET_DIR, fn),
                os.path.join(TARGET_DIR, f"{parent_dir}_{fn}"),
            )
        except Exception as e:
            logger.error(e)


def load_annotations() -> dict[str, list[tuple]]:
    annotations: dict = dict()
    for anno_path in ANNOTATIONS_FILES:
        logger.info(f"Loading Annotations file : {anno_path}")
        with open(anno_path, "r") as f:
            for line in tqdm(f.readlines()):
                r: list[str] = line.split()[0].split("/")
                parent_dir: str = r[0][1:]
                fname: str = r[1].strip()[:-2]

                pattern = re.compile(
                    r"\((\d+\.\d+), (\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\),"
                )
                matches: list[str] = re.findall(pattern, line)

                coordinates: list[tuple] = [
                    normalize(*map(float, coord)) for coord in matches
                ]

                fname = f"{parent_dir}_{fname}"
                annotations[fname] = coordinates
    return annotations


def normalize(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: float = 640.0,
    height: float = 480.0,
) -> tuple[float, float, float, float]:
    bbox_xc: float = (xmin + xmax) / 2.0 / width
    bbox_yc: float = (ymin + ymax) / 2.0 / height
    bbox_w: float = (xmax - xmin) / width
    bbox_h: float = (ymax - ymin) / height
    return bbox_xc, bbox_yc, bbox_w, bbox_h


def main():
    if ENABLE_RENAME:
        rename_file()

    annotations: dict[str, list[tuple]] = load_annotations()
    for fn in tqdm(os.listdir(TARGET_DIR)):
        fn_, ext = os.path.splitext(fn)
        if ext.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        try:
            coordinates: list[tuple] = annotations[fn]
        except KeyError:
            logger.warning(f"Not found 'annotation' file_name: {fn}")
            continue

        output_label_path: str = os.path.join(TARGET_DIR, f"{fn_}.txt")
        with open(output_label_path, "w") as f:
            for coord in coordinates:
                bbox_xc, bbox_yc, bbox_w, bbox_h = coord
                f.write(f"0 {bbox_xc} {bbox_yc} {bbox_w} {bbox_h}\n")


if __name__ == "__main__":
    main()
