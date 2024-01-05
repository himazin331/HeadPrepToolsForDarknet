import json
import os

from loguru import logger
from tqdm import tqdm

ANNOTATIONS_FILE: str = "custom-data/_dataset/RGBD_Indoor_Dataset/annotations.json"
TARGET_DIR: str = "custom-data/_dataset/RGBD_Indoor_Dataset/train/color"


def normalize(
    xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int
) -> tuple[float, float, float, float]:
    bbox_xc: float = (xmin + xmax) / 2.0 / width
    bbox_yc: float = (ymin + ymax) / 2.0 / height
    bbox_w: float = (xmax - xmin) / width
    bbox_h: float = (ymax - ymin) / height
    return bbox_xc, bbox_yc, bbox_w, bbox_h


def main():
    logger.info(f"Loading Annotations file : {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations: dict = json.load(f)

    for fn in tqdm(os.listdir(TARGET_DIR)):
        fn_, ext = os.path.splitext(fn)
        if ext.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        output_label_path: str = os.path.join(TARGET_DIR, f"{fn_}.txt")
        with open(output_label_path, "w") as f:
            try:
                target: dict = annotations[fn]
            except KeyError:
                logger.info(f"Not find 'annotation' file_name: {fn}")
                continue

            width: int = target["width"]
            height: int = target["height"]
            for annotation in target["annotations"]:
                if annotation["category"] == "Head":
                    xmin: int = annotation["x"]
                    ymin: int = annotation["y"]
                    bbox_xc, bbox_yc, bbox_w, bbox_h = normalize(
                        xmin,
                        ymin,
                        xmin + annotation["width"],
                        ymin + annotation["height"],
                        width,
                        height,
                    )
                    f.write(f"0 {bbox_xc} {bbox_yc} {bbox_w} {bbox_h}\n")


if __name__ == "__main__":
    main()
