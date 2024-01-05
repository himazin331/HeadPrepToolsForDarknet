import os
import xml.etree.ElementTree as ET

from loguru import logger
from tqdm import tqdm

ANNOTATIONS_DIR: str = "custom-data/_dataset/HollywoodHeads/Annotations"
OUTPUT_DIR: str = "custom-data/_dataset/HollywoodHeads/JPEGImages"


def normalize(
    xmin: float, ymin: float, xmax: float, ymax: float, width: float, height: float
) -> tuple[float, float, float, float]:
    bbox_xc: float = (xmin + xmax) / 2.0 / width
    bbox_yc: float = (ymin + ymax) / 2.0 / height
    bbox_w: float = (xmax - xmin) / width
    bbox_h: float = (ymax - ymin) / height
    return bbox_xc, bbox_yc, bbox_w, bbox_h


def main():
    for fn in tqdm(os.listdir(ANNOTATIONS_DIR)):
        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, fn))
        root = tree.getroot()

        size = root.find("size")
        width: float = float(size.find("width").text)
        height: float = float(size.find("height").text)

        output_label_path: str = os.path.join(OUTPUT_DIR, f"{fn.split('.')[0]}.txt")
        with open(output_label_path, "w") as f:
            for object in root.findall("object"):
                bndbox = object.find("bndbox")
                if bndbox is None:
                    logger.info(f"Not found 'bndbox' file_name: {fn}")
                    continue

                xmin: float = float(bndbox.find("xmin").text)
                ymin: float = float(bndbox.find("ymin").text)
                xmax: float = float(bndbox.find("xmax").text)
                ymax: float = float(bndbox.find("ymax").text)
                xmin, ymin, xmax, ymax = normalize(
                    xmin, ymin, xmax, ymax, width, height
                )

                f.write(f"0 {xmin} {ymin} {xmax} {ymax}\n")


if __name__ == "__main__":
    main()
