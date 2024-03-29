import os
import random
import shutil

from loguru import logger
from tqdm import tqdm

INPUT_DIR: str = "custom-data/_dataset/HollywoodHeads/JPEGImages"

TRAIN_DIR: str = "custom-data/train"
TRAIN_TXT: str = "custom-data/train.txt"
TRAIN_IMAGES_NUM: int = 179784

TEST_DIR: str = "custom-data/test"
TEST_TXT: str = "custom-data/test.txt"
TEST_IMAGES_NUM: int = 44946

EVAL_DIR: str = "custom-data/eval"
EVAL_TXT: str = "custom-data/test.txt"


def make_dirs(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        logger.warning(f"Directory already exists: {dir_path}")


def write_txt(images: list[str], output_dir: str, output_txt_path: str):
    with open(output_txt_path, "a") as f:
        for fn in tqdm(images):
            fpath: str = os.path.join(output_dir, fn)
            try:
                shutil.copy2(os.path.join(INPUT_DIR, fn), output_dir)

                fn_txt = os.path.splitext(fn)[0] + ".txt"
                shutil.copy2(os.path.join(INPUT_DIR, fn_txt), output_dir)
            except Exception as e:
                print(e)
            f.write(fpath + "\n")


def main():
    make_dirs(TRAIN_DIR)
    make_dirs(TEST_DIR)
    make_dirs(EVAL_DIR)

    images: list[str] = [
        f for f in os.listdir(INPUT_DIR) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    assert (
        len(images) >= TRAIN_IMAGES_NUM + TEST_IMAGES_NUM
    ), f"Input imgs num: {len(images)} < train: {TRAIN_IMAGES_NUM} + test: {TEST_IMAGES_NUM}"

    train_images: list[str] = [
        images.pop(random.randint(0, len(images) - 1)) for _ in range(TRAIN_IMAGES_NUM)
    ]
    test_images: list[str] = [
        images.pop(random.randint(0, len(images) - 1)) for _ in range(TEST_IMAGES_NUM)
    ]
    eval_images: list[str] = images

    logger.info(f"Copy train images and annotations: {INPUT_DIR} -> {TRAIN_DIR}")
    write_txt(train_images, TRAIN_DIR, TRAIN_TXT)
    logger.info(f"Copy test images and annotations: {INPUT_DIR} -> {TEST_DIR}")
    write_txt(test_images, TEST_DIR, TEST_TXT)
    logger.info(f"Copy eval images and annotations: {INPUT_DIR} -> {EVAL_DIR}")
    write_txt(eval_images, EVAL_DIR, EVAL_TXT)


if __name__ == "__main__":
    main()
