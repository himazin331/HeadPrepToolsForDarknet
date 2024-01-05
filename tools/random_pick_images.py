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


def write_txt(images: list[str], output_dir: str, output_txt_path: str = None):
    if output_txt_path is None:
        for fn in tqdm(images):
            fpath: str = os.path.join(output_dir, fn)
            try:
                shutil.copy2(os.path.join(INPUT_DIR, fn), output_dir)

                fn_txt = os.path.splitext(fn)[0] + ".txt"
                shutil.copy2(os.path.join(INPUT_DIR, fn_txt), output_dir)
            except Exception as e:
                print(e)
    else:
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
    write_txt(eval_images, EVAL_DIR)


if __name__ == "__main__":
    main()
