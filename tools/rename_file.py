import os
import shutil

from tqdm import tqdm

TARGET_DIR: str = "custom-data/_dataset/brainwash/brainwash_11_24_2014_images"

parent_dir: str = TARGET_DIR.split("/")[-1]
for fn in tqdm(os.listdir(TARGET_DIR)):
    try:
        shutil.move(
            os.path.join(TARGET_DIR, fn), os.path.join(TARGET_DIR, f"{parent_dir}_{fn}")
        )
    except Exception as e:
        print(e)
