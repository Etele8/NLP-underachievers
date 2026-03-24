import os
import zipfile
from pathlib import Path

DATASET = "thedevastator/unlock-universal-language-with-the-lince-dataset"
TARGET_FILES = {
    "ner_spaeng_test.csv",
    "ner_spaeng_train.csv",
    "ner_spaeng_validation.csv",
}

def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    os.system(f"kaggle datasets download -d {DATASET} -p {raw_dir}")

    zip_path = raw_dir / "unlock-universal-language-with-the-lince-dataset.zip"

    if not zip_path.exists():
        raise FileNotFoundError("Download failed. ZIP not found.")

    print("📦 Extracting required files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if Path(member).name in TARGET_FILES:
                zip_ref.extract(member, raw_dir)

    print("Cleaning up ZIP...")
    zip_path.unlink()

    print("Dataset ready at:", raw_dir)


if __name__ == "__main__":
    main()
