"""Download the Kaggle GTSRB archive into this repo's dataset directory.

The loaders read dataset/Test.csv, dataset/<Path>, and dataset/Train/<class>/.
Run it from the repository root. On Colab, run it inside the notebook kernel
so an upload dialog can open:

    %run download_dataset.py

A token from https://www.kaggle.com/settings/api also works when
KAGGLE_API_TOKEN is set before the script starts. A legacy kaggle.json in
~/.kaggle/ is accepted too.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
REQUIRED = ("Test.csv", "Train", "Test")
OPTIONAL = ("Train.csv", "Meta.csv", "Meta")


def dataset_dir():
    return Path(__file__).resolve().parent / "dataset"


def ensure_kaggle_package():
    """The current package accepts both a KGAT token and a legacy kaggle.json."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle"])


def ensure_credentials():
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_file = kaggle_dir / "access_token"
    legacy_file = kaggle_dir / "kaggle.json"
    if os.environ.get("KAGGLE_API_TOKEN") or token_file.is_file() or legacy_file.is_file():
        for path in (token_file, legacy_file):
            if path.is_file():
                path.chmod(0o600)
        return

    try:
        from google.colab import files
    except ImportError:
        raise SystemExit(
            "No Kaggle credentials. Set KAGGLE_API_TOKEN, save the token in "
            "~/.kaggle/access_token, or save a legacy key as ~/.kaggle/kaggle.json. "
            "Create either one at https://www.kaggle.com/settings/api."
        )

    print("Upload one file from https://www.kaggle.com/settings/api:")
    print("  access_token — contents are the KGAT token")
    print("  kaggle.json  — Legacy API key")
    uploaded = files.upload()
    if not uploaded:
        raise SystemExit("No file uploaded.")
    name, payload = next(iter(uploaded.items()))
    text = payload.decode("utf-8").strip()
    if text.startswith("KGAT_") or name == "access_token":
        token_file.write_text(text + "\n", encoding="utf-8")
        token_file.chmod(0o600)
        return
    legacy_file.write_bytes(payload)
    legacy_file.chmod(0o600)


def find_archive_root(extracted):
    """Return the directory that contains Test.csv beside Train/ and Test/."""
    for current, dirnames, filenames in os.walk(extracted):
        if "Test.csv" in filenames and "Train" in dirnames and "Test" in dirnames:
            return Path(current)
    raise SystemExit(f"The archive at {extracted} has no Test.csv next to Train/ and Test/.")


def place_dataset(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    for name in REQUIRED + OPTIONAL:
        src = source / name
        if not src.exists():
            if name in REQUIRED:
                raise SystemExit(f"Missing {name} in {source}")
            continue
        dest = destination / name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(src), str(dest))


def check_dataset(destination):
    csv_path = destination / "Test.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Path" not in rows[0] or "ClassId" not in rows[0]:
        raise SystemExit(f"{csv_path} has no Path and ClassId columns.")

    missing = [row["Path"] for row in rows if not (destination / row["Path"]).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"{len(missing)} test paths from Test.csv are missing. First: {preview}")

    class_names = {path.name for path in (destination / "Train").iterdir() if path.is_dir()}
    expected = {str(class_id) for class_id in range(43)}
    absent = sorted(expected - class_names, key=int)
    if absent:
        raise SystemExit(f"dataset/Train is missing class directories: {', '.join(absent)}")

    print(f"dataset/Test.csv: {len(rows)} images")
    print(f"dataset/Train: {len(expected)} class directories")
    print(f"Ready at {destination}")


def already_ready(destination):
    try:
        check_dataset(destination)
    except (SystemExit, OSError):
        return False
    return True


def download(destination):
    ensure_kaggle_package()
    ensure_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory() as temporary:
        api.dataset_download_files(DATASET, path=temporary, unzip=True, quiet=False)
        place_dataset(find_archive_root(Path(temporary)), destination)


def main():
    parser = argparse.ArgumentParser(description="Download GTSRB into dataset/.")
    parser.add_argument("--force", action="store_true", help="Download again even if dataset/ is already complete.")
    args = parser.parse_args()
    destination = dataset_dir()
    if args.force or not already_ready(destination):
        download(destination)
        check_dataset(destination)


if __name__ == "__main__":
    main()
