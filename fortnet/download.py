"""Download and verify the FORTMCF split archives."""

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from .assets import DIRECTORY, sha256
from .data import CONDITIONS

REPOSITORY = "LoyCeavy11/FORTMCF"
REVISION = "5d9c39e501b6df66a037165be36c58922cc472f1"


def extract_archive(archive, destination, relative_name, count):
    condition, split_zip = relative_name.split("/")
    split = split_zip.removesuffix(".zip")
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        pattern = re.compile(rf"{re.escape(condition)}/{re.escape(split)}/[0-9]{{6}}\.png")
        if (
            len(entries) != count
            or len(set(names)) != count
            or any(not pattern.fullmatch(n) for n in names)
        ):
            raise ValueError(f"Unexpected archive members: {relative_name}")
        for entry in entries:
            target = destination.joinpath(*PurePosixPath(entry.filename).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Archive destination escapes the dataset directory.")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".part")
            with bundle.open(entry) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(target)


def main():
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/FORTMCF"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["train", "validation", "test"],
    )
    parser.add_argument(
        "--all-exposure-train",
        action="store_true",
        help="Also download the four real extreme training splits; they are not neural training inputs.",
    )
    args = parser.parse_args()
    manifest = json.loads((DIRECTORY / "archives.json").read_text())
    for split in dict.fromkeys(args.splits):
        conditions = (
            (*CONDITIONS, "gt")
            if split != "train" or args.all_exposure_train
            else ("nominal", "gt")
        )
        for condition in conditions:
            name = f"{condition}/{split}.zip"
            expected = manifest[name]
            print(f"Downloading {name}", flush=True)
            archive = hf_hub_download(REPOSITORY, name, repo_type="dataset", revision=REVISION)
            if (
                Path(archive).stat().st_size != expected["bytes"]
                or sha256(archive) != expected["sha256"]
            ):
                raise ValueError(f"Archive checksum mismatch: {name}")
            extract_archive(archive, args.output, name, expected["images"])
            print(f"Extracted {name}: {expected['images']} images", flush=True)
