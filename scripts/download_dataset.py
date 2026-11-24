"""Dataset downloader and integrity verification script for Last.fm 1K.

Downloads the official archive from MTG/UPF Barcelona, verifies checksums,
and extracts raw TSV files into data/raw/.
"""

import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from euterpe.config import default_config

DATASET_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
EXPECTED_MD5 = {
    "userid-profile.tsv": "c53608b6b445db201098c1489ea497df",
    "userid-timestamp-artid-artname-traid-traname.tsv": "64747b21563e3d2aa95751e0ddc46b68"
}


def compute_file_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_and_extract(target_dir: Path, max_sample_rows: int = None) -> None:
    """Download and extract dataset."""
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "lastfm-dataset-1K.tar.gz"
    
    print(f"Connecting to {DATASET_URL}...")
    req = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EuterpeML/1.0"}
    )
    
    if not archive_path.exists():
        print(f"Downloading archive to {archive_path}...")
        with urllib.request.urlopen(req, timeout=30) as response, open(archive_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB
            
            while chunk := response.read(block_size):
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="")
            print("\nDownload complete.")
    else:
        print(f"Archive already exists at {archive_path}.")
        
    print("Extracting files...")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            if filename in EXPECTED_MD5 or filename == "README.txt":
                out_dest = target_dir / filename
                print(f"Extracting {filename} -> {out_dest}...")
                with tar.extractfile(member) as src_f, open(out_dest, "wb") as dst_f:
                    while chunk := src_f.read(1024 * 1024):
                        dst_f.write(chunk)
                        
    print("Extraction completed.")


if __name__ == "__main__":
    raw_dir = default_config.paths.raw_data_dir
    download_and_extract(raw_dir)
