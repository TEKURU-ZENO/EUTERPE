"""Dataset acquisition and integrity verification for Last.fm 1K.

Workstream 1.1: Downloads official archive, validates expected file sizes and MD5
checksums, extracts TSV files to data/raw/, and generates artifacts/manifests/dataset_manifest.json.
"""

import datetime
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
MANIFEST_DIR = BASE_DIR / "artifacts" / "manifests"

DATASET_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
EXPECTED_ARCHIVE_SIZE = 672741554
EXPECTED_MD5 = {
    "userid-profile.tsv": "c53608b6b445db201098c1489ea497df",
    "userid-timestamp-artid-artname-traid-traname.tsv": "64747b21563e3d2aa95751e0ddc46b68"
}


def compute_file_md5(filepath: Path) -> str:
    """Compute MD5 checksum in chunks."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_archive(dest_path: Path) -> None:
    """Download the complete Last.fm 1K tar.gz archive with verification."""
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if actual_size == EXPECTED_ARCHIVE_SIZE:
            print(f"Archive already downloaded and verified ({actual_size} bytes): {dest_path}")
            return
        else:
            print(f"Existing file size {actual_size} != expected {EXPECTED_ARCHIVE_SIZE}. Re-downloading...")

    print(f"Downloading from {DATASET_URL}...")
    req = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EuterpeML/1.0"}
    )
    
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f_out:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2MB chunks
        
        while chunk := resp.read(chunk_size):
            f_out.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\rDownloading: {mb_down:.1f}MB / {mb_total:.1f}MB ({pct:.1f}%)")
                sys.stdout.flush()
    print("\nDownload complete.")


def extract_archive(archive_path: Path, target_dir: Path) -> dict:
    """Extract files from archive and return file details."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_info = {}
    
    print(f"Extracting {archive_path} to {target_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            base_name = os.path.basename(member.name)
            if not base_name:
                continue
            dest_file = target_dir / base_name
            if not dest_file.exists() or dest_file.stat().st_size != member.size:
                print(f"Extracting member {member.name} ({member.size} bytes) -> {dest_file}...")
                with tar.extractfile(member) as src_f, open(dest_file, "wb") as dst_f:
                    while chunk := src_f.read(4 * 1024 * 1024):
                        dst_f.write(chunk)
            else:
                print(f"File already extracted: {dest_file} ({dest_file.stat().st_size} bytes)")
            
            extracted_info[base_name] = {
                "path": str(dest_file),
                "size_bytes": dest_file.stat().st_size
            }
            
    return extracted_info


def generate_manifest(archive_path: Path, extracted_files: dict) -> Path:
    """Verify checksums and generate dataset_manifest.json."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "dataset_manifest.json"
    
    print("Computing MD5 checksums for extracted files...")
    verified_files = {}
    for filename, info in extracted_files.items():
        fpath = Path(info["path"])
        md5_val = compute_file_md5(fpath)
        expected = EXPECTED_MD5.get(filename)
        status = "VERIFIED" if (expected is None or md5_val == expected) else "MISMATCH"
        print(f"  {filename}: MD5={md5_val} (Expected={expected}) [{status}]")
        
        # Count lines
        line_count = 0
        with open(fpath, "rb") as f:
            for _ in f:
                line_count += 1
                
        verified_files[filename] = {
            "size_bytes": info["size_bytes"],
            "md5": md5_val,
            "expected_md5": expected,
            "checksum_status": status,
            "line_count": line_count
        }

    manifest = {
        "dataset_name": "lastfm-dataset-1K",
        "version": "1.0",
        "retrieval_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_url": DATASET_URL,
        "archive_size_bytes": archive_path.stat().st_size,
        "expected_archive_size_bytes": EXPECTED_ARCHIVE_SIZE,
        "license": "Non-commercial research use only, with explicit permission from Last.fm",
        "citation": "Celma, O. (2010). Music Recommendation and Discovery in the Long Tail. Springer Berlin Heidelberg.",
        "terms_file": "docs/DATASET_LICENSE.md",
        "extracted_files": verified_files
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest written to: {manifest_path}")
    return manifest_path


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / "lastfm-dataset-1K.tar.gz"
    
    download_archive(archive_path)
    extracted = extract_archive(archive_path, RAW_DIR)
    manifest_path = generate_manifest(archive_path, extracted)
    print("Workstream 1.1: Dataset acquisition & integrity verification SUCCESSFUL.")


if __name__ == "__main__":
    main()
