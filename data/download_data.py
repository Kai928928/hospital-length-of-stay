"""Download and extract the UCI Diabetes 130-US Hospitals dataset.

Source: https://archive.ics.uci.edu/dataset/296/diabetes

The archive contains `diabetic_data.csv` (101,766 inpatient encounters) which
this script extracts into the project `data/` directory.

Usage:
    python data/download_data.py
"""
import io
import sys
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGET_CSV = DATA_DIR / "diabetic_data.csv"

# Primary: UCI static file mirror. Fallback: the UCI dataset page itself.
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/296/diabetes"
STATIC_URLS = [
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}


def _download_bytes(url: str, timeout: int = 90) -> bytes:
    print(f"  downloading {url}")
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def _extract_diabetic_csv(archive_bytes: bytes) -> bytes:
    """Locate diabetic_data.csv inside the downloaded zip archive."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        candidates = [n for n in zf.namelist() if n.endswith("diabetic_data.csv")]
        if not candidates:
            raise FileNotFoundError("diabetic_data.csv not found inside archive")
        name = candidates[0]
        print(f"  found {name} in archive")
        return zf.read(name)


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_CSV.exists():
        print(f"already present: {TARGET_CSV} — skipping download")
        return 0

    last_err = None
    for url in STATIC_URLS:
        try:
            content = _download_bytes(url)
            csv_bytes = _extract_diabetic_csv(content)
            TARGET_CSV.write_bytes(csv_bytes)
            size_mb = TARGET_CSV.stat().st_size / 1e6
            print(f"OK: {TARGET_CSV} ({size_mb:.1f} MB)")
            return 0
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            print(f"  failed ({type(exc).__name__}: {exc})")
            last_err = exc

    # Fallback: instruct the user to download manually from the dataset page.
    print(
        "\nAutomatic download failed. Please download the dataset manually:\n"
        f"  1. Open {UCI_DATASET_PAGE}\n"
        "  2. Click 'Download' and unzip the archive\n"
        f"  3. Place diabetic_data.csv into: {DATA_DIR}\n"
    )
    print(f"Last error: {type(last_err).__name__}: {last_err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
