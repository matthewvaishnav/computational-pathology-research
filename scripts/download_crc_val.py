"""Download CRC-VAL-HE-7K validation dataset from Zenodo."""
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)

if __name__ == "__main__":
    # CRC-VAL-HE-7K validation dataset
    url = "https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip"
    output_dir = Path("data/multi_disease/colon")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "CRC-VAL-HE-7K.zip"
    
    print(f"Downloading {url}")
    print(f"Output: {output_path}")
    
    download_file(url, output_path)
    
    print("\nDownload complete!")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
