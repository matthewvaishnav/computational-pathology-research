"""Extract CRC-VAL-HE-7K validation dataset."""
import zipfile
from pathlib import Path
from tqdm import tqdm

zip_path = Path("data/multi_disease/colon/CRC-VAL-HE-7K.zip")
extract_dir = Path("data/multi_disease/colon")

print(f"Extracting {zip_path}")
print(f"To: {extract_dir}")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    members = zip_ref.namelist()
    for member in tqdm(members, desc="Extracting"):
        try:
            zip_ref.extract(member, extract_dir)
        except Exception as e:
            print(f"Warning: Could not extract {member}: {e}")

print("\nExtraction complete!")
