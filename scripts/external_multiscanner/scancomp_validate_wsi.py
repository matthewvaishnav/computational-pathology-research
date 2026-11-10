#!/usr/bin/env python3
"""Read-only ScanComp WSI intake through the repository's canonical WSIReader.

This is a pixel decode and source-file provenance check; contact sheets use
*unregistered* relative image coordinates. It does not train or evaluate PA-NF.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.wsi_pipeline.reader import WSIReader

SCANNERS = ("hms360", "hstp1000", "lcap450", "phufs")
POSITIONS = (0.15, 0.35, 0.55, 0.75, 0.90)


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_one(path: Path, out_dir: Path, patch_size: int, do_hash: bool) -> dict:
    info: dict = {'file_name': path.name, 'exists': path.is_file(), 'status': 'BLOCKED'}
    if not path.is_file():
        info['error'] = 'FILE_MISSING'
        return info
    info['size_bytes'] = path.stat().st_size
    if do_hash:
        info['sha256'] = checksum(path)
    try:
        with WSIReader(path) as reader:
            info['backend'] = reader._format
            info['dimensions'] = list(reader.dimensions)
            info['level_dimensions'] = [list(x) for x in reader.level_dimensions]
            info['properties'] = reader.properties
            width, height = reader.dimensions
            if min(width, height) < patch_size:
                raise RuntimeError('Full-resolution image smaller than requested patch')
            preview = Image.new('RGB', (5 * 160, 5 * 160), (242, 242, 242))
            cells = []
            for row, yf in enumerate(POSITIONS):
                for col, xf in enumerate(POSITIONS):
                    x = min(width - patch_size, int(xf * width))
                    y = min(height - patch_size, int(yf * height))
                    cell = {'x_level0': x, 'y_level0': y, 'width': patch_size,
                            'height': patch_size, 'row': row, 'col': col}
                    try:
                        rgb = reader.read_region((x, y), level=0, size=(patch_size, patch_size))
                        if rgb.shape != (patch_size, patch_size, 3) or rgb.dtype != np.uint8:
                            raise ValueError(f'Invalid patch shape/dtype {rgb.shape}/{rgb.dtype}')
                        cell['status'] = 'PIXELS_DECODED'
                        cell['rgb_mean'] = [round(float(v), 3) for v in rgb.mean(axis=(0, 1))]
                        sample = Image.fromarray(rgb)
                        sample.thumbnail((160, 160), Image.Resampling.BILINEAR)
                        preview.paste(sample, (col * 160, row * 160))
                    except Exception as exc:
                        cell['status'] = 'BLOCKED'
                        cell['error'] = f'{type(exc).__name__}: {exc}'[:500]
                        ImageDraw.Draw(preview).text((col * 160 + 4, row * 160 + 4),
                                                     'BLOCKED', fill=(160, 0, 0))
                    cells.append(cell)
            info['patches'] = cells
            info['decoded_count'] = sum(c['status'] == 'PIXELS_DECODED' for c in cells)
            info['failed_count'] = len(cells) - info['decoded_count']
            image_name = f'{path.stem}_UNREGISTERED_sparse_contact_sheet.png'
            preview.save(out_dir / image_name)
            info['preview'] = image_name
            info['status'] = 'DECODE_PASS' if info['decoded_count'] == len(cells) else 'BLOCKED'
    except Exception as exc:
        info['error'] = f'{type(exc).__name__}: {exc}'[:1000]
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Directory containing the four downloaded *.big.tiff files')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='Directory for metadata, reports, and sparse contact sheets')
    parser.add_argument('--slide-id', default='hmi825',
                        help='Public ScanComp slide alias, e.g. hmi825')
    parser.add_argument('--patch-size', type=int, default=224)
    parser.add_argument('--sha256', action='store_true',
                        help='Stream/hash each original file (optional and read-only)')
    args = parser.parse_args(argv)
    if not __import__('re').fullmatch(r'[a-z]{3}[0-9]{3}', args.slide_id):
        parser.error('--slide-id must match ScanComp public alias pattern')
    if not (16 <= args.patch_size <= 1024):
        parser.error('--patch-size must be 16..1024')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {'date_utc': datetime.now(timezone.utc).isoformat(),
               'slide_id': args.slide_id, 'inputs': str(args.data_dir),
               'stage': 'SOURCE_PIXEL_DECODE_ONLY',
               'registration': 'NOT_PERFORMED',
               'independent_biological_labels': 'NOT_AVAILABLE',
               'clinical_G0': 'DATA_BLOCKED',
               'images': {}}
    for scanner in SCANNERS:
        filename = f'{args.slide_id}_{scanner}.big.tiff'
        print(f'[SCAN] {filename}', flush=True)
        info = validate_one(args.data_dir / filename, args.out_dir, args.patch_size, args.sha256)
        receipt['images'][scanner] = info
        print(f"  {info['status']} {info.get('decoded_count', 0)}/25 regions; "
              f"{info.get('error', '')}", flush=True)
    success = all(x['status'] == 'DECODE_PASS' for x in receipt['images'].values())
    receipt['all_four_decode_pass'] = success
    (args.out_dir / 'SCANCOMP_PIXEL_DECODE_RECEIPT.json').write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding='utf-8')
    (args.out_dir / 'README_RESULTS.txt').write_text(
        'ScanComp first-IFD source pixel check. 25 sparse patch decodes per scanner.\n'
        'Contact sheets are NOT registrations, whole-slide thumbnails, or biology labels.\n'
        f'All four decode PASS: {success}\n'
        'G0 remains DATA_BLOCKED independently of this pixel intake result.\n',
        encoding='utf-8')
    print(f"Wrote receipt to {args.out_dir} (ALL_FOUR={'PASS' if success else 'BLOCKED'})")
    return 0 if success else 2


if __name__ == '__main__':
    raise SystemExit(main())
