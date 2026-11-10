"""Bounded random-access reader for ScanComp's single-IFD *.big.tiff exports.

ScanComp's public BigTIFFs have been observed to expose a readable first image
IFD followed by a malformed second IFD.  TiffFile.series and OpenSlide can fail
while first-IFD compressed tiles remain potentially readable.  This module
*never* enumerates TIFF pages, reconstructs a full WSI, or repairs the source.
It deliberately exposes only level 0 until a genuine pyramid is validated.

This is an acquisition data-admission adapter, not a registration, color
calibration, or biological ground-truth implementation.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

# Explicit filename namespace: no fallback for unrelated TIFFs.
_SCANCOMP_NAME = re.compile(r"^[a-z]{3}[0-9]{3}_(?:hms360|hstp1000|lcap450|phufs)\.big\.tiff$", re.I)
_MAX_COMPRESSED_SEGMENT_BYTES = 16 * 1024 * 1024
_MAX_DECODED_SEGMENT_BYTES = 64 * 1024 * 1024
_MAX_REGION_PIXELS = 4_194_304
_MAX_SEGMENTS_PER_REGION = 64


def is_scancomp_tiff(path: Union[str, Path]) -> bool:
    """Match public ScanComp filenames, not generic .tiff files."""
    return bool(_SCANCOMP_NAME.fullmatch(Path(path).name))


class ScanCompTiffReader:
    """Open TIFF IFD 0 and expose OpenSlide-like bounded level-0 RGB patches.

    Requires tifffile and an appropriate segment codec (often imagecodecs).
    All region coordinates are level-0 *pixel* coordinates; no scanner-to-scanner
    spatial correspondence is assumed.
    """

    def __init__(self, path: Union[str, Path]):
        import tifffile  # optional dependency; import only for this fallback

        self.path = Path(path)
        if not is_scancomp_tiff(self.path):
            raise ValueError("ScanComp adapter requires a public ScanComp *.big.tiff filename")
        self._tiff = None
        self._source = None
        try:
            self._tiff = tifffile.TiffFile(str(self.path))
            if not self._tiff.is_bigtiff:
                raise ValueError("ScanComp source is not BigTIFF")
            # Access ONLY first IFD; never len(t.pages), t.series, or page.asarray().
            self._page = self._tiff.pages[0]
            p = self._page
            self.width, self.height = int(p.imagewidth), int(p.imagelength)
            self.tile_width, self.tile_height = int(p.tilewidth or 0), int(p.tilelength or 0)
            if self.width <= 0 or self.height <= 0:
                raise ValueError("Invalid first image dimensions")
            if not p.is_tiled or self.tile_width <= 0 or self.tile_height <= 0:
                raise ValueError("FIRST_IFD_NOT_TILED: bounded strip reader is not implemented")
            if int(p.planarconfig) != 1 or int(p.samplesperpixel) not in (3, 4):
                raise ValueError("Unsupported planar configuration or RGB channel count")
            # 2 = RGB, 6 = YCbCr (codec must deliver decoded RGB to the caller).
            if int(p.photometric) not in (2, 6) or p.dtype != np.dtype('uint8'):
                raise ValueError("Expected 8-bit RGB/YCbCr image data")
            self._columns = math.ceil(self.width / self.tile_width)
            self._rows = math.ceil(self.height / self.tile_height)
            self._offsets = tuple(int(x) for x in p.dataoffsets)
            self._counts = tuple(int(x) for x in p.databytecounts)
            if len(self._offsets) != self._columns * self._rows or len(self._counts) != len(self._offsets):
                raise ValueError("First-IFD source-tile layout does not match image dimensions")
            if self.tile_width * self.tile_height * int(p.samplesperpixel) > _MAX_DECODED_SEGMENT_BYTES:
                raise ValueError("One decoded source tile would exceed 64 MiB")
            self._file_size = self.path.stat().st_size
            self._source = self.path.open('rb')
        except BaseException:
            self.close()
            raise

    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def level_count(self) -> int:
        return 1

    @property
    def level_dimensions(self):
        return [self.dimensions]

    @property
    def level_downsamples(self):
        return [1.0]

    @property
    def properties(self) -> Dict[str, str]:
        # No patient or case ID inference; filenames identify physical slides only.
        return {
            'scancomp.adapter': 'first-ifd-bounded',
            'scancomp.slide-alias': self.path.name.split('_', 1)[0],
            'scancomp.scanner-alias': self.path.name.rsplit('_', 1)[1].removesuffix('.big.tiff'),
            'scancomp.registration': 'not-verified',
            'scancomp.pyramid': 'not-verified',
        }

    def _decode_tile(self, index: int) -> np.ndarray:
        if self._source is None:
            raise RuntimeError("Reader is closed")
        offset, bytecount = self._offsets[index], self._counts[index]
        if not (0 < bytecount <= _MAX_COMPRESSED_SEGMENT_BYTES):
            raise ValueError(f"Compressed tile size outside safety bounds: {bytecount}")
        if offset < 16 or offset + bytecount > self._file_size:
            raise ValueError(f"Compressed tile outside file: {offset}+{bytecount}")
        self._source.seek(offset)
        encoded = self._source.read(bytecount)
        if len(encoded) != bytecount:
            raise IOError("Short read from compressed TIFF tile")
        p = self._page
        kwargs = {}
        if int(p.compression) in (6, 7, 34892, 33007):
            kwargs['jpegtables'] = p.jpegtables
            kwargs['jpegheader'] = p.jpegheader
        tile, _position, _shape = p.decode(encoded, index, **kwargs)
        if tile is None:
            raise ValueError("TIFF tile codec returned no pixels")
        array = np.asarray(tile)
        if array.nbytes > _MAX_DECODED_SEGMENT_BYTES:
            raise ValueError("Decoded source tile exceeds memory cap")
        # tifffile commonly returns (1, H, W, C).
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[2] not in (3, 4) or array.dtype != np.uint8:
            raise ValueError(f"Unexpected decoded tile RGB shape/dtype: {array.shape}/{array.dtype}")
        if array.shape[0] > self.tile_height or array.shape[1] > self.tile_width:
            raise ValueError("Decoded segment dimensions exceed TIFF tile geometry")
        return array[:, :, :3]

    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> np.ndarray:
        if level != 0:
            raise ValueError("Only TIFF IFD 0 is validated; no artificial pyramid levels")
        x, y = (int(v) for v in location)
        w, h = (int(v) for v in size)
        if w <= 0 or h <= 0 or w * h > _MAX_REGION_PIXELS:
            raise ValueError("Requested region is empty or exceeds 4 MiPixels")
        if x < 0 or y < 0 or x + w > self.width or y + h > self.height:
            raise ValueError("Requested region must lie fully inside the original image")
        col0, col1 = x // self.tile_width, (x + w - 1) // self.tile_width
        row0, row1 = y // self.tile_height, (y + h - 1) // self.tile_height
        n_tiles = (col1 - col0 + 1) * (row1 - row0 + 1)
        if n_tiles > _MAX_SEGMENTS_PER_REGION:
            raise ValueError("Requested region spans too many source tiles")
        # Track exact coverage: a short/partially decoded source tile must never
        # leave uninitialized output bytes masquerading as valid image pixels.
        out = np.zeros((h, w, 3), dtype=np.uint8)
        covered = np.zeros((h, w), dtype=np.bool_)
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                tile = self._decode_tile(row * self._columns + col)
                sx, sy = col * self.tile_width, row * self.tile_height
                left, top = max(x, sx), max(y, sy)
                right, bottom = min(x + w, sx + tile.shape[1]), min(y + h, sy + tile.shape[0])
                if right <= left or bottom <= top:
                    raise ValueError("Source tile does not cover expected region")
                out[top-y:bottom-y, left-x:right-x] = tile[top-sy:bottom-sy, left-sx:right-sx]
                covered[top-y:bottom-y, left-x:right-x] = True
        if not np.all(covered):
            raise ValueError("INCOMPLETE_SOURCE_TILE_COVERAGE: requested RGB region has undecoded pixels")
        return out

    def close(self) -> None:
        if getattr(self, '_source', None) is not None:
            self._source.close()
            self._source = None
        if getattr(self, '_tiff', None) is not None:
            self._tiff.close()
            self._tiff = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
