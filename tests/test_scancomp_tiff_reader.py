"""Small synthetic BigTIFF regression tests; never treat these as real ScanComp QC."""

from __future__ import annotations

import struct

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
from src.data.wsi_pipeline.scancomp_tiff import ScanCompTiffReader, is_scancomp_tiff


def make_slide(tmp_path, name="hmi825_hms360.big.tiff", broken_next=False):
    arr = np.zeros((96, 128, 3), dtype=np.uint8)
    yy, xx = np.indices((96, 128))
    arr[..., 0] = xx.astype("uint8")
    arr[..., 1] = yy.astype("uint8")
    arr[..., 2] = 117
    path = tmp_path / name
    tifffile.imwrite(
        path,
        arr,
        bigtiff=True,
        photometric="rgb",
        tile=(16, 16),
        compression="deflate",
        metadata=None,
    )
    if broken_next:
        # Mutate the first IFD's next-pointer only; leave valid image tile data alone.
        with path.open("r+b") as handle:
            header = handle.read(16)
            assert header[:2] == b"II" and struct.unpack_from("<H", header, 2)[0] == 43
            first = struct.unpack_from("<Q", header, 8)[0]
            handle.seek(first)
            count = struct.unpack("<Q", handle.read(8))[0]
            assert count < 4096
            handle.seek(first + 8 + count * 20)
            handle.write(struct.pack("<Q", path.stat().st_size - 16))
            handle.seek(path.stat().st_size - 16)
            handle.write(struct.pack("<Q", 80_966_811_461))
    return path, arr


def test_scope_matches_only_scancomp_names():
    assert is_scancomp_tiff("hmi825_phufs.big.tiff")
    assert not is_scancomp_tiff("fake_slide.big.tiff")
    assert not is_scancomp_tiff("slide.svs")


@pytest.mark.parametrize("broken", [False, True])
def test_bounded_region_matches_synthetic_source(tmp_path, broken):
    path, pixels = make_slide(tmp_path, broken_next=broken)
    with ScanCompTiffReader(path) as slide:
        assert slide.level_count == 1
        assert slide.dimensions == (128, 96)
        result = slide.read_region((9, 13), 0, (37, 29))
        assert np.array_equal(result, pixels[13:42, 9:46])
        assert slide.properties["scancomp.registration"] == "not-verified"


def test_reject_unbounded_reads_and_fake_pyramid(tmp_path):
    path, _ = make_slide(tmp_path)
    with ScanCompTiffReader(path) as slide:
        with pytest.raises(ValueError, match="level"):
            slide.read_region((0, 0), 1, (16, 16))
        with pytest.raises(ValueError, match="inside"):
            slide.read_region((127, 0), 0, (16, 16))
        with pytest.raises(ValueError, match="4 MiPixels"):
            slide.read_region((0, 0), 0, (4096, 4096))


def test_reject_stripped_source(tmp_path):
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    path = tmp_path / "hmi825_hms360.big.tiff"
    tifffile.imwrite(path, arr, bigtiff=True, photometric="rgb", metadata=None)
    with pytest.raises(ValueError, match="FIRST_IFD_NOT_TILED"):
        ScanCompTiffReader(path)


def test_canonical_wsi_reader_fallback_without_openslide(tmp_path, monkeypatch):
    from src.data.wsi_pipeline import reader as wsi_module
    from src.data.wsi_pipeline.exceptions import ProcessingError

    path, pixels = make_slide(tmp_path, broken_next=True)
    monkeypatch.setattr(wsi_module, "OPENSLIDE_AVAILABLE", False)
    with wsi_module.WSIReader(path) as reader:
        assert reader._format == "scancomp_tiff"
        assert reader.level_count == 1
        assert reader.level_downsamples == [1.0]
        assert reader.properties["scancomp.registration"] == "not-verified"
        assert np.array_equal(reader.read_region((8, 8), 0, (24, 32)), pixels[8:40, 8:32])
        with pytest.raises(ProcessingError, match="no verified image pyramid"):
            reader.get_thumbnail()


def test_non_scancomp_tiff_does_not_use_fallback(tmp_path, monkeypatch):
    from src.data.wsi_pipeline import reader as wsi_module
    from src.data.wsi_pipeline.exceptions import FileFormatError

    path, _ = make_slide(tmp_path, name="ordinary.tiff")
    monkeypatch.setattr(wsi_module, "OPENSLIDE_AVAILABLE", False)
    with pytest.raises(FileFormatError, match="OpenSlide is not installed"):
        wsi_module.WSIReader(path)


def test_incomplete_decode_rejected(tmp_path, monkeypatch):
    path, _ = make_slide(tmp_path)
    with ScanCompTiffReader(path) as slide:
        original_decode = slide._decode_tile

        def short_first_tile(index):
            decoded = original_decode(index)
            if index == 0:
                return decoded[:-1, :, :]
            return decoded

        monkeypatch.setattr(slide, "_decode_tile", short_first_tile)
        with pytest.raises(ValueError, match="INCOMPLETE_SOURCE_TILE_COVERAGE"):
            slide.read_region((0, 0), 0, (16, 16))


def test_incomplete_decode_does_not_escape_through_canonical_reader(tmp_path, monkeypatch):
    from src.data.wsi_pipeline import reader as wsi_module
    from src.data.wsi_pipeline.exceptions import ProcessingError

    path, _ = make_slide(tmp_path)
    monkeypatch.setattr(wsi_module, "OPENSLIDE_AVAILABLE", False)
    with wsi_module.WSIReader(path) as reader:
        original_decode = reader._reader._decode_tile

        def short_first_tile(index):
            decoded = original_decode(index)
            return decoded[:-1, :, :] if index == 0 else decoded

        monkeypatch.setattr(reader._reader, "_decode_tile", short_first_tile)
        with pytest.raises(ProcessingError, match="INCOMPLETE_SOURCE_TILE_COVERAGE"):
            reader.read_region((0, 0), 0, (16, 16))
