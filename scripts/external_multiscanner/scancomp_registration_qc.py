#!/usr/bin/env python3
"""Exploratory registration diagnostics for four public ScanComp H&E acquisitions.

No training, frozen-checkpoint inference, test-set admission or pixel-level
supervision is performed. Reads only bounded level-0 RGB source regions via the
repository's WSIReader (including ScanComp's first-IFD BigTIFF fallback).

Matching is a deliberately *local* cross-scanner translation search. A global
translation/similarity/affine map is fitted from train landmarks and evaluated
on deterministic held-out landmarks. This is a registration diagnostic, not
certification of fully registered patches or clinical biological labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCANNERS = ('hms360', 'hstp1000', 'lcap450', 'phufs')
REFERENCE = 'hms360'


def grid_positions(width: int, height: int, patch: int, n: int) -> list[dict[str, Any]]:
    if width <= patch + 4 or height <= patch + 4:
        raise ValueError('Slide dimensions too small for QC grid')
    if n < 3 or n > 19:
        raise ValueError('Grid must have 3..19 sites per axis')
    fractions = np.linspace(0.08, 0.92, n)
    rows = []
    for row, yf in enumerate(fractions):
        for col, xf in enumerate(fractions):
            x = int(round(xf * (width - patch)))
            y = int(round(yf * (height - patch)))
            rows.append({'grid_row': row, 'grid_col': col, 'x': x, 'y': y})
    return rows


def split_for(slide_id: str, row: int, col: int) -> str:
    # Frozen before reading target scanner data; independent of match quality.
    digest = hashlib.sha256(f'{slide_id}:{row}:{col}:scancomp-reg-v1'.encode()).digest()
    return 'heldout' if int.from_bytes(digest[:8], 'big') % 4 == 0 else 'fit'


def tissue_and_texture(rgb: np.ndarray, min_tissue: float, min_gray_std: float) -> tuple[bool, float, float]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB uint8 patch')
    fraction = float(np.mean(np.min(rgb, axis=2) < 210))
    spread = float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).std())
    return fraction >= min_tissue and spread >= min_gray_std, fraction, spread


def template_shift(ref_rgb: np.ndarray, target_window_rgb: np.ndarray, margin: int) -> tuple[int, int, float]:
    if margin <= 0:
        raise ValueError('Search margin must be positive')
    h, w = ref_rgb.shape[:2]
    if target_window_rgb.shape[:2] != (h + 2*margin, w + 2*margin):
        raise ValueError('Unexpected target search window dimensions')
    ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    target = cv2.cvtColor(target_window_rgb, cv2.COLOR_RGB2GRAY)
    if float(ref.std()) < 5.0:
        raise ValueError('Reference texture inadequate for matching')
    surface = cv2.matchTemplate(target, ref, cv2.TM_CCOEFF_NORMED)
    _minval, maxval, _minloc, peak = cv2.minMaxLoc(surface)
    if not np.isfinite(maxval):
        raise ValueError('Nonfinite normalized matching score')
    return int(peak[0] - margin), int(peak[1] - margin), float(maxval)


def predict(model: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return coordinates @ model[:, :2].T + model[:, 2][None, :]


def fit_models(points: np.ndarray, shifts: np.ndarray) -> dict[str, np.ndarray]:
    """Fit coordinate transforms target_pixel = M @ source_pixel.

    Coordinates are absolute level-0 source pixels and shifts are signed target
    offsets. A translation model is always fitted; similarity and full affine
    require sufficient nondegenerate training landmarks.
    """
    points = np.asarray(points, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    if len(points) != len(shifts) or points.shape[-1] != 2 or shifts.shape[-1] != 2:
        raise ValueError('Expected Nx2 points and shifts')
    if len(points) < 3:
        raise ValueError('At least three fit landmarks required')
    offset = np.median(shifts, axis=0)
    trans = np.array([[1., 0., offset[0]], [0., 1., offset[1]]], dtype=np.float64)
    out = {'translation': trans}
    targets = points + shifts
    # Exactly the same coordinate estimates are used for all competing models.
    try:
        similarity, _inliers = cv2.estimateAffinePartial2D(points, targets,
                            method=cv2.RANSAC, ransacReprojThreshold=5.0,
                            maxIters=3000, confidence=0.999, refineIters=10)
        if similarity is not None and np.all(np.isfinite(similarity)):
            out['similarity'] = np.asarray(similarity, dtype=np.float64)
    except cv2.error:
        pass
    try:
        affine, _inliers = cv2.estimateAffine2D(points, targets,
                            method=cv2.RANSAC, ransacReprojThreshold=5.0,
                            maxIters=3000, confidence=0.999, refineIters=10)
        if affine is not None and np.all(np.isfinite(affine)):
            out['affine'] = np.asarray(affine, dtype=np.float64)
    except cv2.error:
        pass
    return out


def metrics_for(models: dict[str, np.ndarray], held_points: np.ndarray, held_shifts: np.ndarray) -> dict[str, Any]:
    held_points = np.asarray(held_points, dtype=np.float64)
    held_shifts = np.asarray(held_shifts, dtype=np.float64)
    targets = held_points + held_shifts
    out: dict[str, Any] = {}
    identity = held_points
    candidates = {'identity': identity}
    for name, model in models.items():
        candidates[name] = predict(model, held_points)
    for name, estimate in candidates.items():
        error = np.linalg.norm(estimate - targets, axis=1)
        out[name] = {
            'n': int(len(error)),
            'median_level0_pixels': float(np.median(error)),
            'p90_level0_pixels': float(np.percentile(error, 90)),
            'max_level0_pixels': float(error.max()),
            'per_holdout_level0_pixels': [float(v) for v in error],
        }
    return out


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    from src.data.wsi_pipeline.reader import WSIReader

    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'ScanComp public H&E hmi825 four-scanner WSI',
        'slide_alias': args.slide_id,
        'stage': 'EXPLORATORY_L0_REGISTRATION_DIAGNOSTIC',
        'G0': 'DATA_BLOCKED_INDEPENDENT_CASE_AND_CLINICAL_LABELS',
        'verified_physical_registration': False,
        'frozen_encoder_evaluated': False,
        'patch': args.patch,
        'margin': args.margin,
        'grid': args.grid,
        'minimum_match_score': args.min_match,
        'reference': REFERENCE,
        'scanners': {},
    }
    rows = []
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            readers = {s: stack.enter_context(WSIReader(args.data_dir / f'{args.slide_id}_{s}.big.tiff')) for s in SCANNERS}
            dimensions = {s: list(reader.dimensions) for s, reader in readers.items()}
            receipt['dimensions'] = dimensions
            if len({tuple(d) for d in dimensions.values()}) != 1:
                raise RuntimeError('Four full-resolution source coordinate frames have different dimensions')
            width, height = readers[REFERENCE].dimensions
            grid = grid_positions(width, height, args.patch, args.grid)
            reference_candidates: list[tuple[dict, np.ndarray]] = []
            for site in grid:
                rgb = readers[REFERENCE].read_region((site['x'], site['y']), 0, (args.patch, args.patch))
                suitable, fraction, spread = tissue_and_texture(rgb, args.min_tissue, args.min_std)
                if suitable:
                    reference_candidates.append((site, rgb))
                rows.append({'scanner': REFERENCE, **site, 'split': split_for(args.slide_id, site['grid_row'], site['grid_col']),
                             'status': 'REFERENCE_TISSUE' if suitable else 'REFERENCE_SKIPPED',
                             'tissue_fraction': round(fraction, 5), 'gray_std': round(spread, 4)})
            receipt['reference_candidates'] = len(reference_candidates)
            print(f'[REG] {REFERENCE}: {len(reference_candidates)}/{len(grid)} texture-qualified tissue candidates', flush=True)
            if len(reference_candidates) < 8:
                raise RuntimeError('Fewer than eight tissue landmarks; increase grid or use tissue-guided sampling')
            for scanner in SCANNERS:
                if scanner == REFERENCE:
                    continue
                print(f'[REG] {scanner}: matching {len(reference_candidates)} reference positions', flush=True)
                scanner_rows = []
                for site, reference in reference_candidates:
                    x, y = site['x'], site['y']
                    info: dict[str, Any] = {'scanner': scanner, **site,
                           'split': split_for(args.slide_id,site['grid_row'],site['grid_col'])}
                    if x < args.margin or y < args.margin or x + args.patch + args.margin > width or y + args.patch + args.margin > height:
                        info['status'] = 'SEARCH_WINDOW_OUT_OF_BOUNDS'
                    else:
                        try:
                            other = readers[scanner].read_region((x-args.margin,y-args.margin),0,
                                                           (args.patch+2*args.margin,args.patch+2*args.margin))
                            dx, dy, score = template_shift(reference, other, args.margin)
                            info.update({'dx_level0_px':dx, 'dy_level0_px':dy, 'match_ncc':round(score,7),
                                         'status':'MATCH' if score>=args.min_match else 'LOW_MATCH_NCC'})
                        except Exception as exc:
                            info['status'] = 'MATCH_ERROR'
                            info['error'] = str(exc)[:400]
                    rows.append(info)
                    scanner_rows.append(info)
                fit=[x for x in scanner_rows if x['status']=='MATCH' and x['split']=='fit']
                held=[x for x in scanner_rows if x['status']=='MATCH' and x['split']=='heldout']
                stats: dict[str, Any] = {'accepted_fit':len(fit),'accepted_heldout':len(held),
                                 'all_candidates':len(scanner_rows),
                                 'failed_or_low_score':sum(x['status']!='MATCH' for x in scanner_rows)}
                if len(fit) >= 5 and len(held)>=3:
                    a=np.array([[e['x']+args.patch/2,e['y']+args.patch/2] for e in fit],dtype=np.float64)
                    da=np.array([[e['dx_level0_px'],e['dy_level0_px']] for e in fit],dtype=np.float64)
                    b=np.array([[e['x']+args.patch/2,e['y']+args.patch/2] for e in held],dtype=np.float64)
                    db=np.array([[e['dx_level0_px'],e['dy_level0_px']] for e in held],dtype=np.float64)
                    models=fit_models(a,da)
                    stats['models']={k:v.tolist() for k,v in models.items()}
                    stats['heldout_errors']=metrics_for(models,b,db)
                    stats['match_ncc_heldout']=[e['match_ncc'] for e in held]
                    stats['modeling_status']='HELDOUT_GEOMETRIC_DIAGNOSTIC_COMPUTED'
                    for e in held:
                        e['heldout_landmark']=True
                else:
                    stats['modeling_status']='INSUFFICIENT_FIT_OR_HELDOUT_MATCHES'
                receipt['scanners'][scanner]=stats
                print(f"  accepted fit={len(fit)}, heldout={len(held)}; {stats['modeling_status']}",flush=True)
    except Exception as exc:
        receipt['error']=f'{type(exc).__name__}: {exc}'
        print('[REG] BLOCKED',receipt['error'],flush=True)
    finally:
        save_csv(args.out_dir/'SCANCOMP_REGISTRATION_LANDMARKS.csv',rows)
        (args.out_dir/'SCANCOMP_REGISTRATION_DIAGNOSTIC.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        (args.out_dir/'README_RESULTS.txt').write_text(
            'EXPLORATORY ONLY. Source-level pixel-template registration estimates; NOT verified physical matching.\n'
            'Fits use hash-disjoint landmarks; held-out errors are displacement prediction errors at sampled sites, not image-level registration scores.\n'
            'No PA-NF training, frozen evaluation, clinical diagnoses, or patient/case linkage.\n',encoding='utf-8')
    success=('error' not in receipt and all(d.get('modeling_status')=='HELDOUT_GEOMETRIC_DIAGNOSTIC_COMPUTED' for d in receipt['scanners'].values()))
    print(f"[REG] {'DIAGNOSTIC_COMPLETE' if success else 'BLOCKED'}: {args.out_dir}",flush=True)
    return 0 if success else 2


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True,type=Path)
    parser.add_argument('--out-dir',required=True,type=Path)
    parser.add_argument('--slide-id',default='hmi825')
    parser.add_argument('--patch',default=256,type=int)
    parser.add_argument('--margin',default=96,type=int)
    parser.add_argument('--grid',default=9,type=int)
    parser.add_argument('--min-tissue',default=0.16,type=float)
    parser.add_argument('--min-std',default=12.0,type=float)
    parser.add_argument('--min-match',default=0.80,type=float)
    args=parser.parse_args(argv)
    import re
    if not re.fullmatch(r'[a-z]{3}[0-9]{3}',args.slide_id):parser.error('Invalid slide alias')
    if not 64 <= args.patch <= 512:parser.error('Patch must be 64..512')
    if not 16 <= args.margin <= 128:parser.error('Margin must be 16..128')
    if args.patch+2*args.margin > 768:parser.error('Search window must be <=768px on each side')
    if not 3 <= args.grid <= 19:parser.error('Grid must be 3..19')
    if not 0<=args.min_tissue<=1:parser.error('Invalid tissue fraction')
    if not 0<=args.min_std<=100:parser.error('Invalid std')
    if not 0.0<=args.min_match<=1.0:parser.error('Invalid NCC')
    return run(args)

if __name__ == '__main__':
    raise SystemExit(main())
