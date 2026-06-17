#!/usr/bin/env python3
"""Discover whether CPTAC pathology slide images are exposed through the GDC API.

This script is metadata-only. It enumerates GDC projects, selects any project or
program containing ``CPTAC``, and separately inventories all public GDC files
whose data type is ``Slide Image``. It never downloads slide bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_ROOT = "https://api.gdc.cancer.gov"
PROJECT_FIELDS = [
    "project_id",
    "name",
    "program.name",
    "primary_site",
    "disease_type",
    "summary.case_count",
    "summary.file_count",
]
FILE_FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "access",
    "data_category",
    "data_type",
    "data_format",
    "experimental_strategy",
    "cases.case_id",
    "cases.submitter_id",
    "cases.project.project_id",
    "cases.samples.sample_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
    "cases.samples.portions.slides.slide_id",
    "cases.samples.portions.slides.submitter_id",
]


def request_json(
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    url = f"{API_ROOT}/{endpoint.lstrip('/')}"
    if payload is None:
        response = requests.get(url, params=params, timeout=timeout)
    else:
        response = requests.post(url, json=payload, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected non-object response from {endpoint}")
    return data


def hits_from_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", {})
    hits = data.get("hits", []) if isinstance(data, dict) else []
    return hits if isinstance(hits, list) else []


def flatten_project(project: dict[str, Any]) -> dict[str, Any]:
    program = project.get("program")
    summary = project.get("summary")
    return {
        "project_id": project.get("project_id"),
        "name": project.get("name"),
        "program_name": program.get("name") if isinstance(program, dict) else None,
        "primary_site": project.get("primary_site"),
        "disease_type": project.get("disease_type"),
        "case_count": summary.get("case_count") if isinstance(summary, dict) else None,
        "file_count": summary.get("file_count") if isinstance(summary, dict) else None,
    }


def flatten_slide_file(file_record: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "file_id": file_record.get("file_id"),
        "file_name": file_record.get("file_name"),
        "file_size": file_record.get("file_size"),
        "md5sum": file_record.get("md5sum"),
        "access": file_record.get("access"),
        "data_category": file_record.get("data_category"),
        "data_type": file_record.get("data_type"),
        "data_format": file_record.get("data_format"),
        "experimental_strategy": file_record.get("experimental_strategy"),
    }
    cases = file_record.get("cases")
    if not isinstance(cases, list) or not cases:
        return [{**base, "case_id": None, "case_submitter_id": None, "project_id": None}]

    rows: list[dict[str, Any]] = []
    for case in cases:
        project = case.get("project") if isinstance(case, dict) else None
        samples = case.get("samples") if isinstance(case, dict) else None
        case_base = {
            **base,
            "case_id": case.get("case_id") if isinstance(case, dict) else None,
            "case_submitter_id": case.get("submitter_id") if isinstance(case, dict) else None,
            "project_id": project.get("project_id") if isinstance(project, dict) else None,
        }
        if not isinstance(samples, list) or not samples:
            rows.append(case_base)
            continue
        for sample in samples:
            rows.append(
                {
                    **case_base,
                    "sample_id": sample.get("sample_id") if isinstance(sample, dict) else None,
                    "sample_submitter_id": sample.get("submitter_id") if isinstance(sample, dict) else None,
                    "sample_type": sample.get("sample_type") if isinstance(sample, dict) else None,
                    "tissue_type": sample.get("tissue_type") if isinstance(sample, dict) else None,
                }
            )
    return rows


def contains_cptac(value: Any) -> bool:
    return "CPTAC" in str(value or "").upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--page-size", type=int, default=10000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    project_response = request_json(
        "projects",
        params={
            "format": "JSON",
            "size": args.page_size,
            "fields": ",".join(PROJECT_FIELDS),
        },
    )
    project_rows = [flatten_project(hit) for hit in hits_from_response(project_response)]
    projects = pd.DataFrame(project_rows)
    projects.to_csv(args.out_dir / "all_gdc_projects.csv", index=False)

    if projects.empty:
        cptac_projects = projects.copy()
    else:
        mask = (
            projects["project_id"].map(contains_cptac)
            | projects["name"].map(contains_cptac)
            | projects["program_name"].map(contains_cptac)
        )
        cptac_projects = projects.loc[mask].copy()
    cptac_projects.to_csv(args.out_dir / "cptac_gdc_projects.csv", index=False)

    slide_filter = {
        "op": "in",
        "content": {
            "field": "files.data_type",
            "value": ["Slide Image"],
        },
    }
    file_response = request_json(
        "files",
        params={
            "format": "JSON",
            "size": args.page_size,
            "expand": "cases,cases.project,cases.samples",
            "fields": ",".join(FILE_FIELDS),
        },
        payload={"filters": slide_filter},
    )
    file_hits = hits_from_response(file_response)
    slide_rows: list[dict[str, Any]] = []
    for hit in file_hits:
        slide_rows.extend(flatten_slide_file(hit))
    slide_files = pd.DataFrame(slide_rows)
    slide_files.to_csv(args.out_dir / "all_gdc_slide_images.csv", index=False)

    if slide_files.empty or "project_id" not in slide_files.columns:
        cptac_slide_files = slide_files.copy()
    else:
        cptac_slide_files = slide_files[
            slide_files["project_id"].map(contains_cptac)
        ].copy()
    cptac_slide_files.to_csv(args.out_dir / "cptac_gdc_slide_images.csv", index=False)

    project_ids = (
        sorted(cptac_projects["project_id"].dropna().astype(str).unique().tolist())
        if not cptac_projects.empty
        else []
    )
    slide_project_ids = (
        sorted(cptac_slide_files["project_id"].dropna().astype(str).unique().tolist())
        if not cptac_slide_files.empty and "project_id" in cptac_slide_files.columns
        else []
    )
    summary = {
        "status": (
            "cptac_slide_images_found_in_gdc"
            if not cptac_slide_files.empty
            else (
                "cptac_projects_found_but_no_slide_images"
                if project_ids
                else "no_cptac_projects_found_in_gdc"
            )
        ),
        "all_gdc_project_count": int(len(projects)),
        "cptac_project_count": int(len(cptac_projects)),
        "cptac_project_ids": project_ids,
        "all_gdc_slide_image_rows": int(len(slide_files)),
        "all_gdc_slide_image_files": (
            int(slide_files["file_id"].nunique())
            if not slide_files.empty and "file_id" in slide_files.columns
            else 0
        ),
        "cptac_slide_image_rows": int(len(cptac_slide_files)),
        "cptac_slide_image_files": (
            int(cptac_slide_files["file_id"].nunique())
            if not cptac_slide_files.empty and "file_id" in cptac_slide_files.columns
            else 0
        ),
        "cptac_slide_project_ids": slide_project_ids,
        "next_gate": (
            "Audit patient labels and scanner metadata before downloading any slide."
            if not cptac_slide_files.empty
            else "Do not use GDC as the CPTAC pathology route; inspect PDC/publication supporting-data resources."
        ),
    }
    (args.out_dir / "cptac_gdc_discovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("CPTAC GDC PATHOLOGY DISCOVERY PASSED")
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as exc:
        print(f"CPTAC GDC PATHOLOGY DISCOVERY FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
