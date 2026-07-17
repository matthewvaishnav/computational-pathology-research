#!/usr/bin/env python3
# ruff: noqa: UP045
"""Deterministic structural-identifiability audit for crossed acquisition designs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import combinations, product
from math import prod
from pathlib import Path
from typing import Any, Optional

AUDIT_ID = "crossed_preparation_identifiability_v2"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PACKAGE_DIR / "example_design_matrix.csv"
DEFAULT_REPORT = PACKAGE_DIR / "identifiability_report.md"

REQUIRED_COLUMNS = (
    "observation_id",
    "biological_unit",
    "block_id",
    "section_id",
    "preparation_condition",
    "scanner",
    "site_workflow",
    "preparation_batch",
    "scan_batch",
    "acquisition_order",
    "technical_replicate",
    "biological_replicate",
    "notes",
)

OPTIONAL_COLUMNS = (
    "repeat_acquisition_id",
    "operator_id",
    "preparation_order",
    "scanner_order",
    "temporal_window",
    "section_order",
    "section_distance",
    "fold_id",
    "registration_quality",
)

FACTORS = (
    "biological_unit",
    "preparation_condition",
    "scanner",
    "site_workflow",
)

FACTOR_LABELS = {
    "biological_unit": "biological unit",
    "preparation_condition": "preparation",
    "scanner": "scanner",
    "site_workflow": "site/workflow",
}

EFFECT_TO_FACTOR = {
    "preparation": "preparation_condition",
    "scanner": "scanner",
    "site_workflow": "site_workflow",
}

DEFAULT_REQUESTED_EFFECTS = tuple(EFFECT_TO_FACTOR)

PAIR_SPECS = (
    ("biological_unit", "preparation_condition"),
    ("biological_unit", "scanner"),
    ("biological_unit", "site_workflow"),
    ("preparation_condition", "scanner"),
    ("preparation_condition", "site_workflow"),
    ("scanner", "site_workflow"),
)

HIGHER_ORDER_SPECS = {
    "biology_preparation_scanner": (
        "biological_unit",
        "preparation_condition",
        "scanner",
    ),
    "preparation_scanner_workflow": (
        "preparation_condition",
        "scanner",
        "site_workflow",
    ),
    "biology_preparation_scanner_workflow": FACTORS,
}

INTERACTION_SPECS = {
    "preparation_scanner": ("preparation_condition", "scanner"),
    "scanner_site_workflow": ("scanner", "site_workflow"),
    "preparation_site_workflow": (
        "preparation_condition",
        "site_workflow",
    ),
}

IDENTIFIER_FIELDS = (
    "observation_id",
    "biological_unit",
    "block_id",
    "section_id",
    "preparation_condition",
    "scanner",
    "site_workflow",
    "preparation_batch",
    "scan_batch",
    "technical_replicate",
    "biological_replicate",
)

OPTIONAL_IDENTIFIER_FIELDS = ("repeat_acquisition_id", "operator_id", "fold_id")
OPTIONAL_INTEGER_FIELDS = ("preparation_order", "scanner_order", "section_order")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

VERDICT_DIRECT = "directly estimable"
VERDICT_PARTIAL = "estimable with partial crossing"
VERDICT_ASSUMPTION = "estimable only under modeling assumptions"
VERDICT_NOT = "not estimable"
MISSING_COMBINATION_REPORT_LIMIT = 1000

LIMITATIONS = (
    "Structural identifiability is not statistical power or a power guarantee.",
    "Balanced crossing does not prove causal attribution.",
    "Matched serial sections are not identical cells, regions, or pixels.",
    "A same-block serial-section preparation bridge applies only to interventions physically assignable at or after sectioning.",
    "Site/workflow effects may aggregate multiple upstream factors.",
    "Biological heterogeneity can remain within blocks and sections.",
    "Acquisition, preparation, batch, order, and operator metadata must be recorded prospectively.",
    "No model quality can recover an effect that the sampling design does not identify.",
    "A scanner level reused across sites must denote a defensible repeatable acquisition condition; unique devices fixed at sites remain nested.",
    "The checked example crosses a post-preparation acquisition-workflow label; it does not identify an upstream site preparation effect.",
    "Randomization metadata availability does not prove randomized execution.",
    "Row-level residual degrees of freedom are not independent biological degrees of freedom.",
    "Declared repeat-acquisition identifiers document technical repeats but cannot prove source-file identity without immutable acquisition provenance.",
)


class InputValidationError(Exception):
    """Fail-closed error for malformed or internally inconsistent input."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code
        self.detail = detail


class RankCalculationError(Exception):
    """Fail-closed error for exact-rank calculation failures."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code if not detail else f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def alias_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def pair_key(first: str, second: str) -> str:
    return f"{first}__x__{second}"


def canonical_positive_integer(value: str, field: str) -> int:
    if not value or not value.isascii() or not value.isdigit():
        raise InputValidationError(f"invalid_{field}", value)
    parsed = int(value)
    if parsed < 1 or str(parsed) != value:
        raise InputValidationError(f"invalid_{field}", value)
    return parsed


def parse_requested_effects(value: str) -> tuple[str, ...]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise InputValidationError("empty_requested_effects")
    unknown = sorted(set(requested) - set(EFFECT_TO_FACTOR))
    if unknown:
        raise InputValidationError("unknown_requested_effect", ",".join(unknown))
    return tuple(effect for effect in EFFECT_TO_FACTOR if effect in set(requested))


def validate_aliases(rows: Sequence[Mapping[str, str]], fields: Iterable[str]) -> None:
    for field in fields:
        by_alias: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.get(field, "")
            if value:
                by_alias[alias_key(value)].add(value)
        for key in sorted(by_alias):
            values = sorted(by_alias[key])
            if len(values) > 1:
                raise InputValidationError(f"string_aliasing:{field}", "|".join(values))


def validate_rows(
    rows: Sequence[Mapping[str, str]],
    headers: Sequence[str],
    requested_effects: Sequence[str],
) -> None:
    if not rows:
        raise InputValidationError("empty_design_matrix")

    active_optional = [field for field in OPTIONAL_COLUMNS if field in headers]
    nonempty_fields = [field for field in REQUIRED_COLUMNS if field != "notes"]
    nonempty_fields.extend(active_optional)
    for row_index, row in enumerate(rows, start=2):
        for field in nonempty_fields:
            value = row.get(field, "")
            if not value:
                raise InputValidationError(f"empty_required_value:{field}", f"row={row_index}")
            if value != value.strip() or any(
                unicodedata.category(character) == "Cc" for character in value
            ):
                raise InputValidationError(f"noncanonical_value:{field}", f"row={row_index}")
        notes = row.get("notes", "")
        if any(unicodedata.category(character) == "Cc" for character in notes):
            raise InputValidationError("noncanonical_value:notes", f"row={row_index}")

    for field in IDENTIFIER_FIELDS:
        for row_index, row in enumerate(rows, start=2):
            if not IDENTIFIER_PATTERN.fullmatch(row[field]):
                raise InputValidationError(f"invalid_identifier:{field}", f"row={row_index}")
    for field in OPTIONAL_IDENTIFIER_FIELDS:
        if field not in headers:
            continue
        for row_index, row in enumerate(rows, start=2):
            if not IDENTIFIER_PATTERN.fullmatch(row[field]):
                raise InputValidationError(f"invalid_identifier:{field}", f"row={row_index}")

    observation_ids = [row["observation_id"] for row in rows]
    duplicate_ids = sorted(value for value, count in Counter(observation_ids).items() if count > 1)
    if duplicate_ids:
        raise InputValidationError("duplicate_observation_id", "|".join(duplicate_ids))

    validate_aliases(
        rows,
        tuple(IDENTIFIER_FIELDS)
        + tuple(field for field in OPTIONAL_IDENTIFIER_FIELDS if field in headers),
    )

    physical_fields = (
        "section_id",
        "preparation_condition",
        "scanner",
        "site_workflow",
    )
    repeat_field_present = "repeat_acquisition_id" in headers
    physical_seen: dict[tuple[str, ...], dict[str, str]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in physical_fields)
        repeat_id = row["repeat_acquisition_id"] if repeat_field_present else ""
        repeats = physical_seen[key]
        if not repeat_field_present and repeats:
            prior = next(iter(repeats.values()))
            raise InputValidationError(
                "duplicate_physical_observation",
                f"{prior}|{row['observation_id']}",
            )
        if repeat_id in repeats:
            raise InputValidationError(
                "duplicate_physical_observation",
                f"{repeats[repeat_id]}|{row['observation_id']}|repeat={repeat_id or '<absent>'}",
            )
        repeats[repeat_id] = row["observation_id"]

    acquisition_keys: set[tuple[str, int]] = set()
    for row in rows:
        order = canonical_positive_integer(row["acquisition_order"], "acquisition_order")
        key = (row["scan_batch"], order)
        if key in acquisition_keys:
            raise InputValidationError(
                "duplicate_acquisition_order_within_scan_batch",
                "|".join(map(str, key)),
            )
        acquisition_keys.add(key)
    for field in OPTIONAL_INTEGER_FIELDS:
        if field in headers:
            for row in rows:
                canonical_positive_integer(row[field], field)

    block_to_biology: dict[str, str] = {}
    section_mapping: dict[str, tuple[str, str, str, str, str]] = {}
    biology_to_replicate: dict[str, str] = {}
    replicate_to_biology: dict[str, str] = {}
    technical_to_section: dict[str, str] = {}
    for row in rows:
        block = row["block_id"]
        biology = row["biological_unit"]
        if block in block_to_biology and block_to_biology[block] != biology:
            raise InputValidationError("impossible_block_biology_relationship", block)
        block_to_biology[block] = biology

        section = row["section_id"]
        mapping = (
            biology,
            block,
            row["preparation_condition"],
            row["preparation_batch"],
            row["technical_replicate"],
        )
        if section in section_mapping and section_mapping[section] != mapping:
            raise InputValidationError("impossible_section_block_relationship", section)
        section_mapping[section] = mapping

        technical_replicate = row["technical_replicate"]
        if (
            technical_replicate in technical_to_section
            and technical_to_section[technical_replicate] != section
        ):
            raise InputValidationError(
                "technical_replicate_mapping_conflict",
                technical_replicate,
            )
        technical_to_section[technical_replicate] = section

        replicate = row["biological_replicate"]
        if biology in biology_to_replicate and biology_to_replicate[biology] != replicate:
            raise InputValidationError("biological_replicate_mapping_conflict", biology)
        if replicate in replicate_to_biology and replicate_to_biology[replicate] != biology:
            raise InputValidationError("biological_replicate_mapping_conflict", replicate)
        biology_to_replicate[biology] = replicate
        replicate_to_biology[replicate] = biology

    requested_factors = {EFFECT_TO_FACTOR[effect] for effect in requested_effects}
    requested_factors.add("biological_unit")
    for factor in FACTORS:
        levels = {row[factor] for row in rows}
        if factor in requested_factors and len(levels) < 2:
            raise InputValidationError(f"factor_too_few_levels:{factor}", str(len(levels)))


def load_design(
    path: Path, requested_effects: Sequence[str]
) -> tuple[list[dict[str, str]], str, tuple[str, ...]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InputValidationError("input_unreadable", str(exc)) from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raise InputValidationError("utf8_bom_forbidden")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InputValidationError("input_not_utf8", str(exc)) from exc

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise InputValidationError("missing_csv_header") from exc
    duplicates = sorted(value for value, count in Counter(header).items() if count > 1)
    if duplicates:
        raise InputValidationError("duplicate_header", "|".join(duplicates))
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise InputValidationError("missing_required_columns", "|".join(missing))
    unknown = sorted(set(header) - set(REQUIRED_COLUMNS) - set(OPTIONAL_COLUMNS))
    if unknown:
        raise InputValidationError("unknown_columns", "|".join(unknown))

    rows: list[dict[str, str]] = []
    for row_number, values in enumerate(reader, start=2):
        if len(values) != len(header):
            raise InputValidationError("malformed_csv_row", f"row={row_number}")
        rows.append(dict(zip(header, values)))
    rows.sort(key=lambda row: row["observation_id"])
    validate_rows(rows, header, requested_effects)
    return rows, sha256_bytes(raw), tuple(header)


def exact_fraction(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        raise InputValidationError("invalid_fraction_denominator")
    fraction = Fraction(numerator, denominator)
    with localcontext() as context:
        context.prec = 28
        decimal = format(Decimal(numerator) / Decimal(denominator), ".6f")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": f"{fraction.numerator}/{fraction.denominator}",
        "decimal": decimal,
    }


def sorted_levels(rows: Sequence[Mapping[str, str]], factor: str) -> list[str]:
    return sorted({row[factor] for row in rows})


def graph_components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for first, second in edges:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    components: list[list[str]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        queue = deque([start])
        unseen.remove(start)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return sorted(components, key=lambda component: component[0])


def factor_inventory(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for factor in FACTORS:
        levels = sorted_levels(rows, factor)
        level_rows: dict[str, list[Mapping[str, str]]] = {
            level: [row for row in rows if row[factor] == level] for level in levels
        }
        cross_coverage: dict[str, dict[str, int]] = {}
        for level in levels:
            cross_coverage[level] = {
                other: len({row[other] for row in level_rows[level]})
                for other in FACTORS
                if other != factor
            }
        inventory[factor] = {
            "level_count": len(levels),
            "levels": levels,
            "observations_per_level": {level: len(level_rows[level]) for level in levels},
            "biological_units_per_level": {
                level: len({row["biological_unit"] for row in level_rows[level]})
                for level in levels
            },
            "cross_factor_level_counts": cross_coverage,
        }
    return inventory


def pairwise_crossing(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for first, second in PAIR_SPECS:
        first_levels = sorted_levels(rows, first)
        second_levels = sorted_levels(rows, second)
        counts: Counter[tuple[str, str]] = Counter((row[first], row[second]) for row in rows)
        biological_support: dict[tuple[str, str], set[str]] = defaultdict(set)
        section_support: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            cell = (row[first], row[second])
            biological_support[cell].add(row["biological_unit"])
            section_support[cell].add(row["section_id"])

        nodes = [f"{first}={level}" for level in first_levels]
        nodes.extend(f"{second}={level}" for level in second_levels)
        edges = [
            (f"{first}={first_level}", f"{second}={second_level}")
            for first_level, second_level in sorted(counts)
        ]
        degrees_first = {
            level: len(
                {second_level for first_level, second_level in counts if first_level == level}
            )
            for level in first_levels
        }
        degrees_second = {
            level: len(
                {first_level for first_level, second_level in counts if second_level == level}
            )
            for level in second_levels
        }
        cells = []
        for cell in sorted(counts):
            cells.append(
                {
                    "first_level": cell[0],
                    "second_level": cell[1],
                    "observations": counts[cell],
                    "biological_units": len(biological_support[cell]),
                    "sections": len(section_support[cell]),
                }
            )
        observed = len(counts)
        possible = len(first_levels) * len(second_levels)
        result[pair_key(first, second)] = {
            "first_factor": first,
            "second_factor": second,
            "observed_combinations": observed,
            "possible_combinations": possible,
            "coverage": exact_fraction(observed, possible),
            "minimum_observations_per_observed_combination": min(counts.values()),
            "minimum_biological_units_per_observed_combination": min(
                len(values) for values in biological_support.values()
            ),
            "minimum_sections_per_observed_combination": min(
                len(values) for values in section_support.values()
            ),
            "degrees": {
                first: degrees_first,
                second: degrees_second,
            },
            "components": graph_components(nodes, edges),
            "component_count": len(graph_components(nodes, edges)),
            "cells": cells,
        }
    return result


def crossing_coverage(
    rows: Sequence[Mapping[str, str]],
    factors: Sequence[str],
) -> dict[str, Any]:
    ordered_factors = tuple(factors)
    if len(ordered_factors) < 2 or len(set(ordered_factors)) != len(ordered_factors):
        raise InputValidationError("invalid_crossing_factor_set", "|".join(ordered_factors))
    levels = {factor: sorted_levels(rows, factor) for factor in ordered_factors}
    possible = prod(len(levels[factor]) for factor in ordered_factors)
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(row[factor] for factor in ordered_factors) for row in rows
    )
    biological_support: dict[tuple[str, ...], set[str]] = defaultdict(set)
    section_support: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        cell = tuple(row[factor] for factor in ordered_factors)
        biological_support[cell].add(row["biological_unit"])
        section_support[cell].add(row["section_id"])
    missing_count = possible - len(counts)
    missing: list[tuple[str, ...]] = []
    if missing_count:
        for cell in product(*(levels[factor] for factor in ordered_factors)):
            if cell not in counts:
                missing.append(cell)
                if len(missing) == MISSING_COMBINATION_REPORT_LIMIT:
                    break
    cells = [
        {
            "levels": dict(zip(ordered_factors, cell)),
            "observations": counts[cell],
            "biological_units": len(biological_support[cell]),
            "sections": len(section_support[cell]),
        }
        for cell in sorted(counts)
    ]
    return {
        "factors": list(ordered_factors),
        "levels": levels,
        "observed_combinations": len(counts),
        "possible_combinations": possible,
        "coverage": exact_fraction(len(counts), possible),
        "minimum_observations_per_observed_combination": min(counts.values()),
        "minimum_biological_units_per_observed_combination": min(
            len(values) for values in biological_support.values()
        ),
        "minimum_sections_per_observed_combination": min(
            len(values) for values in section_support.values()
        ),
        "missing_combinations": [
            dict(zip(ordered_factors, cell))
            for cell in missing
        ],
        "missing_combination_count": missing_count,
        "missing_combinations_complete": missing_count == len(missing),
        "cells": cells,
    }


def higher_order_crossing(
    rows: Sequence[Mapping[str, str]],
    requested_effects: Sequence[str],
    requested_interactions: Sequence[str],
) -> dict[str, Any]:
    result = {
        name: crossing_coverage(rows, factors)
        for name, factors in HIGHER_ORDER_SPECS.items()
    }
    requested_factor_set = {"biological_unit"}
    requested_factor_set.update(EFFECT_TO_FACTOR[effect] for effect in requested_effects)
    for interaction_name in requested_interactions:
        requested_factor_set.update(INTERACTION_SPECS[interaction_name])
    requested_factors = tuple(
        factor for factor in FACTORS if factor in requested_factor_set
    )
    result["requested_factor_product"] = crossing_coverage(rows, requested_factors)
    return result


def directional_nesting(
    source: str,
    target: str,
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    degrees = dict(pair["degrees"][source])
    singleton_levels = sorted(level for level, degree in degrees.items() if degree == 1)
    total = len(degrees)
    if total and len(singleton_levels) == total:
        status = "exact nesting"
    elif singleton_levels:
        status = "partial nesting"
    else:
        status = "not nested"
    return {
        "source_factor": source,
        "target_factor": target,
        "status": status,
        "singleton_level_count": len(singleton_levels),
        "level_count": total,
        "singleton_fraction": exact_fraction(len(singleton_levels), total),
        "singleton_levels": singleton_levels,
        "degrees": {level: degrees[level] for level in sorted(degrees)},
    }


def nesting_analysis(
    pairwise: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    relationships: list[dict[str, Any]] = []
    one_to_one: list[dict[str, str]] = []
    for first, second in PAIR_SPECS:
        pair = pairwise[pair_key(first, second)]
        forward = directional_nesting(first, second, pair)
        reverse = directional_nesting(second, first, pair)
        relationships.extend((forward, reverse))
        if forward["status"] == "exact nesting" and reverse["status"] == "exact nesting":
            one_to_one.append({"first_factor": first, "second_factor": second})
    relationships.sort(key=lambda item: (item["source_factor"], item["target_factor"]))
    return {
        "relationships": relationships,
        "one_to_one_aliases": one_to_one,
    }


def global_factor_graph(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    nodes = [f"{factor}={level}" for factor in FACTORS for level in sorted_levels(rows, factor)]
    edges: set[tuple[str, str]] = set()
    chain = (
        ("biological_unit", "preparation_condition"),
        ("preparation_condition", "scanner"),
        ("scanner", "site_workflow"),
    )
    for row in rows:
        for first, second in chain:
            edges.add((f"{first}={row[first]}", f"{second}={row[second]}"))
    components = graph_components(nodes, sorted(edges))
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "component_count": len(components),
        "connected": len(components) == 1,
        "components": components,
    }


def matrix_rank(matrix: Sequence[Sequence[int]]) -> int:
    if not matrix:
        raise RankCalculationError("rank_matrix_has_no_rows")
    width = len(matrix[0])
    if width == 0:
        return 0
    if any(len(row) != width for row in matrix):
        raise RankCalculationError("rank_matrix_is_ragged")
    try:
        work = [[Fraction(value) for value in row] for row in matrix]
        rank = 0
        for column in range(width):
            pivot = next(
                (row_index for row_index in range(rank, len(work)) if work[row_index][column] != 0),
                None,
            )
            if pivot is None:
                continue
            work[rank], work[pivot] = work[pivot], work[rank]
            pivot_value = work[rank][column]
            work[rank] = [value / pivot_value for value in work[rank]]
            for row_index in range(len(work)):
                if row_index == rank or work[row_index][column] == 0:
                    continue
                multiplier = work[row_index][column]
                work[row_index] = [
                    current - multiplier * pivot_entry
                    for current, pivot_entry in zip(work[row_index], work[rank])
                ]
            rank += 1
            if rank == len(work):
                break
        return rank
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise RankCalculationError("rank_calculation_failed", str(exc)) from exc


def design_metadata(
    rows: Sequence[Mapping[str, str]],
    interactions: Sequence[str] = (),
) -> dict[str, Any]:
    levels = {factor: sorted_levels(rows, factor) for factor in FACTORS}
    references = {factor: values[0] for factor, values in levels.items()}
    columns = ["intercept"]
    for factor in FACTORS:
        columns.extend(f"{factor}={level}" for level in levels[factor][1:])
    for interaction_name in interactions:
        first, second = INTERACTION_SPECS[interaction_name]
        for first_level in levels[first][1:]:
            for second_level in levels[second][1:]:
                columns.append(f"{interaction_name}:{first}={first_level}*{second}={second_level}")
    return {
        "levels": levels,
        "references": references,
        "columns": columns,
        "interactions": list(interactions),
    }


def encode_assignment(assignment: Mapping[str, str], metadata: Mapping[str, Any]) -> list[int]:
    values = [1]
    levels = metadata["levels"]
    for factor in FACTORS:
        values.extend(int(assignment[factor] == level) for level in levels[factor][1:])
    for interaction_name in metadata["interactions"]:
        first, second = INTERACTION_SPECS[interaction_name]
        for first_level in levels[first][1:]:
            for second_level in levels[second][1:]:
                values.append(
                    int(assignment[first] == first_level and assignment[second] == second_level)
                )
    return values


def design_matrix(
    rows: Sequence[Mapping[str, str]],
    interactions: Sequence[str] = (),
) -> tuple[list[list[int]], dict[str, Any]]:
    metadata = design_metadata(rows, interactions)
    matrix = [encode_assignment(row, metadata) for row in rows]
    if any(len(row) != len(metadata["columns"]) for row in matrix):
        raise RankCalculationError("design_matrix_column_mismatch")
    return matrix, metadata


def incremental_aliases(matrix: Sequence[Sequence[int]], columns: Sequence[str]) -> list[str]:
    aliases: list[str] = []
    previous_rank = 0
    for column_index, column_name in enumerate(columns):
        candidate = [row[: column_index + 1] for row in matrix]
        candidate_rank = matrix_rank(candidate)
        if candidate_rank == previous_rank:
            aliases.append(column_name)
        else:
            previous_rank = candidate_rank
    return aliases


def rank_summary(matrix: Sequence[Sequence[int]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    rank = matrix_rank(matrix)
    unique_matrix = sorted({tuple(row) for row in matrix})
    columns = list(metadata["columns"])
    term_df: dict[str, dict[str, int]] = {}
    full_rank = rank
    for factor in FACTORS:
        indices = [index for index, name in enumerate(columns) if name.startswith(f"{factor}=")]
        nominal = len(indices)
        if not indices:
            estimable = 0
        else:
            reduced = [
                [value for index, value in enumerate(row) if index not in indices] for row in matrix
            ]
            estimable = full_rank - matrix_rank(reduced)
        term_df[factor] = {
            "nominal_degrees_of_freedom": nominal,
            "uniquely_estimable_degrees_of_freedom": estimable,
        }
    return {
        "rank": rank,
        "column_count": len(columns),
        "row_count": len(matrix),
        "residual_degrees_of_freedom": len(matrix) - rank,
        "row_level_residual_degrees_of_freedom": len(matrix) - rank,
        "unique_design_row_count": len(unique_matrix),
        "unique_design_residual_degrees_of_freedom": len(unique_matrix) - rank,
        "residual_df_interpretation": (
            "Row-level n minus rank; not independent biological degrees of freedom. "
            "Repeated acquisitions can increase this value without increasing biological support."
        ),
        "rank_deficiency": len(columns) - rank,
        "aliased_columns": incremental_aliases(matrix, columns),
        "columns": columns,
        "term_degrees_of_freedom": term_df,
    }


def vector_subtract(*vectors: Sequence[int]) -> list[int]:
    if not vectors:
        raise RankCalculationError("contrast_has_no_vectors")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise RankCalculationError("contrast_vector_width_mismatch")
    result = list(vectors[0])
    for vector in vectors[1:]:
        result = [left - right for left, right in zip(result, vector)]
    return result


def contrast_is_estimable(matrix: Sequence[Sequence[int]], contrast: Sequence[int]) -> bool:
    if len(contrast) != len(matrix[0]):
        raise RankCalculationError("contrast_design_width_mismatch")
    return matrix_rank(matrix) == matrix_rank(list(matrix) + [list(contrast)])


def grouped_rows(
    rows: Sequence[Mapping[str, str]], fields: Sequence[str]
) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return dict(groups)


def groups_spanning_levels(
    rows: Sequence[Mapping[str, str]],
    group_fields: Sequence[str],
    factor: str,
    first_level: str,
    second_level: str,
) -> list[tuple[tuple[str, ...], list[Mapping[str, str]]]]:
    required = {first_level, second_level}
    result = []
    for key, group in sorted(grouped_rows(rows, group_fields).items()):
        if required.issubset({row[factor] for row in group}):
            result.append((key, group))
    return result


def effect_support(
    rows: Sequence[Mapping[str, str]],
    factor: str,
    first_level: str,
    second_level: str,
) -> dict[str, Any]:
    if factor == "preparation_condition":
        base_fields = ("biological_unit",)
        direct_fields = (
            "biological_unit",
            "block_id",
            "scanner",
            "site_workflow",
            "preparation_batch",
            "scan_batch",
        )
        support_mode = (
            "matched serial sections within the same biological unit and block "
            "under matched scanner/workflow and preparation/scan batches"
        )
    elif factor == "scanner":
        base_fields = ("biological_unit", "preparation_condition")
        direct_fields = (
            "biological_unit",
            "preparation_condition",
            "section_id",
            "site_workflow",
            "scan_batch",
        )
        support_mode = "same prepared physical section, workflow, and scan batch across scanners"
    elif factor == "site_workflow":
        base_fields = ("biological_unit", "preparation_condition", "scanner")
        direct_fields = (
            "biological_unit",
            "preparation_condition",
            "scanner",
            "section_id",
            "scan_batch",
        )
        support_mode = (
            "same prepared physical section, scanner, and scan batch across a "
            "declared repeatable workflow exposure"
        )
    else:
        raise RankCalculationError("unsupported_effect_factor", factor)

    base_groups = groups_spanning_levels(
        rows,
        base_fields,
        factor,
        first_level,
        second_level,
    )
    direct_groups = groups_spanning_levels(
        rows,
        direct_fields,
        factor,
        first_level,
        second_level,
    )

    base_biology = sorted({row["biological_unit"] for _, group in base_groups for row in group})
    direct_biology = sorted({row["biological_unit"] for _, group in direct_groups for row in group})
    base_support_rows = [
        row
        for _, group in base_groups
        for row in group
        if row[factor] in {first_level, second_level}
    ]
    direct_support_rows = [
        row
        for _, group in direct_groups
        for row in group
        if row[factor] in {first_level, second_level}
    ]
    support_rows = direct_support_rows or base_support_rows
    fold_metadata_present = bool(rows) and "fold_id" in rows[0]

    carrier_fields = direct_fields if direct_groups else base_fields

    def carrier_values(field: str) -> list[str]:
        fields = tuple(dict.fromkeys((*carrier_fields, field)))
        carriers = groups_spanning_levels(
            rows,
            fields,
            factor,
            first_level,
            second_level,
        )
        return sorted({row[field] for _, group in carriers for row in group})

    matched_serial_section_pairs: set[tuple[str, str, str, str, str, str]] = set()
    ambiguous_matched_serial_section_strata = 0
    if factor == "preparation_condition":
        pair_fields = (
            "biological_unit",
            "block_id",
            "preparation_batch",
            "scan_batch",
        )
        for _, group in groups_spanning_levels(
            rows,
            pair_fields,
            factor,
            first_level,
            second_level,
        ):
            first_sections = sorted(
                {row["section_id"] for row in group if row[factor] == first_level}
            )
            second_sections = sorted(
                {row["section_id"] for row in group if row[factor] == second_level}
            )
            if len(first_sections) == 1 and len(second_sections) == 1:
                matched_serial_section_pairs.add(
                    (
                        group[0]["biological_unit"],
                        group[0]["block_id"],
                        group[0]["preparation_batch"],
                        group[0]["scan_batch"],
                        first_sections[0],
                        second_sections[0],
                    )
                )
            else:
                ambiguous_matched_serial_section_strata += 1

    supporting_blocks = carrier_values("block_id")
    supporting_preparation_batches = carrier_values("preparation_batch")
    supporting_scan_batches = carrier_values("scan_batch")
    supporting_sections = sorted({row["section_id"] for row in support_rows})

    return {
        "support_mode": support_mode,
        "biological_supporters": base_biology,
        "biological_supporter_count": len(base_biology),
        "direct_biological_supporters": direct_biology,
        "direct_biological_supporter_count": len(direct_biology),
        "controlled_stratum_count": len(direct_groups),
        "bridge_count": len(direct_groups),
        "supporting_blocks": supporting_blocks,
        "supporting_block_count": len(supporting_blocks),
        "supporting_preparation_batches": supporting_preparation_batches,
        "supporting_preparation_batch_count": len(supporting_preparation_batches),
        "supporting_scan_batches": supporting_scan_batches,
        "supporting_scan_batch_count": len(supporting_scan_batches),
        "supporting_sections": supporting_sections,
        "supporting_section_count": len(supporting_sections),
        "matched_serial_section_pairs": [list(item) for item in sorted(matched_serial_section_pairs)],
        "matched_serial_section_pair_count": len(matched_serial_section_pairs),
        "ambiguous_matched_serial_section_strata_count": (
            ambiguous_matched_serial_section_strata
        ),
        "fold_metadata_present": fold_metadata_present,
        "supporting_folds": carrier_values("fold_id") if fold_metadata_present else [],
    }


def levels_nested_within_factor(
    rows: Sequence[Mapping[str, str]],
    nested: str,
    factor: str,
) -> bool:
    nested_to_factor: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        nested_to_factor[row[nested]].add(row[factor])
    return (
        len(nested_to_factor) > 1
        and len({row[factor] for row in rows}) > 1
        and all(len(values) == 1 for values in nested_to_factor.values())
    )


def apply_batch_assumptions(
    rows: Sequence[Mapping[str, str]],
    effects: dict[str, dict[str, Any]],
) -> None:
    specifications = (
        (
            "preparation",
            "preparation_condition",
            "preparation_batch",
            "preparation_batches_nested_within_preparation",
        ),
        (
            "preparation",
            "preparation_condition",
            "scan_batch",
            "scan_batches_nested_within_preparation",
        ),
        (
            "scanner",
            "scanner",
            "scan_batch",
            "scan_batches_nested_within_scanner",
        ),
        (
            "site_workflow",
            "site_workflow",
            "scan_batch",
            "scan_batches_nested_within_site_workflow",
        ),
    )
    for effect, factor, batch, dependency in specifications:
        if not levels_nested_within_factor(rows, batch, factor):
            continue
        for contrast in effects[effect]["contrasts"]:
            if contrast["verdict"] in {VERDICT_DIRECT, VERDICT_PARTIAL}:
                contrast["verdict"] = VERDICT_ASSUMPTION
        effects[effect]["verdict"] = worst_verdict(
            [item["verdict"] for item in effects[effect]["contrasts"]]
        )
        effects[effect]["assumption_dependencies"].append(dependency)
        effects[effect]["interpretation"] = (
            "Structural separation is assumption-dependent because the target "
            f"factor has nested recorded batches ({dependency}). No "
            "outcome or effect was measured."
        )


def relevant_pair_keys(factor: str) -> tuple[str, ...]:
    if factor == "preparation_condition":
        return (
            pair_key("biological_unit", "preparation_condition"),
            pair_key("preparation_condition", "scanner"),
            pair_key("preparation_condition", "site_workflow"),
        )
    if factor == "scanner":
        return (
            pair_key("biological_unit", "scanner"),
            pair_key("preparation_condition", "scanner"),
            pair_key("scanner", "site_workflow"),
        )
    if factor == "site_workflow":
        return (
            pair_key("biological_unit", "site_workflow"),
            pair_key("preparation_condition", "site_workflow"),
            pair_key("scanner", "site_workflow"),
        )
    raise RankCalculationError("unsupported_effect_factor", factor)


def worst_verdict(verdicts: Sequence[str]) -> str:
    priority = {
        VERDICT_DIRECT: 0,
        VERDICT_PARTIAL: 1,
        VERDICT_ASSUMPTION: 2,
        VERDICT_NOT: 3,
    }
    if not verdicts:
        return VERDICT_NOT
    return max(verdicts, key=lambda verdict: priority[verdict])


def main_effect_contrasts(
    rows: Sequence[Mapping[str, str]],
    matrix: Sequence[Sequence[int]],
    metadata: Mapping[str, Any],
    pairwise: Mapping[str, Mapping[str, Any]],
    requested_effects: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    requested_set = set(requested_effects)
    reference_assignment = {factor: metadata["references"][factor] for factor in FACTORS}
    for effect, factor in EFFECT_TO_FACTOR.items():
        levels = list(metadata["levels"][factor])
        contrasts: list[dict[str, Any]] = []
        full_relevant_crossing = all(
            pairwise[key]["coverage"]["numerator"] == pairwise[key]["coverage"]["denominator"]
            for key in relevant_pair_keys(factor)
        )
        for first_level, second_level in combinations(levels, 2):
            first_assignment = dict(reference_assignment)
            second_assignment = dict(reference_assignment)
            first_assignment[factor] = first_level
            second_assignment[factor] = second_level
            first_vector = encode_assignment(first_assignment, metadata)
            second_vector = encode_assignment(second_assignment, metadata)
            contrast = vector_subtract(second_vector, first_vector)
            algebraic = contrast_is_estimable(matrix, contrast)
            support = effect_support(rows, factor, first_level, second_level)
            if algebraic and support["direct_biological_supporter_count"] >= 2:
                verdict = VERDICT_DIRECT if full_relevant_crossing else VERDICT_PARTIAL
            elif algebraic and support["biological_supporter_count"] >= 2:
                verdict = VERDICT_ASSUMPTION
            else:
                verdict = VERDICT_NOT
            contrasts.append(
                {
                    "first_level": first_level,
                    "second_level": second_level,
                    "algebraically_estimable": algebraic,
                    "verdict": verdict,
                    "structural_verdict": verdict,
                    "support": support,
                }
            )
        verdict = worst_verdict([item["verdict"] for item in contrasts])
        if factor == "site_workflow":
            interpretation = (
                "Structural support applies only to the declared repeatable "
                "site/workflow exposure; no outcome was measured, and a "
                "post-preparation bridge does not identify an upstream site "
                "preparation effect."
            )
        else:
            interpretation = "Structural support only; no outcome or effect was measured."
        if verdict == VERDICT_ASSUMPTION:
            interpretation = (
                "Algebraic support exists, but fewer than two biological units "
                "supply the required section/workflow/batch-controlled bridge; "
                "the verdict therefore depends on additivity or exchangeability "
                "assumptions. No outcome or effect was measured."
            )
        result[effect] = {
            "factor": factor,
            "requested": effect in requested_set,
            "verdict": verdict,
            "structural_verdict": verdict,
            "full_relevant_pairwise_crossing": full_relevant_crossing,
            "contrasts": contrasts,
            "assumption_dependencies": [],
            "interpretation": interpretation,
        }
    return result


def difference_in_differences_vector(
    metadata: Mapping[str, Any],
    first_factor: str,
    first_low: str,
    first_high: str,
    second_factor: str,
    second_low: str,
    second_high: str,
) -> list[int]:
    reference = {factor: metadata["references"][factor] for factor in FACTORS}

    def encoded(first_value: str, second_value: str) -> list[int]:
        assignment = dict(reference)
        assignment[first_factor] = first_value
        assignment[second_factor] = second_value
        return encode_assignment(assignment, metadata)

    low_low = encoded(first_low, second_low)
    low_high = encoded(first_low, second_high)
    high_low = encoded(first_high, second_low)
    high_high = encoded(first_high, second_high)
    return [hh - hl - lh + ll for hh, hl, lh, ll in zip(high_high, high_low, low_high, low_low)]


def interaction_direct_support(
    rows: Sequence[Mapping[str, str]],
    first: str,
    first_low: str,
    first_high: str,
    second: str,
    second_low: str,
    second_high: str,
) -> dict[str, Any]:
    pair = frozenset((first, second))
    if pair == frozenset(("preparation_condition", "scanner")):
        held_fields = (
            "biological_unit",
            "block_id",
            "site_workflow",
            "preparation_batch",
            "scan_batch",
        )
    elif pair == frozenset(("scanner", "site_workflow")):
        held_fields = (
            "biological_unit",
            "preparation_condition",
            "section_id",
            "scan_batch",
        )
    elif pair == frozenset(("preparation_condition", "site_workflow")):
        held_fields = (
            "biological_unit",
            "block_id",
            "scanner",
            "preparation_batch",
            "scan_batch",
        )
    else:
        raise RankCalculationError("unsupported_interaction_factors", f"{first}|{second}")

    required_cells = {
        (first_low, second_low),
        (first_low, second_high),
        (first_high, second_low),
        (first_high, second_high),
    }

    def group_qualifies(group: Sequence[Mapping[str, str]]) -> bool:
        observed_cells = {(row[first], row[second]) for row in group}
        if not required_cells.issubset(observed_cells):
            return False
        if first == "preparation_condition":
            for preparation in (first_low, first_high):
                section_levels: dict[str, set[str]] = defaultdict(set)
                for row in group:
                    if row[first] == preparation:
                        section_levels[row["section_id"]].add(row[second])
                if not any(
                    {second_low, second_high}.issubset(levels) for levels in section_levels.values()
                ):
                    return False
        return True

    def qualifying_groups(
        fields: Sequence[str],
    ) -> list[tuple[tuple[str, ...], list[Mapping[str, str]]]]:
        return [
            (key, group)
            for key, group in sorted(grouped_rows(rows, fields).items())
            if group_qualifies(group)
        ]

    direct_groups = qualifying_groups(held_fields)
    supporting_strata: list[list[str]] = []
    biological_supporters: set[str] = set()
    supporting_sections: set[str] = set()
    supporting_observations: set[str] = set()
    for key, group in direct_groups:
        supporting_strata.append(list(key))
        biological_supporters.add(group[0]["biological_unit"])
        supporting_sections.update(row["section_id"] for row in group)
        supporting_observations.update(row["observation_id"] for row in group)

    def carrier_values(field: str) -> list[str]:
        fields = tuple(dict.fromkeys((*held_fields, field)))
        carriers = qualifying_groups(fields)
        return sorted({row[field] for _, group in carriers for row in group})

    fold_metadata_present = bool(rows) and "fold_id" in rows[0]
    supporting_blocks = carrier_values("block_id")
    supporting_preparation_batches = carrier_values("preparation_batch")
    supporting_scan_batches = carrier_values("scan_batch")
    return {
        "held_fields": list(held_fields),
        "supporting_strata": supporting_strata,
        "supporting_stratum_count": len(supporting_strata),
        "complete_rectangle_count": len(supporting_strata),
        "biological_supporters": sorted(biological_supporters),
        "biological_supporter_count": len(biological_supporters),
        "supporting_blocks": supporting_blocks,
        "supporting_block_count": len(supporting_blocks),
        "supporting_sections": sorted(supporting_sections),
        "supporting_section_count": len(supporting_sections),
        "supporting_observations": sorted(supporting_observations),
        "supporting_observation_count": len(supporting_observations),
        "supporting_preparation_batches": supporting_preparation_batches,
        "supporting_preparation_batch_count": len(supporting_preparation_batches),
        "supporting_scan_batches": supporting_scan_batches,
        "supporting_scan_batch_count": len(supporting_scan_batches),
        "fold_metadata_present": fold_metadata_present,
        "supporting_folds": carrier_values("fold_id") if fold_metadata_present else [],
    }


def interaction_analysis(
    rows: Sequence[Mapping[str, str]],
    pairwise: Mapping[str, Mapping[str, Any]],
    requested_interactions: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    requested_set = set(requested_interactions)
    for interaction_name, (first, second) in INTERACTION_SPECS.items():
        matrix, metadata = design_matrix(rows, (interaction_name,))
        summary = rank_summary(matrix, metadata)
        first_levels = metadata["levels"][first]
        second_levels = metadata["levels"][second]
        contrasts = []
        for first_low, first_high in combinations(first_levels, 2):
            for second_low, second_high in combinations(second_levels, 2):
                contrast = difference_in_differences_vector(
                    metadata,
                    first,
                    first_low,
                    first_high,
                    second,
                    second_low,
                    second_high,
                )
                direct_support = interaction_direct_support(
                    rows,
                    first,
                    first_low,
                    first_high,
                    second,
                    second_low,
                    second_high,
                )
                contrasts.append(
                    {
                        "first_levels": [first_low, first_high],
                        "second_levels": [second_low, second_high],
                        "algebraically_estimable": contrast_is_estimable(matrix, contrast),
                        "direct_support": direct_support,
                    }
                )
        cell_biology: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in rows:
            cell_biology[(row[first], row[second])].add(row["biological_unit"])
        minimum_biology = min((len(values) for values in cell_biology.values()), default=0)
        pair = pairwise[
            pair_key(first, second) if (first, second) in PAIR_SPECS else pair_key(second, first)
        ]
        full_crossing = pair["coverage"]["numerator"] == pair["coverage"]["denominator"]
        all_estimable = bool(contrasts) and all(
            item["algebraically_estimable"] for item in contrasts
        )
        minimum_direct_biology = min(
            (item["direct_support"]["biological_supporter_count"] for item in contrasts),
            default=0,
        )
        if (
            all_estimable
            and full_crossing
            and minimum_biology >= 2
            and minimum_direct_biology >= 2
            and summary["unique_design_residual_degrees_of_freedom"] > 0
        ):
            verdict = VERDICT_DIRECT
        elif (
            all_estimable
            and minimum_biology >= 2
            and minimum_direct_biology >= 2
            and summary["unique_design_residual_degrees_of_freedom"] > 0
        ):
            verdict = VERDICT_PARTIAL
        elif all_estimable:
            verdict = VERDICT_ASSUMPTION
        else:
            verdict = VERDICT_NOT
        result[interaction_name] = {
            "factors": [first, second],
            "requested": interaction_name in requested_set,
            "verdict": verdict,
            "structural_verdict": verdict,
            "full_pairwise_crossing": full_crossing,
            "minimum_biological_units_per_observed_cell": minimum_biology,
            "minimum_direct_biological_supporters_per_contrast": minimum_direct_biology,
            "rank": summary,
            "difference_in_differences_contrasts": contrasts,
        }
    return result


def finding(code: str, detail: str = "") -> dict[str, str]:
    return {"code": code, "detail": detail}


def relationship_status(nesting: Mapping[str, Any], source: str, target: str) -> str:
    for relationship in nesting["relationships"]:
        if relationship["source_factor"] == source and relationship["target_factor"] == target:
            return str(relationship["status"])
    raise RankCalculationError("nesting_relationship_missing", f"{source}|{target}")


def categorical_relationship(
    rows: Sequence[Mapping[str, str]],
    first: str,
    second: str,
) -> dict[str, Any]:
    first_levels = sorted({row[first] for row in rows})
    second_levels = sorted({row[second] for row in rows})
    pairs = sorted({(row[first], row[second]) for row in rows})
    first_degrees = {
        level: len({right for left, right in pairs if left == level})
        for level in first_levels
    }
    second_degrees = {
        level: len({left for left, right in pairs if right == level})
        for level in second_levels
    }
    first_nested = bool(first_levels) and all(value == 1 for value in first_degrees.values())
    second_nested = bool(second_levels) and all(value == 1 for value in second_degrees.values())
    possible = len(first_levels) * len(second_levels)
    constant_axis = len(first_levels) < 2 or len(second_levels) < 2
    if constant_axis:
        status = "constant axis"
    elif first_nested and second_nested and len(first_levels) > 1 and len(second_levels) > 1:
        status = "exact one-to-one alias"
    elif first_nested and len(second_levels) > 1:
        status = f"{first} nested in {second}"
    elif second_nested and len(first_levels) > 1:
        status = f"{second} nested in {first}"
    elif len(pairs) == possible:
        status = "fully crossed"
    else:
        status = "partial association"
    return {
        "first_field": first,
        "second_field": second,
        "status": status,
        "observed_combinations": len(pairs),
        "possible_combinations": possible,
        "coverage": exact_fraction(len(pairs), possible),
        "first_degrees": first_degrees,
        "second_degrees": second_degrees,
        "first_nested_in_second": first_nested,
        "second_nested_in_first": second_nested,
        "one_to_one_alias": (
            first_nested
            and second_nested
            and len(first_levels) > 1
            and len(second_levels) > 1
        ),
        "constant_axis": constant_axis,
    }


def carrier_relationship_analysis(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    preparation_scan_batch = categorical_relationship(
        rows, "preparation_batch", "scan_batch"
    )
    if preparation_scan_batch["constant_axis"]:
        adjustment_status = "constant axis; separate nuisance adjustment unavailable"
    elif preparation_scan_batch["one_to_one_alias"]:
        adjustment_status = "exact aliasing; separate nuisance adjustment unavailable"
    elif preparation_scan_batch["status"] == "fully crossed":
        adjustment_status = "independent enough for separate nuisance adjustment structurally"
    else:
        adjustment_status = "partial association; separate nuisance adjustment qualified"
    preparation_scan_batch["nuisance_adjustment_status"] = adjustment_status
    return {
        "preparation_batch_vs_scan_batch": preparation_scan_batch,
        "block_vs_biological_unit": categorical_relationship(
            rows, "block_id", "biological_unit"
        ),
        "block_vs_preparation": categorical_relationship(
            rows, "block_id", "preparation_condition"
        ),
        "preparation_batch_vs_preparation": categorical_relationship(
            rows, "preparation_batch", "preparation_condition"
        ),
        "scan_batch_vs_preparation": categorical_relationship(
            rows, "scan_batch", "preparation_condition"
        ),
        "scan_batch_vs_scanner": categorical_relationship(rows, "scan_batch", "scanner"),
        "scan_batch_vs_workflow": categorical_relationship(
            rows, "scan_batch", "site_workflow"
        ),
        "technical_replicate_vs_section": categorical_relationship(
            rows, "technical_replicate", "section_id"
        ),
        "technical_replicate_vs_biological_unit": categorical_relationship(
            rows, "technical_replicate", "biological_unit"
        ),
    }


ORDER_FACTOR_SPECS = {
    "biological_unit": (
        "preparation_condition",
        "scanner",
        "site_workflow",
        "scan_batch",
    ),
    "block_id": (
        "biological_unit",
        "preparation_condition",
        "scanner",
        "site_workflow",
        "scan_batch",
    ),
    "preparation_condition": (
        "biological_unit",
        "block_id",
        "scanner",
        "site_workflow",
        "preparation_batch",
        "scan_batch",
    ),
    "scanner": (
        "biological_unit",
        "block_id",
        "section_id",
        "preparation_condition",
        "site_workflow",
        "scan_batch",
    ),
    "site_workflow": (
        "biological_unit",
        "block_id",
        "section_id",
        "preparation_condition",
        "scanner",
        "scan_batch",
    ),
    "preparation_batch": (
        "biological_unit",
        "block_id",
        "preparation_condition",
        "scanner",
        "site_workflow",
        "scan_batch",
    ),
}


def acquisition_order_analysis(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    factors: dict[str, Any] = {}
    findings: list[dict[str, str]] = []
    for factor in (
        "biological_unit",
        "block_id",
        "preparation_condition",
        "scanner",
        "site_workflow",
        "scan_batch",
        "preparation_batch",
    ):
        levels = sorted_levels(rows, factor)
        if factor == "scan_batch":
            factors[factor] = {
                "status": "order metadata insufficient",
                "reason": (
                    "acquisition_order is comparable within, not between, scan batches"
                ),
                "level_pair_summaries": [],
            }
            continue
        group_fields = ORDER_FACTOR_SPECS[factor]
        pair_summaries: list[dict[str, Any]] = []
        factor_statuses: list[str] = []
        for first_level, second_level in combinations(levels, 2):
            forward = 0
            reverse = 0
            interleaved = 0
            eligible = 0
            for _, group in sorted(grouped_rows(rows, group_fields).items()):
                first_orders = sorted(
                    int(row["acquisition_order"])
                    for row in group
                    if row[factor] == first_level
                )
                second_orders = sorted(
                    int(row["acquisition_order"])
                    for row in group
                    if row[factor] == second_level
                )
                if not first_orders or not second_orders:
                    continue
                eligible += 1
                if max(first_orders) < min(second_orders):
                    forward += 1
                elif max(second_orders) < min(first_orders):
                    reverse += 1
                else:
                    interleaved += 1
            separated_forward = 0
            separated_reverse = 0
            comparable_batches = 0
            for _, batch_rows in sorted(grouped_rows(rows, ("scan_batch",)).items()):
                first_batch_orders = sorted(
                    int(row["acquisition_order"])
                    for row in batch_rows
                    if row[factor] == first_level
                )
                second_batch_orders = sorted(
                    int(row["acquisition_order"])
                    for row in batch_rows
                    if row[factor] == second_level
                )
                if not first_batch_orders or not second_batch_orders:
                    continue
                comparable_batches += 1
                if max(first_batch_orders) < min(second_batch_orders):
                    separated_forward += 1
                elif max(second_batch_orders) < min(first_batch_orders):
                    separated_reverse += 1
            perfect_batch_separation = bool(comparable_batches) and (
                separated_forward == comparable_batches
                or separated_reverse == comparable_batches
            )
            fixed_within_strata = bool(eligible) and (
                (forward == eligible and reverse == 0)
                or (reverse == eligible and forward == 0)
            )
            if perfect_batch_separation or fixed_within_strata:
                status = "order-confounded"
                direction = (
                    f"{first_level}_before_{second_level}"
                    if forward or separated_forward
                    else f"{second_level}_before_{first_level}"
                )
                if perfect_batch_separation:
                    append_unique_finding(
                        findings,
                        "acquisition_order_perfectly_separated_by_factor",
                        f"{factor}|{direction}|scan_batches={comparable_batches}",
                    )
                if fixed_within_strata:
                    append_unique_finding(
                        findings,
                        "fixed_acquisition_order_within_matched_strata",
                        f"{factor}|{direction}|{eligible}/{eligible}",
                    )
            elif eligible == 0:
                status = "counterbalancing not demonstrated"
            elif eligible and forward == reverse:
                status = "counterbalancing demonstrated"
            else:
                status = "order-imbalanced"
                append_unique_finding(
                    findings,
                    "acquisition_order_imbalanced",
                    (
                        f"{factor}|{first_level}|{second_level}|"
                        f"forward={forward}|reverse={reverse}|interleaved={interleaved}"
                    ),
                )
            pair_summaries.append(
                {
                    "first_level": first_level,
                    "second_level": second_level,
                    "eligible_matched_strata": eligible,
                    "first_before_second": forward,
                    "second_before_first": reverse,
                    "interleaved": interleaved,
                    "comparable_scan_batches": comparable_batches,
                    "scan_batches_first_before_second": separated_forward,
                    "scan_batches_second_before_first": separated_reverse,
                    "perfect_scan_batch_separation": perfect_batch_separation,
                    "directional_consistency": exact_fraction(
                        max(forward, reverse), eligible
                    )
                    if eligible
                    else None,
                    "status": status,
                }
            )
            factor_statuses.append(status)
        priority = {
            "counterbalancing demonstrated": 0,
            "counterbalancing not demonstrated": 1,
            "order metadata insufficient": 1,
            "order-imbalanced": 2,
            "order-confounded": 3,
        }
        overall = max(factor_statuses, key=lambda value: priority[value]) if factor_statuses else (
            "order metadata insufficient"
        )
        factors[factor] = {
            "status": overall,
            "held_fields": list(group_fields),
            "level_pair_summaries": pair_summaries,
        }
    findings.sort(key=lambda item: (item["code"], item["detail"]))
    return {
        "order_domain": "scan_batch",
        "factor_assessments": factors,
        "findings": findings,
        "causal_bias_claimed": False,
        "interpretation": (
            "Order associations identify operational confounding or imbalance; "
            "they do not prove that order caused an outcome difference."
        ),
    }


def append_unique_finding(findings: list[dict[str, str]], code: str, detail: str = "") -> None:
    candidate = finding(code, detail)
    if candidate not in findings:
        findings.append(candidate)


def structural_findings(
    rows: Sequence[Mapping[str, str]],
    pairwise: Mapping[str, Mapping[str, Any]],
    nesting: Mapping[str, Any],
    graph: Mapping[str, Any],
    main_rank: Mapping[str, Any],
    main_effects: Mapping[str, Mapping[str, Any]],
    interactions: Mapping[str, Mapping[str, Any]],
    carrier_relationships: Mapping[str, Mapping[str, Any]],
    requested_effects: Sequence[str],
    requested_interactions: Sequence[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    blocking: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    requested = set(requested_effects)

    if not graph["connected"]:
        append_unique_finding(
            blocking,
            "disconnected_factor_graph",
            f"components={graph['component_count']}",
        )
    rank_confounds_requested = any(
        not contrast["algebraically_estimable"]
        for effect in requested_effects
        for contrast in main_effects[effect]["contrasts"]
    )
    if main_rank["rank_deficiency"] and rank_confounds_requested:
        append_unique_finding(
            blocking,
            "rank_deficient_fixed_effect_design",
            f"rank={main_rank['rank']}/{main_rank['column_count']}",
        )
    elif main_rank["rank_deficiency"]:
        append_unique_finding(
            warnings,
            "rank_deficiency_outside_requested_contrasts",
            f"rank={main_rank['rank']}/{main_rank['column_count']}",
        )
    for alias in nesting["one_to_one_aliases"]:
        first = alias["first_factor"]
        second = alias["second_factor"]
        relevant = {
            "preparation_condition": "preparation",
            "scanner": "scanner",
            "site_workflow": "site_workflow",
        }
        named = {relevant[factor] for factor in (first, second) if factor in relevant}
        if named & requested and all(
            len(sorted_levels(rows, factor)) > 1 for factor in (first, second)
        ):
            append_unique_finding(
                blocking,
                "exact_one_to_one_factor_aliasing",
                f"{first}|{second}",
            )

    nesting_rules = (
        ("preparation", "preparation_condition", "scanner"),
        ("scanner", "scanner", "preparation_condition"),
        ("site_workflow", "site_workflow", "preparation_condition"),
        ("site_workflow", "site_workflow", "scanner"),
        ("site_workflow", "preparation_condition", "site_workflow"),
        ("site_workflow", "scanner", "site_workflow"),
    )
    for requested_effect, source, target in nesting_rules:
        if (
            requested_effect in requested
            and len(sorted_levels(rows, target)) > 1
            and relationship_status(nesting, source, target) == "exact nesting"
        ):
            append_unique_finding(
                blocking,
                f"{source}_nested_in_{target}",
            )
    if (
        "preparation" in requested
        and relationship_status(nesting, "biological_unit", "preparation_condition")
        == "exact nesting"
    ):
        append_unique_finding(blocking, "biology_assigned_to_only_one_preparation")

    prep_scanner = pairwise[pair_key("preparation_condition", "scanner")]
    if {"preparation", "scanner"}.issubset(requested):
        for level, degree in sorted(prep_scanner["degrees"]["preparation_condition"].items()):
            if degree < 2:
                append_unique_finding(
                    warnings,
                    "preparation_level_without_scanner_replication",
                    level,
                )
        for level, degree in sorted(prep_scanner["degrees"]["scanner"].items()):
            if degree < 2:
                append_unique_finding(
                    warnings,
                    "scanner_level_without_preparation_replication",
                    level,
                )

    if "preparation" in requested:
        bio_prep = pairwise[pair_key("biological_unit", "preparation_condition")]
        for level, degree in sorted(bio_prep["degrees"]["biological_unit"].items()):
            if degree < 2:
                append_unique_finding(
                    warnings,
                    "biological_unit_without_preparation_replication",
                    level,
                )

    if "site_workflow" in requested:
        prep_site = pairwise[pair_key("preparation_condition", "site_workflow")]
        scanner_site = pairwise[pair_key("scanner", "site_workflow")]
        for site in sorted_levels(rows, "site_workflow"):
            prep_degree = prep_site["degrees"]["site_workflow"][site]
            scanner_degree = scanner_site["degrees"]["site_workflow"][site]
            if prep_degree < 2:
                append_unique_finding(
                    warnings,
                    "site_workflow_without_preparation_replication",
                    site,
                )
            if scanner_degree < 2:
                append_unique_finding(
                    warnings,
                    "site_workflow_without_scanner_replication",
                    site,
                )

    for effect in requested_effects:
        effect_result = main_effects[effect]
        if effect_result["verdict"] in {VERDICT_DIRECT, VERDICT_PARTIAL}:
            continue
        contrast_support = [
            item["support"]["biological_supporter_count"] for item in effect_result["contrasts"]
        ]
        if contrast_support and min(contrast_support) < 2:
            append_unique_finding(
                blocking,
                f"{effect}_contrast_fewer_than_two_biological_units",
            )
        elif effect_result["verdict"] == VERDICT_ASSUMPTION:
            append_unique_finding(warnings, f"{effect}_estimable_only_under_assumptions")
        else:
            append_unique_finding(blocking, f"{effect}_contrast_not_estimable")

    for interaction_name in requested_interactions:
        interaction = interactions[interaction_name]
        if interaction["verdict"] in {VERDICT_DIRECT, VERDICT_PARTIAL}:
            continue
        if interaction["minimum_biological_units_per_observed_cell"] < 2:
            append_unique_finding(
                blocking,
                "interaction_insufficient_biological_replication",
                interaction_name,
            )
        elif interaction["verdict"] == VERDICT_ASSUMPTION:
            append_unique_finding(
                warnings,
                "interaction_estimable_only_under_assumptions",
                interaction_name,
            )
        else:
            append_unique_finding(blocking, "interaction_not_estimable", interaction_name)

    for relationship in nesting["relationships"]:
        if relationship["status"] == "partial nesting":
            append_unique_finding(
                warnings,
                "partial_nesting",
                f"{relationship['source_factor']}|{relationship['target_factor']}|"
                f"{relationship['singleton_level_count']}/{relationship['level_count']}",
            )

    batch_relationships = (
        (
            "preparation_batch",
            "preparation_condition",
            "preparation_batches_nested_within_preparation",
        ),
        ("scan_batch", "preparation_condition", "scan_batches_nested_within_preparation"),
        ("scan_batch", "scanner", "scan_batches_nested_within_scanner"),
        ("scan_batch", "site_workflow", "scan_batches_nested_within_site_workflow"),
    )
    for batch, factor, code in batch_relationships:
        if levels_nested_within_factor(rows, batch, factor):
            append_unique_finding(warnings, code)

    prep_scan_batches = carrier_relationships["preparation_batch_vs_scan_batch"]
    if prep_scan_batches["one_to_one_alias"]:
        append_unique_finding(
            warnings,
            "preparation_batch_scan_batch_exact_alias",
            (
                f"observed={prep_scan_batches['observed_combinations']}|"
                f"possible={prep_scan_batches['possible_combinations']}"
            ),
        )
    elif prep_scan_batches["constant_axis"]:
        append_unique_finding(
            warnings,
            "preparation_batch_scan_batch_constant_axis",
            "separate nuisance adjustment unavailable",
        )
    elif prep_scan_batches["status"] == "partial association":
        append_unique_finding(
            warnings,
            "preparation_batch_scan_batch_partial_association",
            (
                f"observed={prep_scan_batches['observed_combinations']}|"
                f"possible={prep_scan_batches['possible_combinations']}"
            ),
        )

    block_biology = carrier_relationships["block_vs_biological_unit"]
    if block_biology["one_to_one_alias"]:
        append_unique_finding(
            warnings,
            "block_biological_unit_one_to_one_alias",
            "block and biological unit are not independent replication layers",
        )

    block_preparation = carrier_relationships["block_vs_preparation"]
    if block_preparation["first_nested_in_second"] and len(
        block_preparation["first_degrees"]
    ) > 1:
        append_unique_finding(
            warnings,
            "block_nested_in_preparation_condition",
            "preparation separation requires between-block exchangeability",
        )

    blocking.sort(key=lambda item: (item["code"], item["detail"]))
    warnings.sort(key=lambda item: (item["code"], item["detail"]))
    return blocking, warnings


def randomization_metadata_summary(headers: Sequence[str]) -> dict[str, Any]:
    required_controls = (
        "biological_unit",
        "block_id",
        "section_id",
        "preparation_batch",
        "scan_batch",
        "acquisition_order",
        "technical_replicate",
        "biological_replicate",
        "site_workflow",
    )
    optional_present = [field for field in OPTIONAL_COLUMNS if field in headers]
    optional_missing = [field for field in OPTIONAL_COLUMNS if field not in headers]
    return {
        "required_control_fields_present": list(required_controls),
        "recognized_optional_fields_present": optional_present,
        "recognized_optional_fields_missing": optional_missing,
        "randomization_execution_verified": False,
        "interpretation": (
            "Field availability supports a future execution audit; it does not "
            "verify random assignment or balanced temporal execution."
        ),
    }


def design_classification(
    pairwise: Mapping[str, Mapping[str, Any]],
    higher_order: Mapping[str, Mapping[str, Any]],
    nesting: Mapping[str, Any],
    graph: Mapping[str, Any],
    main_rank: Mapping[str, Any],
    main_effects: Mapping[str, Mapping[str, Any]],
    requested_effects: Sequence[str],
    interactions: Mapping[str, Mapping[str, Any]],
    requested_interactions: Sequence[str],
    blocking: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    labels: list[str] = []
    requested_factors = {"biological_unit"}
    requested_factors.update(EFFECT_TO_FACTOR[effect] for effect in requested_effects)
    for interaction_name in requested_interactions:
        requested_factors.update(INTERACTION_SPECS[interaction_name])
    relevant_pairs = [
        pair
        for pair in pairwise.values()
        if {pair["first_factor"], pair["second_factor"]}.issubset(requested_factors)
    ]
    pairwise_complete = all(
        pair["coverage"]["numerator"] == pair["coverage"]["denominator"]
        for pair in relevant_pairs
    )
    requested_product = higher_order["requested_factor_product"]
    full_factor_crossing = (
        requested_product["coverage"]["numerator"]
        == requested_product["coverage"]["denominator"]
    )
    blocking_codes = {item["code"] for item in blocking}
    relevant_nesting = any(
        "_nested_in_" in code or code == "biology_assigned_to_only_one_preparation"
        for code in blocking_codes
    )
    rank_confounds_requested = any(
        not contrast["algebraically_estimable"]
        for effect in requested_effects
        for contrast in main_effects[effect]["contrasts"]
    )
    if not graph["connected"]:
        labels.append("disconnected")
    if relevant_nesting:
        labels.append("nested/confounded")
    if rank_confounds_requested or "exact_one_to_one_factor_aliasing" in blocking_codes:
        labels.append("nested/confounded")
    if any(
        main_effects[effect]["verdict"] == VERDICT_ASSUMPTION for effect in requested_effects
    ) or any(
        interactions[name]["verdict"] == VERDICT_ASSUMPTION for name in requested_interactions
    ):
        labels.append("estimable only under additional assumptions")
    if graph["connected"]:
        if full_factor_crossing:
            labels.append("fully crossed")
        elif pairwise_complete:
            labels.append("pairwise complete, higher-order incomplete")
        else:
            labels.append("partially crossed")
    if not labels:
        labels.append("nested/confounded")
    precedence = (
        "disconnected",
        "nested/confounded",
        "estimable only under additional assumptions",
        "pairwise complete, higher-order incomplete",
        "partially crossed",
        "fully crossed",
    )
    primary = next(label for label in precedence if label in labels)
    return {
        "primary": primary,
        "labels": list(dict.fromkeys(labels)),
        "pairwise_complete_for_requested_factors": pairwise_complete,
        "requested_factor_product_complete": full_factor_crossing,
    }


def add_support_warnings(
    warnings: list[dict[str, str]],
    main_effects: Mapping[str, Mapping[str, Any]],
    requested_effects: Sequence[str],
) -> None:
    for effect in requested_effects:
        for contrast in main_effects[effect]["contrasts"]:
            support = contrast["support"]
            level_pair = f"{contrast['first_level']}|{contrast['second_level']}"
            if len(support["supporting_blocks"]) < 2:
                append_unique_finding(
                    warnings,
                    "contrast_supported_by_fewer_than_two_blocks",
                    f"{effect}|{level_pair}",
                )
            if len(support["supporting_preparation_batches"]) < 2:
                append_unique_finding(
                    warnings,
                    "contrast_supported_by_fewer_than_two_preparation_batches",
                    f"{effect}|{level_pair}",
                )
            if len(support["supporting_scan_batches"]) < 2:
                append_unique_finding(
                    warnings,
                    "contrast_supported_by_fewer_than_two_scan_batches",
                    f"{effect}|{level_pair}",
                )
            if support["fold_metadata_present"] and len(support["supporting_folds"]) < 2:
                append_unique_finding(
                    warnings,
                    "contrast_supported_by_fewer_than_two_folds",
                    f"{effect}|{level_pair}",
                )
            if support["ambiguous_matched_serial_section_strata_count"]:
                append_unique_finding(
                    warnings,
                    "matched_serial_section_pairing_ambiguous",
                    (
                        f"{effect}|{level_pair}|strata="
                        f"{support['ambiguous_matched_serial_section_strata_count']}"
                    ),
                )


def add_interaction_support_warnings(
    warnings: list[dict[str, str]],
    interactions: Mapping[str, Mapping[str, Any]],
    requested_interactions: Sequence[str],
) -> None:
    carrier_specs = (
        ("supporting_blocks", "blocks"),
        ("supporting_preparation_batches", "preparation_batches"),
        ("supporting_scan_batches", "scan_batches"),
    )
    for interaction_name in requested_interactions:
        for contrast in interactions[interaction_name]["difference_in_differences_contrasts"]:
            first_levels = "~".join(contrast["first_levels"])
            second_levels = "~".join(contrast["second_levels"])
            detail = f"{interaction_name}|{first_levels}|{second_levels}"
            support = contrast["direct_support"]
            for field, label in carrier_specs:
                if len(support[field]) < 2:
                    append_unique_finding(
                        warnings,
                        f"interaction_contrast_supported_by_fewer_than_two_{label}",
                        detail,
                    )
            if support["fold_metadata_present"] and len(support["supporting_folds"]) < 2:
                append_unique_finding(
                    warnings,
                    "interaction_contrast_supported_by_fewer_than_two_folds",
                    detail,
                )


def operational_validity_for_factors(
    factors: Sequence[str],
    order_analysis: Mapping[str, Any],
    batch_confounded: bool,
) -> dict[str, Any]:
    qualifications: list[str] = []
    factor_assessments = order_analysis["factor_assessments"]
    order_statuses = [factor_assessments[factor]["status"] for factor in factors]
    if "order-confounded" in order_statuses:
        qualifications.append("order-confounded")
    elif "order-imbalanced" in order_statuses:
        qualifications.append("order-imbalanced")
    elif any(
        status in {"counterbalancing not demonstrated", "order metadata insufficient"}
        for status in order_statuses
    ):
        qualifications.append("counterbalancing not demonstrated")
    if batch_confounded:
        qualifications.append("batch-confounded")
    if "site_workflow" in factors:
        qualifications.append("workflow under-specified")
    qualifications.append("randomization/counterbalancing unverified")
    priority = (
        "order-confounded",
        "batch-confounded",
        "order-imbalanced",
        "workflow under-specified",
        "counterbalancing not demonstrated",
        "randomization/counterbalancing unverified",
    )
    overall = next(item for item in priority if item in qualifications)
    return {
        "status": overall,
        "qualifications": qualifications,
        "interpretation": (
            "Operational qualifications do not change the separately reported "
            "sampling-matrix structural verdict and do not prove causal bias."
        ),
    }


def attach_operational_validity(
    rows: Sequence[Mapping[str, str]],
    main_effects: dict[str, dict[str, Any]],
    interactions: dict[str, dict[str, Any]],
    order_analysis: Mapping[str, Any],
) -> None:
    batch_dependencies = {
        "preparation": (
            levels_nested_within_factor(rows, "preparation_batch", "preparation_condition")
            or levels_nested_within_factor(rows, "scan_batch", "preparation_condition")
        ),
        "scanner": levels_nested_within_factor(rows, "scan_batch", "scanner"),
        "site_workflow": levels_nested_within_factor(rows, "scan_batch", "site_workflow"),
    }
    for effect, factor in EFFECT_TO_FACTOR.items():
        assessment = operational_validity_for_factors(
            (factor,), order_analysis, batch_dependencies[effect]
        )
        main_effects[effect]["operational_validity"] = assessment
        for contrast in main_effects[effect]["contrasts"]:
            contrast["operational_validity"] = assessment
    for interaction_name, (first, second) in INTERACTION_SPECS.items():
        batch_confounded = any(
            batch_dependencies[effect]
            for effect, factor in EFFECT_TO_FACTOR.items()
            if factor in {first, second}
        )
        interactions[interaction_name]["operational_validity"] = (
            operational_validity_for_factors(
                (first, second), order_analysis, batch_confounded
            )
        )


def analyze_design(
    rows: Sequence[Mapping[str, str]],
    input_sha256: str,
    input_label: str,
    headers: Sequence[str],
    requested_effects: Sequence[str],
    requested_interactions: Sequence[str],
) -> dict[str, Any]:
    physical_identity_count = len(
        {
            (
                row["section_id"],
                row["preparation_condition"],
                row["scanner"],
                row["site_workflow"],
            )
            for row in rows
        }
    )
    inventory = factor_inventory(rows)
    pairwise = pairwise_crossing(rows)
    higher_order = higher_order_crossing(
        rows, requested_effects, requested_interactions
    )
    nesting = nesting_analysis(pairwise)
    carrier_relationships = carrier_relationship_analysis(rows)
    graph = global_factor_graph(rows)
    main_matrix, main_metadata = design_matrix(rows)
    main_rank = rank_summary(main_matrix, main_metadata)
    main_effects = main_effect_contrasts(
        rows,
        main_matrix,
        main_metadata,
        pairwise,
        requested_effects,
    )
    apply_batch_assumptions(rows, main_effects)
    interactions = interaction_analysis(rows, pairwise, requested_interactions)
    order_analysis = acquisition_order_analysis(rows)
    attach_operational_validity(rows, main_effects, interactions, order_analysis)
    blocking, warnings = structural_findings(
        rows,
        pairwise,
        nesting,
        graph,
        main_rank,
        main_effects,
        interactions,
        carrier_relationships,
        requested_effects,
        requested_interactions,
    )
    randomization = randomization_metadata_summary(headers)
    for field in randomization["recognized_optional_fields_missing"]:
        append_unique_finding(warnings, "optional_control_metadata_not_recorded", field)
    append_unique_finding(warnings, "randomization_execution_not_verified")
    for item in order_analysis["findings"]:
        append_unique_finding(warnings, item["code"], item["detail"])
    append_unique_finding(warnings, "serial_sections_not_identical_pixels")
    append_unique_finding(
        warnings,
        "preparation_condition_semantics_require_intervention_stage_documentation",
    )
    append_unique_finding(warnings, "scanner_level_semantics_require_device_instance_documentation")
    append_unique_finding(
        warnings,
        "site_workflow_semantics_require_process_stage_documentation",
    )
    add_support_warnings(warnings, main_effects, requested_effects)
    add_interaction_support_warnings(warnings, interactions, requested_interactions)
    warnings.sort(key=lambda item: (item["code"], item["detail"]))

    classification = design_classification(
        pairwise,
        higher_order,
        nesting,
        graph,
        main_rank,
        main_effects,
        requested_effects,
        interactions,
        requested_interactions,
        blocking,
    )
    if blocking:
        overall_status = "not_identifiable"
    elif any(
        main_effects[effect]["verdict"] == VERDICT_ASSUMPTION for effect in requested_effects
    ) or any(
        interactions[name]["verdict"] == VERDICT_ASSUMPTION for name in requested_interactions
    ):
        overall_status = "assumption_dependent"
    else:
        overall_status = "identifiable"

    future_targets = {
        "scanner_suppressed_residual_association_with_preparation": {
            "verdict": worst_verdict(
                [main_effects["preparation"]["verdict"], main_effects["scanner"]["verdict"]]
            ),
            "future_test_only": True,
            "interpretation": (
                "The design verdict is the worse of the preparation and scanner "
                "prerequisites; scanner suppression and residual association "
                "remain untested."
            ),
        },
        "scanner_suppressed_residual_association_with_site_workflow": {
            "verdict": worst_verdict(
                [main_effects["site_workflow"]["verdict"], main_effects["scanner"]["verdict"]]
            ),
            "future_test_only": True,
            "interpretation": (
                "The design verdict is the worse of the workflow and scanner "
                "prerequisites; scanner suppression and residual association "
                "remain untested."
            ),
        },
    }

    return {
        "schema_version": 2,
        "audit_id": AUDIT_ID,
        "input_label": input_label,
        "input_sha256": input_sha256,
        "overall_status": overall_status,
        "design_classification": classification,
        "requested_effects": list(requested_effects),
        "requested_interactions": list(requested_interactions),
        "execution_boundary": {
            "model_training_run": False,
            "representation_payload_loaded": False,
            "experiment_output_modified": False,
            "causal_attribution_claimed": False,
        },
        "design_summary": {
            "observations": len(rows),
            "biological_units": inventory["biological_unit"]["level_count"],
            "blocks": len({row["block_id"] for row in rows}),
            "sections": len({row["section_id"] for row in rows}),
            "preparation_levels": inventory["preparation_condition"]["level_count"],
            "scanners": inventory["scanner"]["level_count"],
            "site_workflow_levels": inventory["site_workflow"]["level_count"],
            "preparation_batches": len({row["preparation_batch"] for row in rows}),
            "scan_batches": len({row["scan_batch"] for row in rows}),
            "physical_acquisition_identities": physical_identity_count,
            "intentional_repeat_acquisition_rows": len(rows) - physical_identity_count,
            "repeat_acquisition_id_recorded": "repeat_acquisition_id" in headers,
        },
        "factor_inventory": inventory,
        "pairwise_crossing": pairwise,
        "higher_order_crossing": higher_order,
        "nesting": nesting,
        "carrier_relationships": carrier_relationships,
        "graph_connectedness": graph,
        "rank_summary": {
            "main_effects": main_rank,
            "interactions": interactions,
        },
        "contrast_verdicts": {
            "biology_controlled_preparation_effect": main_effects["preparation"],
            "biology_preparation_controlled_scanner_effect": main_effects["scanner"],
            "biology_preparation_scanner_controlled_site_workflow_effect": (
                main_effects["site_workflow"]
            ),
            **future_targets,
        },
        "randomization_and_blocking_metadata": {
            **randomization,
            "acquisition_order_analysis": order_analysis,
        },
        "blocking_findings": blocking,
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }


def finalize_audit(audit: dict[str, Any], self_tests: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(audit)
    result["self_tests"] = dict(self_tests)
    fingerprint_payload = dict(result)
    fingerprint_payload.pop("audit_fingerprint_sha256", None)
    result["audit_fingerprint_sha256"] = sha256_bytes(
        stable_json(fingerprint_payload).encode("utf-8")
    )
    return result


def render_report(audit: Mapping[str, Any]) -> str:
    summary = audit["design_summary"]
    lines = [
        "# Crossed-preparation identifiability report",
        "",
        "## Status and boundary",
        "",
        f"- Overall status: **{audit['overall_status']}**",
        f"- Primary design classification: **{audit['design_classification']['primary']}**",
        f"- Input: `{audit['input_label']}`",
        f"- Input SHA-256: `{audit['input_sha256']}`",
        f"- Audit fingerprint: `{audit['audit_fingerprint_sha256']}`",
        "- This report audits structural design support only. It does not load representations, train a model, estimate an effect, or make a causal claim.",
        "",
        "## Design summary",
        "",
        f"- Observations: {summary['observations']}",
        f"- Biological units: {summary['biological_units']}",
        f"- Blocks: {summary['blocks']}",
        f"- Sections: {summary['sections']}",
        f"- Preparation levels: {summary['preparation_levels']}",
        f"- Scanners: {summary['scanners']}",
        f"- Site/workflow levels: {summary['site_workflow_levels']}",
        f"- Preparation batches: {summary['preparation_batches']}",
        f"- Scan batches: {summary['scan_batches']}",
        f"- Base physical acquisition identities: {summary['physical_acquisition_identities']}",
        f"- Intentional repeat-acquisition rows beyond base identities: {summary['intentional_repeat_acquisition_rows']}",
        f"- Repeat-acquisition identifier recorded: {'yes' if summary['repeat_acquisition_id_recorded'] else 'no'}",
        "",
        "## Factor-level inventory",
        "",
        "| Factor | Levels | Observations per level | Biological units per level |",
        "|---|---:|---|---|",
    ]
    for factor in FACTORS:
        inventory = audit["factor_inventory"][factor]
        observations = ", ".join(
            f"{level}:{count}" for level, count in inventory["observations_per_level"].items()
        )
        biology = ", ".join(
            f"{level}:{count}" for level, count in inventory["biological_units_per_level"].items()
        )
        lines.append(
            f"| {FACTOR_LABELS[factor]} | {inventory['level_count']} | {observations} | {biology} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise crossing",
            "",
            "| Factor pair | Observed / possible | Coverage | Minimum observations | Minimum biological units | Components |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for first, second in PAIR_SPECS:
        pair = audit["pairwise_crossing"][pair_key(first, second)]
        lines.append(
            f"| {FACTOR_LABELS[first]} x {FACTOR_LABELS[second]} | "
            f"{pair['observed_combinations']} / {pair['possible_combinations']} | "
            f"{pair['coverage']['decimal']} | "
            f"{pair['minimum_observations_per_observed_combination']} | "
            f"{pair['minimum_biological_units_per_observed_combination']} | "
            f"{pair['component_count']} |"
        )

    lines.extend(
        [
            "",
            "## Higher-order crossing",
            "",
            "| Factor product | Observed / possible | Coverage | Minimum observations | Missing combinations |",
            "|---|---:|---:|---:|---|",
        ]
    )
    higher_order_names = (
        "biology_preparation_scanner",
        "preparation_scanner_workflow",
        "biology_preparation_scanner_workflow",
    )
    for name in higher_order_names:
        crossing = audit["higher_order_crossing"][name]
        missing = "; ".join(
            ",".join(f"{factor}={cell[factor]}" for factor in crossing["factors"])
            for cell in crossing["missing_combinations"]
        ) or "none"
        if not crossing["missing_combinations_complete"]:
            missing += f"; truncated at {MISSING_COMBINATION_REPORT_LIMIT}"
        label = " x ".join(FACTOR_LABELS[factor] for factor in crossing["factors"])
        lines.append(
            f"| {label} | {crossing['observed_combinations']} / "
            f"{crossing['possible_combinations']} | {crossing['coverage']['decimal']} | "
            f"{crossing['minimum_observations_per_observed_combination']} | {missing} |"
        )
    requested_product = audit["higher_order_crossing"]["requested_factor_product"]
    displayed_factor_sets = {
        tuple(audit["higher_order_crossing"][name]["factors"])
        for name in higher_order_names
    }
    if tuple(requested_product["factors"]) not in displayed_factor_sets:
        missing = "; ".join(
            ",".join(
                f"{factor}={cell[factor]}" for factor in requested_product["factors"]
            )
            for cell in requested_product["missing_combinations"]
        ) or "none"
        if not requested_product["missing_combinations_complete"]:
            missing += f"; truncated at {MISSING_COMBINATION_REPORT_LIMIT}"
        label = "Requested product: " + " x ".join(
            FACTOR_LABELS[factor] for factor in requested_product["factors"]
        )
        lines.append(
            f"| {label} | {requested_product['observed_combinations']} / "
            f"{requested_product['possible_combinations']} | "
            f"{requested_product['coverage']['decimal']} | "
            f"{requested_product['minimum_observations_per_observed_combination']} | "
            f"{missing} |"
        )
    lines.append("")
    lines.append(
        "Full crossing is assigned only from the complete requested factor product; "
        "pairwise completeness alone is insufficient."
    )

    graph = audit["graph_connectedness"]
    lines.extend(
        [
            "",
            "## Connectedness and nesting",
            "",
            f"- Global factor-incidence components: {graph['component_count']}",
            f"- All factor levels connected: {'yes' if graph['connected'] else 'no'}",
        ]
    )
    reported_relationships = [
        item for item in audit["nesting"]["relationships"] if item["status"] != "not nested"
    ]
    if reported_relationships:
        for relationship in reported_relationships:
            lines.append(
                f"- {relationship['source_factor']} in "
                f"{relationship['target_factor']}: {relationship['status']} "
                f"({relationship['singleton_level_count']}/"
                f"{relationship['level_count']} singleton levels)"
            )
    else:
        lines.append("- Exact or partial nesting relationships: none")

    block_biology = audit["carrier_relationships"]["block_vs_biological_unit"]
    lines.append(
        f"- Block versus biological unit: {block_biology['status']}; "
        "these are not independent replication layers when one-to-one aliased."
    )
    block_preparation = audit["carrier_relationships"]["block_vs_preparation"]
    lines.append(f"- Block versus preparation: {block_preparation['status']}.")
    technical_section = audit["carrier_relationships"]["technical_replicate_vs_section"]
    lines.append(
        f"- Technical replicate versus section: {technical_section['status']}; "
        "technical repeats never add independent biological support."
    )

    main_rank = audit["rank_summary"]["main_effects"]
    lines.extend(
        [
            "",
            "## Rank and interaction summary",
            "",
            f"- Main-effect rank: {main_rank['rank']} / {main_rank['column_count']}",
            f"- Main-effect row-level residual degrees of freedom: {main_rank['row_level_residual_degrees_of_freedom']}",
            f"- Unique design rows: {main_rank['unique_design_row_count']}; unique-design residual degrees of freedom: {main_rank['unique_design_residual_degrees_of_freedom']}",
            "- Row-level residual degrees of freedom are not independent biological degrees of freedom. Repeated scans can increase row-level n without increasing biological replication, and residual df cannot compensate for two biological units.",
            f"- Main-effect aliased columns: {', '.join(main_rank['aliased_columns']) or 'none'}",
            "",
            "| Interaction | Structural verdict | Operational validity | Rank | Row-level residual df | Minimum biological units per cell | Minimum direct biological rectangle supporters |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for name in INTERACTION_SPECS:
        interaction = audit["rank_summary"]["interactions"][name]
        rank = interaction["rank"]
        lines.append(
            f"| {name} | {interaction['structural_verdict']} | "
            f"{interaction['operational_validity']['status']} | "
            f"{rank['rank']} / {rank['column_count']} | "
            f"{rank['row_level_residual_degrees_of_freedom']} | "
            f"{interaction['minimum_biological_units_per_observed_cell']} | "
            f"{interaction['minimum_direct_biological_supporters_per_contrast']} |"
        )

    lines.extend(
        [
            "",
            "## Per-contrast structural support and operational validity",
            "",
            "| Contrast | Structural verdict | Operational validity | Biological units | Blocks | Sections | Bridges | Matched serial-section pairs | Complete rectangles | Preparation batches | Scan batches |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    main_support_specs = (
        (
            "biology_controlled_preparation_effect",
            "Preparation main effect",
        ),
        (
            "biology_preparation_controlled_scanner_effect",
            "Scanner main effect",
        ),
        (
            "biology_preparation_scanner_controlled_site_workflow_effect",
            "Post-preparation workflow main effect",
        ),
    )
    for key, label in main_support_specs:
        effect = audit["contrast_verdicts"][key]
        for contrast in effect["contrasts"]:
            support = contrast["support"]
            level_label = f"{contrast['first_level']} vs {contrast['second_level']}"
            lines.append(
                f"| {label}: {level_label} | {contrast['structural_verdict']} | "
                f"{contrast['operational_validity']['status']} | "
                f"{support['direct_biological_supporter_count']} | "
                f"{support['supporting_block_count']} | "
                f"{support['supporting_section_count']} | {support['bridge_count']} | "
                f"{support['matched_serial_section_pair_count']} | 0 | "
                f"{support['supporting_preparation_batch_count']} | "
                f"{support['supporting_scan_batch_count']} |"
            )
    interaction_labels = {
        "preparation_scanner": "Preparation x scanner",
        "scanner_site_workflow": "Scanner x post-preparation workflow",
        "preparation_site_workflow": "Preparation x post-preparation workflow",
    }
    for name, label in interaction_labels.items():
        interaction = audit["rank_summary"]["interactions"][name]
        for contrast in interaction["difference_in_differences_contrasts"]:
            support = contrast["direct_support"]
            lines.append(
                f"| {label} | {interaction['structural_verdict']} | "
                f"{interaction['operational_validity']['status']} | "
                f"{support['biological_supporter_count']} | "
                f"{support['supporting_block_count']} | "
                f"{support['supporting_section_count']} | 0 | 0 | "
                f"{support['complete_rectangle_count']} | "
                f"{support['supporting_preparation_batch_count']} | "
                f"{support['supporting_scan_batch_count']} |"
            )

    lines.extend(
        [
            "",
            "## Contrast verdicts",
            "",
            "| Contrast | Structural/integrated verdict | Operational validity | Boundary |",
            "|---|---|---|---|",
        ]
    )
    contrast_labels = (
        (
            "biology_controlled_preparation_effect",
            "Biology-controlled preparation effect",
        ),
        (
            "biology_preparation_controlled_scanner_effect",
            "Biology/preparation-controlled scanner effect",
        ),
        (
            "biology_preparation_scanner_controlled_site_workflow_effect",
            "Biology/preparation/scanner-controlled site/workflow effect",
        ),
        (
            "scanner_suppressed_residual_association_with_preparation",
            "Future scanner-suppressed residual association with preparation",
        ),
        (
            "scanner_suppressed_residual_association_with_site_workflow",
            "Future scanner-suppressed residual association with workflow",
        ),
    )
    for key, label in contrast_labels:
        contrast = audit["contrast_verdicts"][key]
        operational = contrast.get("operational_validity", {}).get("status", "future test only")
        structural = contrast.get("structural_verdict", contrast["verdict"])
        lines.append(
            f"| {label} | {structural} | {operational} | {contrast['interpretation']} |"
        )

    randomization = audit["randomization_and_blocking_metadata"]
    lines.extend(
        [
            "",
            "## Randomization and blocking metadata",
            "",
            "- Required structural control fields are present.",
            "- Recognized optional controls present: "
            + (", ".join(randomization["recognized_optional_fields_present"]) or "none"),
            "- Recognized optional controls not recorded: "
            + (", ".join(randomization["recognized_optional_fields_missing"]) or "none"),
            "- Randomized execution verified: no; identifiers make a future execution audit possible but do not prove randomization.",
            "",
            "### Acquisition-order assessment",
            "",
            "| Factor | Status | Matched-stratum direction counts |",
            "|---|---|---|",
        ]
    )
    order_assessments = randomization["acquisition_order_analysis"]["factor_assessments"]
    for factor in (
        "biological_unit",
        "block_id",
        "preparation_condition",
        "scanner",
        "site_workflow",
        "preparation_batch",
        "scan_batch",
    ):
        assessment = order_assessments[factor]
        direction = "; ".join(
            (
                f"{item['first_level']}<{item['second_level']}:{item['first_before_second']}, "
                f"reverse:{item['second_before_first']}, interleaved:{item['interleaved']}"
            )
            for item in assessment["level_pair_summaries"]
        ) or assessment.get("reason", "no eligible matched strata")
        lines.append(f"| {factor} | {assessment['status']} | {direction} |")
    lines.extend(
        [
            "",
            "Order findings are operational diagnostics only; they do not establish causal bias.",
            "",
            "### Batch and hierarchy relationships",
            "",
        ]
    )
    batch_relationship = audit["carrier_relationships"]["preparation_batch_vs_scan_batch"]
    lines.append(
        f"- Preparation batch versus scan batch: {batch_relationship['status']} "
        f"({batch_relationship['observed_combinations']}/"
        f"{batch_relationship['possible_combinations']} combinations); "
        f"{batch_relationship['nuisance_adjustment_status']}."
    )
    lines.append(
        "- Batch counts are bookkeeping/support counts. They are not independent "
        "nuisance axes when an exact alias or partial association is reported."
    )
    lines.extend(
        [
            "",
            "### Workflow boundary",
            "",
            "- WF_POST_A and WF_POST_B denote a post-preparation workflow factor only. The operator or operator pool, transfer/storage condition, post-preparation handling, post-processing pipeline, exposure order, timing window, destructive status, and carryover risk require prospective specification.",
            "- Workflow levels may aggregate multiple operational causes; they are not a single causal mechanism and do not identify upstream preparation-site effects.",
            "",
            "## Findings",
            "",
        ]
    )
    if audit["blocking_findings"]:
        lines.append("### Blocking findings")
        lines.append("")
        for item in audit["blocking_findings"]:
            suffix = f": {item['detail']}" if item["detail"] else ""
            lines.append(f"- `{item['code']}`{suffix}")
        lines.append("")
    else:
        lines.extend(["- Blocking findings: none", ""])
    lines.append("### Warnings and design qualifications")
    lines.append("")
    for item in audit["warnings"]:
        suffix = f": {item['detail']}" if item["detail"] else ""
        lines.append(f"- `{item['code']}`{suffix}")

    tests = audit["self_tests"]
    lines.extend(
        [
            "",
            "## Deterministic adversarial and regression tests",
            "",
            f"- Full suite passed: {tests['passed']} / {tests['total']}",
            f"- Preserved core regression suite passed: {tests['core_regression_passed']} / {tests['core_regression_total']}",
            f"- Required negative cases passed: {tests['required_negative_passed']} / {tests['required_negative_total']}",
            f"- Required patch regression cases passed: {tests['required_patch_cases_passed']} / {tests['required_patch_cases_total']}",
            f"- Temporary fixtures removed: {'yes' if tests['temporary_fixtures_removed'] else 'no'}",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in audit["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "The report supplies design support for future attribution tests only. It does not show that preparation, scanner, or workflow effects exist.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: str) -> None:
    package_root = PACKAGE_DIR.resolve(strict=True)
    target = DEFAULT_REPORT
    if target.parent.resolve(strict=True) != package_root or target.name != (
        "identifiability_report.md"
    ):
        raise InputValidationError("report_write_target_forbidden")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".identifiability-report-", suffix=".tmp", dir=str(package_root)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
        os.replace(str(temporary), str(target))
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_fixture(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    headers: Sequence[str] = REQUIRED_COLUMNS,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(headers),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in headers})


def clone_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [dict(row) for row in rows]


def run_self_tests() -> dict[str, Any]:
    base_rows, _, base_headers = load_design(DEFAULT_INPUT, DEFAULT_REQUESTED_EFFECTS)
    passed: list[str] = []
    required_negative = (
        "preparation_perfectly_nested_in_scanner",
        "scanner_perfectly_nested_in_preparation",
        "site_perfectly_nested_in_scanner",
        "site_perfectly_nested_in_preparation",
        "disconnected_scanner_preparation_components",
        "biology_assigned_to_one_preparation",
        "single_biological_unit_carries_contrast",
        "duplicate_observation_id",
        "duplicate_physical_observation",
        "missing_preparation_batch",
        "missing_scan_batch",
        "missing_acquisition_order",
        "invalid_acquisition_order",
        "rank_deficient_fixed_effect_design",
        "interaction_without_cell_replication",
        "string_aliasing_of_factor_levels",
        "empty_factor_value",
        "invalid_replicate_identifier",
        "technical_replicate_mapping_conflict",
        "impossible_section_block_relationship",
        "missing_required_column",
        "one_level_requested_factor",
        "rank_engine_failure",
    )
    required_patch_cases = (
        "pairwise_complete_four_way_incomplete",
        "connected_three_of_four_preparation_scanner_main_effects",
        "nonbridging_biological_unit_is_diagnostic",
        "scanner_order_perfect_separation_is_reported",
        "fixed_scanner_workflow_sequence_is_reported",
        "preparation_order_not_counterbalanced_is_reported",
        "duplicate_physical_event_different_order_fails",
        "explicit_repeat_acquisition_is_allowed",
        "repeat_acquisition_does_not_inflate_biological_support",
        "preparation_scan_batch_alias_is_reported",
        "block_preparation_confounding_is_specific",
        "missing_interaction_does_not_override_main_effects",
    )

    temporary_root: Optional[Path] = None
    with tempfile.TemporaryDirectory(
        prefix="crossed-preparation-identifiability-tests-"
    ) as directory:
        temporary_root = Path(directory)
        fixture_index = 0

        def fixture_path(name: str) -> Path:
            nonlocal fixture_index
            fixture_index += 1
            return temporary_root / f"{fixture_index:02d}_{name}.csv"

        def expect_input_error(
            name: str,
            rows: Sequence[Mapping[str, str]],
            expected_code: str,
            headers: Sequence[str] = REQUIRED_COLUMNS,
        ) -> None:
            path = fixture_path(name)
            write_fixture(path, rows, headers)
            try:
                load_design(path, DEFAULT_REQUESTED_EFFECTS)
            except InputValidationError as exc:
                if exc.code != expected_code:
                    raise InputValidationError(
                        "self_test_wrong_error",
                        f"{name}|expected={expected_code}|observed={exc.code}",
                    ) from exc
            else:
                raise InputValidationError("self_test_expected_input_error", name)
            passed.append(name)

        def expect_finding(
            name: str,
            rows: Sequence[Mapping[str, str]],
            expected_code: str,
            requested_interactions: Sequence[str] = (),
        ) -> None:
            path = fixture_path(name)
            write_fixture(path, rows)
            loaded, digest, headers = load_design(path, DEFAULT_REQUESTED_EFFECTS)
            audit = analyze_design(
                loaded,
                digest,
                "<self-test-fixture>",
                headers,
                DEFAULT_REQUESTED_EFFECTS,
                requested_interactions,
            )
            codes = {item["code"] for item in audit["blocking_findings"]}
            if expected_code not in codes:
                raise InputValidationError(
                    "self_test_missing_finding",
                    f"{name}|expected={expected_code}|observed={'|'.join(sorted(codes))}",
                )
            passed.append(name)

        def audit_fixture(
            name: str,
            rows: Sequence[Mapping[str, str]],
            requested_interactions: Sequence[str] = (),
            headers: Sequence[str] = base_headers,
            requested_effects: Sequence[str] = DEFAULT_REQUESTED_EFFECTS,
        ) -> dict[str, Any]:
            path = fixture_path(name)
            write_fixture(path, rows, headers)
            loaded, digest, loaded_headers = load_design(path, requested_effects)
            return analyze_design(
                loaded,
                digest,
                "<self-test-fixture>",
                loaded_headers,
                requested_effects,
                requested_interactions,
            )

        valid_path = fixture_path("valid_fully_crossed_design")
        write_fixture(valid_path, base_rows)
        valid_rows, valid_digest, valid_headers = load_design(valid_path, DEFAULT_REQUESTED_EFFECTS)
        valid_audit = analyze_design(
            valid_rows,
            valid_digest,
            "<self-test-fixture>",
            valid_headers,
            DEFAULT_REQUESTED_EFFECTS,
            (),
        )
        if valid_audit["blocking_findings"] or (
            valid_audit["design_classification"]["primary"] != "fully crossed"
        ):
            raise InputValidationError(
                "self_test_valid_design_rejected",
                stable_json(valid_audit).strip(),
            )
        passed.append("valid_fully_crossed_design")

        higher_order_hole = [
            dict(row)
            for row in base_rows
            if row["observation_id"] not in {"OBS028", "OBS032"}
        ]
        higher_order_audit = audit_fixture(
            "pairwise_complete_four_way_incomplete",
            higher_order_hole,
        )
        if not all(
            pair["coverage"]["numerator"] == pair["coverage"]["denominator"]
            for pair in higher_order_audit["pairwise_crossing"].values()
        ):
            raise InputValidationError("self_test_pairwise_hole_unexpected")
        four_way = higher_order_audit["higher_order_crossing"][
            "biology_preparation_scanner_workflow"
        ]
        expected_missing = {
            "biological_unit": "BIO_02",
            "preparation_condition": "PREP_B",
            "scanner": "SCN_2",
            "site_workflow": "WF_POST_B",
        }
        if (
            four_way["observed_combinations"] != 15
            or four_way["possible_combinations"] != 16
            or four_way["missing_combinations"] != [expected_missing]
            or higher_order_audit["design_classification"]["primary"]
            != "pairwise complete, higher-order incomplete"
        ):
            raise InputValidationError(
                "self_test_higher_order_hole_misclassified",
                stable_json(higher_order_audit).strip(),
            )
        passed.append("pairwise_complete_four_way_incomplete")

        three_of_four = [
            dict(row)
            for row in base_rows
            if not (
                row["preparation_condition"] == "PREP_B"
                and row["scanner"] == "SCN_2"
            )
        ]
        three_of_four_audit = audit_fixture(
            "connected_three_of_four_preparation_scanner_main_effects",
            three_of_four,
        )
        prep_effect = three_of_four_audit["contrast_verdicts"][
            "biology_controlled_preparation_effect"
        ]
        scanner_effect = three_of_four_audit["contrast_verdicts"][
            "biology_preparation_controlled_scanner_effect"
        ]
        prep_scanner_interaction = three_of_four_audit["rank_summary"]["interactions"][
            "preparation_scanner"
        ]
        if (
            three_of_four_audit["overall_status"] != "identifiable"
            or three_of_four_audit["design_classification"]["primary"]
            != "partially crossed"
            or prep_effect["verdict"] != VERDICT_PARTIAL
            or scanner_effect["verdict"] != VERDICT_PARTIAL
            or prep_scanner_interaction["verdict"] != VERDICT_NOT
        ):
            raise InputValidationError(
                "self_test_connected_partial_design_misclassified",
                stable_json(three_of_four_audit).strip(),
            )
        passed.append("connected_three_of_four_preparation_scanner_main_effects")
        requested_interaction_audit = audit_fixture(
            "missing_interaction_does_not_override_main_effects",
            three_of_four,
            ("preparation_scanner",),
            base_headers,
            ("preparation",),
        )
        if (
            requested_interaction_audit["overall_status"] != "not_identifiable"
            or requested_interaction_audit["design_classification"]["primary"]
            == "fully crossed"
            or requested_interaction_audit["higher_order_crossing"][
                "requested_factor_product"
            ]["possible_combinations"]
            != 8
            or requested_interaction_audit["rank_summary"]["interactions"][
                "preparation_scanner"
            ]["verdict"]
            != VERDICT_NOT
        ):
            raise InputValidationError(
                "self_test_requested_interaction_crossing_universe_wrong",
                stable_json(requested_interaction_audit).strip(),
            )
        passed.append("missing_interaction_does_not_override_main_effects")

        nonbridging_rows: list[dict[str, str]] = []
        for index, source in enumerate(
            (
                row
                for row in base_rows
                if row["biological_unit"] == "BIO_02"
                and row["preparation_condition"] == "PREP_A"
            ),
            start=1,
        ):
            cloned = dict(source)
            cloned["observation_id"] = f"OBS3{index:02d}"
            cloned["biological_unit"] = "BIO_03"
            cloned["block_id"] = "BLK_03"
            cloned["section_id"] = cloned["section_id"].replace("SEC_02", "SEC_03")
            cloned["technical_replicate"] = cloned["technical_replicate"].replace(
                "TR_02", "TR_03"
            )
            cloned["biological_replicate"] = "BR_03"
            cloned["acquisition_order"] = str(int(cloned["acquisition_order"]) + 100)
            nonbridging_rows.append(cloned)
        nonbridging_audit = audit_fixture(
            "nonbridging_biological_unit_is_diagnostic",
            clone_rows(base_rows) + nonbridging_rows,
        )
        nonbridging_prep = nonbridging_audit["contrast_verdicts"][
            "biology_controlled_preparation_effect"
        ]
        if (
            nonbridging_audit["overall_status"] != "identifiable"
            or nonbridging_prep["verdict"] != VERDICT_PARTIAL
            or nonbridging_prep["contrasts"][0]["support"][
                "direct_biological_supporter_count"
            ]
            != 2
            or "BIO_03"
            in nonbridging_prep["contrasts"][0]["support"][
                "direct_biological_supporters"
            ]
        ):
            raise InputValidationError(
                "self_test_nonbridging_biology_invalidated_contrast",
                stable_json(nonbridging_audit).strip(),
            )
        passed.append("nonbridging_biological_unit_is_diagnostic")

        scanner_separated = clone_rows(base_rows)
        for batch in sorted({row["scan_batch"] for row in scanner_separated}):
            batch_rows = sorted(
                (row for row in scanner_separated if row["scan_batch"] == batch),
                key=lambda row: (row["scanner"], row["observation_id"]),
            )
            for order, row in enumerate(batch_rows, start=1):
                row["acquisition_order"] = str(order)
        scanner_order_audit = audit_fixture(
            "scanner_order_perfect_separation_is_reported",
            scanner_separated,
        )
        scanner_order = scanner_order_audit["randomization_and_blocking_metadata"][
            "acquisition_order_analysis"
        ]
        scanner_warning_codes = {item["code"] for item in scanner_order_audit["warnings"]}
        if (
            scanner_order["factor_assessments"]["scanner"]["status"]
            != "order-confounded"
            or "acquisition_order_perfectly_separated_by_factor"
            not in scanner_warning_codes
            or scanner_order_audit["contrast_verdicts"][
                "biology_preparation_controlled_scanner_effect"
            ]["operational_validity"]["status"]
            != "order-confounded"
        ):
            raise InputValidationError(
                "self_test_scanner_order_confounding_not_reported",
                stable_json(scanner_order_audit).strip(),
            )
        passed.append("scanner_order_perfect_separation_is_reported")

        fixed_sequence = clone_rows(base_rows)
        for row in fixed_sequence:
            row["acquisition_order"] = str(int(row["observation_id"].removeprefix("OBS")))
        fixed_order_audit = audit_fixture(
            "fixed_scanner_workflow_sequence_is_reported",
            fixed_sequence,
        )
        fixed_assessments = fixed_order_audit["randomization_and_blocking_metadata"][
            "acquisition_order_analysis"
        ]["factor_assessments"]
        if any(
            fixed_assessments[factor]["status"] != "order-confounded"
            for factor in ("preparation_condition", "scanner", "site_workflow")
        ):
            raise InputValidationError(
                "self_test_fixed_sequence_not_reported",
                stable_json(fixed_order_audit).strip(),
            )
        passed.append("fixed_scanner_workflow_sequence_is_reported")
        passed.append("preparation_order_not_counterbalanced_is_reported")

        duplicate_new_order = clone_rows(base_rows)
        duplicate = dict(duplicate_new_order[0])
        duplicate["observation_id"] = "OBS999"
        duplicate["acquisition_order"] = "999"
        duplicate_new_order.append(duplicate)
        expect_input_error(
            "duplicate_physical_event_different_order_fails",
            duplicate_new_order,
            "duplicate_physical_observation",
            base_headers,
        )

        explicit_repeat = clone_rows(base_rows)
        repeated = dict(explicit_repeat[0])
        repeated["observation_id"] = "OBS998"
        repeated["scan_batch"] = "SB_REPEAT"
        repeated["acquisition_order"] = "1"
        repeated["repeat_acquisition_id"] = "R2"
        explicit_repeat.append(repeated)
        repeat_audit = audit_fixture(
            "explicit_repeat_acquisition_is_allowed",
            explicit_repeat,
        )
        passed.append("explicit_repeat_acquisition_is_allowed")
        base_support = valid_audit["contrast_verdicts"][
            "biology_preparation_controlled_scanner_effect"
        ]["contrasts"][0]["support"]
        repeat_support = repeat_audit["contrast_verdicts"][
            "biology_preparation_controlled_scanner_effect"
        ]["contrasts"][0]["support"]
        if any(
            repeat_support[field] != base_support[field]
            for field in (
                "biological_supporter_count",
                "direct_biological_supporter_count",
                "supporting_section_count",
                "bridge_count",
            )
        ) or repeat_audit["design_summary"]["intentional_repeat_acquisition_rows"] != 1:
            raise InputValidationError(
                "self_test_repeat_inflated_biological_support",
                stable_json(repeat_support).strip(),
            )
        passed.append("repeat_acquisition_does_not_inflate_biological_support")

        batch_alias_rows = clone_rows(base_rows)
        for row in batch_alias_rows:
            row["scan_batch"] = "SB_1" if row["preparation_batch"] == "PB_1" else "SB_2"
        batch_alias_audit = audit_fixture(
            "preparation_scan_batch_alias_is_reported",
            batch_alias_rows,
        )
        if not batch_alias_audit["carrier_relationships"][
            "preparation_batch_vs_scan_batch"
        ]["one_to_one_alias"] or "preparation_batch_scan_batch_exact_alias" not in {
            item["code"] for item in batch_alias_audit["warnings"]
        }:
            raise InputValidationError(
                "self_test_batch_alias_not_reported",
                stable_json(batch_alias_audit).strip(),
            )
        passed.append("preparation_scan_batch_alias_is_reported")

        constant_batch_rows = clone_rows(base_rows)
        for row in constant_batch_rows:
            row["preparation_batch"] = "PB_CONSTANT"
            row["scan_batch"] = "SB_CONSTANT"
        constant_batch_audit = audit_fixture(
            "constant_batch_axes_and_ambiguous_serial_pairs",
            constant_batch_rows,
        )
        constant_relationship = constant_batch_audit["carrier_relationships"][
            "preparation_batch_vs_scan_batch"
        ]
        constant_prep_support = constant_batch_audit["contrast_verdicts"][
            "biology_controlled_preparation_effect"
        ]["contrasts"][0]["support"]
        if (
            constant_relationship["status"] != "constant axis"
            or "unavailable"
            not in constant_relationship["nuisance_adjustment_status"]
            or constant_prep_support["matched_serial_section_pair_count"] != 0
            or constant_prep_support[
                "ambiguous_matched_serial_section_strata_count"
            ]
            != 2
            or "matched_serial_section_pairing_ambiguous"
            not in {item["code"] for item in constant_batch_audit["warnings"]}
        ):
            raise InputValidationError(
                "self_test_constant_batch_or_serial_pairing_overstated",
                stable_json(constant_batch_audit).strip(),
            )

        block_order_rows = clone_rows(base_rows)
        for batch in sorted({row["scan_batch"] for row in block_order_rows}):
            batch_rows = sorted(
                (row for row in block_order_rows if row["scan_batch"] == batch),
                key=lambda row: (row["block_id"], row["observation_id"]),
            )
            for order, row in enumerate(batch_rows, start=1):
                row["acquisition_order"] = str(order)
        block_order_audit = audit_fixture(
            "block_order_perfect_separation_is_reported",
            block_order_rows,
        )
        if block_order_audit["randomization_and_blocking_metadata"][
            "acquisition_order_analysis"
        ]["factor_assessments"]["block_id"]["status"] != "order-confounded":
            raise InputValidationError(
                "self_test_block_order_confounding_not_reported",
                stable_json(block_order_audit).strip(),
            )

        block_preparation_rows = clone_rows(base_rows)
        for row in block_preparation_rows:
            row["block_id"] = (
                f"BLK_{row['biological_unit']}_{row['preparation_condition']}"
            )
        block_preparation_audit = audit_fixture(
            "block_preparation_confounding_is_specific",
            block_preparation_rows,
        )
        if "block_nested_in_preparation_condition" not in {
            item["code"] for item in block_preparation_audit["warnings"]
        }:
            raise InputValidationError(
                "self_test_block_preparation_confounding_not_specific",
                stable_json(block_preparation_audit).strip(),
            )
        passed.append("block_preparation_confounding_is_specific")

        partially_crossed_site = [
            dict(row)
            for row in base_rows
            if row["site_workflow"] == "WF_POST_A" or row["scanner"] == "SCN_1"
        ]
        partial_path = fixture_path("valid_partially_crossed_site_bridge_design")
        write_fixture(partial_path, partially_crossed_site)
        partial_rows, partial_digest, partial_headers = load_design(
            partial_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        partial_audit = analyze_design(
            partial_rows,
            partial_digest,
            "<self-test-fixture>",
            partial_headers,
            DEFAULT_REQUESTED_EFFECTS,
            (),
        )
        if partial_audit["blocking_findings"] or (
            partial_audit["design_classification"]["primary"] != "partially crossed"
        ):
            raise InputValidationError(
                "self_test_valid_partial_design_rejected",
                stable_json(partial_audit).strip(),
            )
        if (
            partial_audit["contrast_verdicts"][
                "biology_preparation_scanner_controlled_site_workflow_effect"
            ]["verdict"]
            != VERDICT_PARTIAL
        ):
            raise InputValidationError(
                "self_test_valid_partial_workflow_verdict_wrong",
                partial_audit["contrast_verdicts"][
                    "biology_preparation_scanner_controlled_site_workflow_effect"
                ]["verdict"],
            )
        passed.append("valid_partially_crossed_site_bridge_design")

        criss_cross = [
            dict(row)
            for row in base_rows
            if (
                row["section_id"].endswith("_1")
                and (
                    (row["scanner"] == "SCN_1" and row["site_workflow"] == "WF_POST_A")
                    or (row["scanner"] == "SCN_2" and row["site_workflow"] == "WF_POST_B")
                )
            )
            or (
                row["section_id"].endswith("_2")
                and (
                    (row["scanner"] == "SCN_1" and row["site_workflow"] == "WF_POST_B")
                    or (row["scanner"] == "SCN_2" and row["site_workflow"] == "WF_POST_A")
                )
            )
        ]
        criss_cross_path = fixture_path("scanner_workflow_criss_cross_is_assumption_dependent")
        write_fixture(criss_cross_path, criss_cross)
        criss_rows, criss_digest, criss_headers = load_design(
            criss_cross_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        criss_audit = analyze_design(
            criss_rows,
            criss_digest,
            "<self-test-fixture>",
            criss_headers,
            DEFAULT_REQUESTED_EFFECTS,
            (),
        )
        if criss_audit["contrast_verdicts"]["biology_preparation_controlled_scanner_effect"][
            "verdict"
        ] != VERDICT_ASSUMPTION or criss_audit["overall_status"] != ("assumption_dependent"):
            raise InputValidationError(
                "self_test_scanner_criss_cross_overstated",
                stable_json(criss_audit).strip(),
            )
        passed.append("scanner_workflow_criss_cross_is_assumption_dependent")

        narrow_effects = ("preparation", "site_workflow")
        one_scanner = [dict(row) for row in base_rows if row["scanner"] == "SCN_1"]
        narrow_path = fixture_path("one_level_unrequested_scanner_is_not_confounding")
        write_fixture(narrow_path, one_scanner)
        narrow_rows, narrow_digest, narrow_headers = load_design(narrow_path, narrow_effects)
        narrow_audit = analyze_design(
            narrow_rows,
            narrow_digest,
            "<self-test-fixture>",
            narrow_headers,
            narrow_effects,
            (),
        )
        if narrow_audit["blocking_findings"] or narrow_audit["overall_status"] != "identifiable":
            raise InputValidationError(
                "self_test_unrequested_constant_factor_rejected",
                stable_json(narrow_audit).strip(),
            )
        for future_name in (
            "scanner_suppressed_residual_association_with_preparation",
            "scanner_suppressed_residual_association_with_site_workflow",
        ):
            if narrow_audit["contrast_verdicts"][future_name]["verdict"] != VERDICT_NOT:
                raise InputValidationError(
                    "self_test_future_scanner_support_overstated",
                    future_name,
                )
        passed.append("one_level_unrequested_scanner_is_not_confounding")

        preparation_only = ("preparation",)
        unrequested_alias = [
            dict(row)
            for row in base_rows
            if (row["scanner"] == "SCN_1" and row["site_workflow"] == "WF_POST_A")
            or (row["scanner"] == "SCN_2" and row["site_workflow"] == "WF_POST_B")
        ]
        unrequested_alias_path = fixture_path("unrequested_factor_alias_is_diagnostic_only")
        write_fixture(unrequested_alias_path, unrequested_alias)
        alias_rows, alias_digest, alias_headers = load_design(
            unrequested_alias_path,
            preparation_only,
        )
        alias_audit = analyze_design(
            alias_rows,
            alias_digest,
            "<self-test-fixture>",
            alias_headers,
            preparation_only,
            (),
        )
        if alias_audit["blocking_findings"] or alias_audit["overall_status"] != "identifiable":
            raise InputValidationError(
                "self_test_unrequested_alias_blocked_requested_effect",
                stable_json(alias_audit).strip(),
            )
        passed.append("unrequested_factor_alias_is_diagnostic_only")

        third_biology: list[dict[str, str]] = []
        for index, source in enumerate(
            (row for row in base_rows if row["biological_unit"] == "BIO_02"),
            start=1,
        ):
            cloned = dict(source)
            cloned["observation_id"] = f"OBS3{index:02d}"
            cloned["biological_unit"] = "BIO_03"
            cloned["block_id"] = "BLK_03"
            cloned["section_id"] = cloned["section_id"].replace("SEC_02", "SEC_03")
            cloned["technical_replicate"] = cloned["technical_replicate"].replace("TR_02", "TR_03")
            cloned["biological_replicate"] = "BR_03"
            cloned["acquisition_order"] = str(int(cloned["acquisition_order"]) + 100)
            third_biology.append(cloned)
        three_biology = clone_rows(base_rows) + third_biology
        allowed_cells = {
            "BIO_01": {("PREP_A", "SCN_1"), ("PREP_A", "SCN_2")},
            "BIO_02": {
                ("PREP_A", "SCN_1"),
                ("PREP_B", "SCN_1"),
                ("PREP_B", "SCN_2"),
            },
            "BIO_03": {
                ("PREP_A", "SCN_2"),
                ("PREP_B", "SCN_1"),
                ("PREP_B", "SCN_2"),
            },
        }
        network_rectangle = [
            row
            for row in three_biology
            if (row["preparation_condition"], row["scanner"])
            in allowed_cells[row["biological_unit"]]
        ]
        network_path = fixture_path("interaction_without_within_biology_rectangles")
        write_fixture(network_path, network_rectangle)
        network_rows, network_digest, network_headers = load_design(
            network_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        network_audit = analyze_design(
            network_rows,
            network_digest,
            "<self-test-fixture>",
            network_headers,
            DEFAULT_REQUESTED_EFFECTS,
            ("preparation_scanner",),
        )
        interaction = network_audit["rank_summary"]["interactions"]["preparation_scanner"]
        if (
            interaction["verdict"] != VERDICT_ASSUMPTION
            or interaction["minimum_direct_biological_supporters_per_contrast"] != 0
        ):
            raise InputValidationError(
                "self_test_interaction_rectangle_support_overstated",
                stable_json(interaction).strip(),
            )
        passed.append("interaction_without_within_biology_rectangles")

        fourth_biology: list[dict[str, str]] = []
        for index, source in enumerate(third_biology, start=1):
            cloned = dict(source)
            cloned["observation_id"] = f"OBS4{index:02d}"
            cloned["biological_unit"] = "BIO_04"
            cloned["block_id"] = "BLK_04"
            cloned["section_id"] = cloned["section_id"].replace("SEC_03", "SEC_04")
            cloned["technical_replicate"] = cloned["technical_replicate"].replace("TR_03", "TR_04")
            cloned["biological_replicate"] = "BR_04"
            cloned["acquisition_order"] = str(int(cloned["acquisition_order"]) + 100)
            fourth_biology.append(cloned)
        interaction_carrier_rows = clone_rows(base_rows) + third_biology + fourth_biology
        partial_cells = {
            "BIO_03": {
                ("PREP_A", "SCN_1"),
                ("PREP_A", "SCN_2"),
                ("PREP_B", "SCN_1"),
            },
            "BIO_04": {
                ("PREP_A", "SCN_2"),
                ("PREP_B", "SCN_1"),
                ("PREP_B", "SCN_2"),
            },
        }
        interaction_carrier_rows = [
            row
            for row in interaction_carrier_rows
            if row["biological_unit"] in {"BIO_01", "BIO_02"}
            or (row["preparation_condition"], row["scanner"])
            in partial_cells[row["biological_unit"]]
        ]
        for row in interaction_carrier_rows:
            first_carrier = row["biological_unit"] in {"BIO_01", "BIO_02"}
            row["preparation_batch"] = "PB_RECT_1" if first_carrier else "PB_RECT_2"
            row["scan_batch"] = "SB_RECT_1" if first_carrier else "SB_RECT_2"
            row["fold_id"] = "FOLD_1" if first_carrier else "FOLD_2"
        carrier_headers = REQUIRED_COLUMNS + ("fold_id",)
        carrier_path = fixture_path("interaction_carrier_strata_are_counted_per_rectangle")
        write_fixture(carrier_path, interaction_carrier_rows, carrier_headers)
        carrier_rows, carrier_digest, loaded_carrier_headers = load_design(
            carrier_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        carrier_audit = analyze_design(
            carrier_rows,
            carrier_digest,
            "<self-test-fixture>",
            loaded_carrier_headers,
            DEFAULT_REQUESTED_EFFECTS,
            ("preparation_scanner",),
        )
        carrier_interaction = carrier_audit["rank_summary"]["interactions"]["preparation_scanner"]
        carrier_support = carrier_interaction["difference_in_differences_contrasts"][0][
            "direct_support"
        ]
        expected_carriers = {
            "supporting_preparation_batches": ["PB_RECT_1"],
            "supporting_scan_batches": ["SB_RECT_1"],
            "supporting_folds": ["FOLD_1"],
        }
        if carrier_interaction["verdict"] != VERDICT_DIRECT or any(
            carrier_support[field] != expected for field, expected in expected_carriers.items()
        ):
            raise InputValidationError(
                "self_test_interaction_carrier_count_wrong",
                stable_json(carrier_interaction).strip(),
            )
        carrier_warning_codes = {item["code"] for item in carrier_audit["warnings"]}
        required_carrier_warnings = {
            "interaction_contrast_supported_by_fewer_than_two_preparation_batches",
            "interaction_contrast_supported_by_fewer_than_two_scan_batches",
            "interaction_contrast_supported_by_fewer_than_two_folds",
        }
        if not required_carrier_warnings.issubset(carrier_warning_codes):
            raise InputValidationError(
                "self_test_interaction_carrier_warning_missing",
                "|".join(sorted(required_carrier_warnings - carrier_warning_codes)),
            )
        passed.append("interaction_carrier_strata_are_counted_per_rectangle")

        preparation_batch_alias = clone_rows(base_rows)
        for row in preparation_batch_alias:
            row["preparation_batch"] = f"PB_{row['preparation_condition']}"
        batch_path = fixture_path("preparation_batch_alias_is_assumption_dependent")
        write_fixture(batch_path, preparation_batch_alias)
        batch_rows, batch_digest, batch_headers = load_design(
            batch_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        batch_audit = analyze_design(
            batch_rows,
            batch_digest,
            "<self-test-fixture>",
            batch_headers,
            DEFAULT_REQUESTED_EFFECTS,
            ("preparation_scanner",),
        )
        if (
            batch_audit["contrast_verdicts"]["biology_controlled_preparation_effect"]["verdict"]
            != VERDICT_ASSUMPTION
        ):
            raise InputValidationError(
                "self_test_preparation_batch_alias_overstated",
                stable_json(batch_audit).strip(),
            )
        if (
            batch_audit["rank_summary"]["interactions"]["preparation_scanner"]["verdict"]
            != VERDICT_ASSUMPTION
        ):
            raise InputValidationError(
                "self_test_preparation_batch_interaction_overstated",
                stable_json(batch_audit).strip(),
            )
        passed.append("preparation_batch_alias_is_assumption_dependent")

        replicated_nested_batches = clone_rows(base_rows)
        for row in replicated_nested_batches:
            row["preparation_batch"] = f"PB_{row['preparation_condition']}_{row['biological_unit']}"
        nested_batch_path = fixture_path("replicated_nested_batches_are_assumption_dependent")
        write_fixture(nested_batch_path, replicated_nested_batches)
        nested_rows, nested_digest, nested_headers = load_design(
            nested_batch_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        nested_batch_audit = analyze_design(
            nested_rows,
            nested_digest,
            "<self-test-fixture>",
            nested_headers,
            DEFAULT_REQUESTED_EFFECTS,
            (),
        )
        if (
            nested_batch_audit["contrast_verdicts"]["biology_controlled_preparation_effect"][
                "verdict"
            ]
            != VERDICT_ASSUMPTION
        ):
            raise InputValidationError(
                "self_test_replicated_nested_batches_overstated",
                stable_json(nested_batch_audit).strip(),
            )
        passed.append("replicated_nested_batches_are_assumption_dependent")

        scan_batch_alias = clone_rows(base_rows)
        for row in scan_batch_alias:
            row["scan_batch"] = f"SB_{row['scanner']}"
        scan_batch_path = fixture_path("scan_batch_alias_is_assumption_dependent")
        write_fixture(scan_batch_path, scan_batch_alias)
        scan_rows, scan_digest, scan_headers = load_design(
            scan_batch_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        scan_audit = analyze_design(
            scan_rows,
            scan_digest,
            "<self-test-fixture>",
            scan_headers,
            DEFAULT_REQUESTED_EFFECTS,
            ("scanner_site_workflow",),
        )
        if (
            scan_audit["contrast_verdicts"]["biology_preparation_controlled_scanner_effect"][
                "verdict"
            ]
            != VERDICT_ASSUMPTION
        ):
            raise InputValidationError(
                "self_test_scan_batch_alias_overstated",
                stable_json(scan_audit).strip(),
            )
        if (
            scan_audit["rank_summary"]["interactions"]["scanner_site_workflow"]["verdict"]
            != VERDICT_ASSUMPTION
        ):
            raise InputValidationError(
                "self_test_scan_batch_interaction_overstated",
                stable_json(scan_audit).strip(),
            )
        passed.append("scan_batch_alias_is_assumption_dependent")

        workflow_batch_alias = clone_rows(base_rows)
        for row in workflow_batch_alias:
            row["scan_batch"] = f"SB_{row['site_workflow']}"
        workflow_batch_path = fixture_path("workflow_batch_alias_is_assumption_dependent")
        write_fixture(workflow_batch_path, workflow_batch_alias)
        workflow_rows, workflow_digest, workflow_headers = load_design(
            workflow_batch_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        workflow_audit = analyze_design(
            workflow_rows,
            workflow_digest,
            "<self-test-fixture>",
            workflow_headers,
            DEFAULT_REQUESTED_EFFECTS,
            ("scanner_site_workflow",),
        )
        workflow_main = workflow_audit["contrast_verdicts"][
            "biology_preparation_scanner_controlled_site_workflow_effect"
        ]["verdict"]
        workflow_interaction = workflow_audit["rank_summary"]["interactions"][
            "scanner_site_workflow"
        ]["verdict"]
        if workflow_main != VERDICT_ASSUMPTION or workflow_interaction != VERDICT_ASSUMPTION:
            raise InputValidationError(
                "self_test_workflow_batch_alias_overstated",
                stable_json(workflow_audit).strip(),
            )
        passed.append("workflow_batch_alias_is_assumption_dependent")

        fold_rows = clone_rows(base_rows)
        for row in fold_rows:
            row["fold_id"] = "FOLD_1"
        fold_headers = REQUIRED_COLUMNS + ("fold_id",)
        fold_path = fixture_path("single_fold_support_is_flagged")
        write_fixture(fold_path, fold_rows, fold_headers)
        loaded_folds, fold_digest, loaded_fold_headers = load_design(
            fold_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        fold_audit = analyze_design(
            loaded_folds,
            fold_digest,
            "<self-test-fixture>",
            loaded_fold_headers,
            DEFAULT_REQUESTED_EFFECTS,
            (),
        )
        fold_warning_codes = {item["code"] for item in fold_audit["warnings"]}
        if "contrast_supported_by_fewer_than_two_folds" not in fold_warning_codes:
            raise InputValidationError("self_test_single_fold_support_not_flagged")
        passed.append("single_fold_support_is_flagged")

        fold_confounded_rows = clone_rows(base_rows)
        for row in fold_confounded_rows:
            row["fold_id"] = f"FOLD_{row['preparation_condition']}"
        fold_confounded_path = fixture_path("fold_must_carry_both_contrast_levels")
        write_fixture(fold_confounded_path, fold_confounded_rows, fold_headers)
        fold_confounded, fold_confounded_digest, fold_confounded_headers = load_design(
            fold_confounded_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        fold_confounded_audit = analyze_design(
            fold_confounded,
            fold_confounded_digest,
            "<self-test-fixture>",
            fold_confounded_headers,
            DEFAULT_REQUESTED_EFFECTS,
            (),
        )
        preparation_support = fold_confounded_audit["contrast_verdicts"][
            "biology_controlled_preparation_effect"
        ]["contrasts"][0]["support"]
        if preparation_support["supporting_folds"]:
            raise InputValidationError(
                "self_test_fold_union_miscounted_as_contrast_support",
                "|".join(preparation_support["supporting_folds"]),
            )
        passed.append("fold_must_carry_both_contrast_levels")

        preparation_blocks = clone_rows(base_rows)
        for row in preparation_blocks:
            row["block_id"] = f"BLK_{row['biological_unit']}_{row['preparation_condition']}"
        block_path = fixture_path("interaction_without_within_block_rectangle")
        write_fixture(block_path, preparation_blocks)
        block_rows, block_digest, block_headers = load_design(
            block_path,
            DEFAULT_REQUESTED_EFFECTS,
        )
        block_audit = analyze_design(
            block_rows,
            block_digest,
            "<self-test-fixture>",
            block_headers,
            DEFAULT_REQUESTED_EFFECTS,
            ("preparation_scanner",),
        )
        if (
            block_audit["rank_summary"]["interactions"]["preparation_scanner"]["verdict"]
            != VERDICT_ASSUMPTION
        ):
            raise InputValidationError(
                "self_test_interaction_block_assumption_overstated",
                stable_json(block_audit).strip(),
            )
        block_support = block_audit["contrast_verdicts"]["biology_controlled_preparation_effect"][
            "contrasts"
        ][0]["support"]
        if block_support["supporting_blocks"]:
            raise InputValidationError(
                "self_test_block_union_miscounted_as_contrast_support",
                "|".join(block_support["supporting_blocks"]),
            )
        passed.append("interaction_without_within_block_rectangle")

        unrelated = dict(base_rows[0])
        unrelated.update(
            {
                "observation_id": "OBS999",
                "block_id": "BLK_EXTRA",
                "section_id": "SEC_EXTRA",
                "preparation_batch": "PB_EXTRA",
                "acquisition_order": "999",
                "technical_replicate": "TR_EXTRA",
            }
        )
        support = effect_support(
            clone_rows(base_rows) + [unrelated],
            "preparation_condition",
            "PREP_A",
            "PREP_B",
        )
        if (
            "BLK_EXTRA" in support["supporting_blocks"]
            or "PB_EXTRA" in support["supporting_preparation_batches"]
        ):
            raise InputValidationError("self_test_unrelated_rows_counted_as_support")
        passed.append("unrelated_rows_not_counted_as_contrast_support")

        preparation_scanner_alias = [
            dict(row)
            for row in base_rows
            if (row["preparation_condition"] == "PREP_A" and row["scanner"] == "SCN_1")
            or (row["preparation_condition"] == "PREP_B" and row["scanner"] == "SCN_2")
        ]
        expect_finding(
            "preparation_perfectly_nested_in_scanner",
            preparation_scanner_alias,
            "preparation_condition_nested_in_scanner",
        )
        expect_finding(
            "scanner_perfectly_nested_in_preparation",
            preparation_scanner_alias,
            "scanner_nested_in_preparation_condition",
        )

        site_scanner_alias = [
            dict(row)
            for row in base_rows
            if (row["site_workflow"] == "WF_POST_A" and row["scanner"] == "SCN_1")
            or (row["site_workflow"] == "WF_POST_B" and row["scanner"] == "SCN_2")
        ]
        expect_finding(
            "site_perfectly_nested_in_scanner",
            site_scanner_alias,
            "site_workflow_nested_in_scanner",
        )

        site_preparation_alias = [
            dict(row)
            for row in base_rows
            if (row["site_workflow"] == "WF_POST_A" and row["preparation_condition"] == "PREP_A")
            or (row["site_workflow"] == "WF_POST_B" and row["preparation_condition"] == "PREP_B")
        ]
        expect_finding(
            "site_perfectly_nested_in_preparation",
            site_preparation_alias,
            "site_workflow_nested_in_preparation_condition",
        )

        disconnected = [
            dict(row)
            for row in base_rows
            if (
                row["biological_unit"] == "BIO_01"
                and row["preparation_condition"] == "PREP_A"
                and row["scanner"] == "SCN_1"
                and row["site_workflow"] == "WF_POST_A"
            )
            or (
                row["biological_unit"] == "BIO_02"
                and row["preparation_condition"] == "PREP_B"
                and row["scanner"] == "SCN_2"
                and row["site_workflow"] == "WF_POST_B"
            )
        ]
        expect_finding(
            "disconnected_scanner_preparation_components",
            disconnected,
            "disconnected_factor_graph",
        )

        biology_one_preparation = [
            dict(row)
            for row in base_rows
            if (row["biological_unit"] == "BIO_01" and row["preparation_condition"] == "PREP_A")
            or (row["biological_unit"] == "BIO_02" and row["preparation_condition"] == "PREP_B")
        ]
        expect_finding(
            "biology_assigned_to_one_preparation",
            biology_one_preparation,
            "biology_assigned_to_only_one_preparation",
        )

        single_biology = [
            dict(row)
            for row in base_rows
            if not (row["biological_unit"] == "BIO_02" and row["preparation_condition"] == "PREP_B")
        ]
        expect_finding(
            "single_biological_unit_carries_contrast",
            single_biology,
            "preparation_contrast_fewer_than_two_biological_units",
        )

        duplicate_id = clone_rows(base_rows)
        duplicate_id[1]["observation_id"] = duplicate_id[0]["observation_id"]
        expect_input_error(
            "duplicate_observation_id",
            duplicate_id,
            "duplicate_observation_id",
        )

        duplicate_physical = clone_rows(base_rows)
        copied = dict(duplicate_physical[0])
        copied["observation_id"] = "OBS999"
        duplicate_physical.append(copied)
        expect_input_error(
            "duplicate_physical_observation",
            duplicate_physical,
            "duplicate_physical_observation",
        )

        missing_preparation_batch = clone_rows(base_rows)
        missing_preparation_batch[0]["preparation_batch"] = ""
        expect_input_error(
            "missing_preparation_batch",
            missing_preparation_batch,
            "empty_required_value:preparation_batch",
        )

        missing_scan_batch = clone_rows(base_rows)
        missing_scan_batch[0]["scan_batch"] = ""
        expect_input_error(
            "missing_scan_batch",
            missing_scan_batch,
            "empty_required_value:scan_batch",
        )

        missing_order = clone_rows(base_rows)
        missing_order[0]["acquisition_order"] = ""
        expect_input_error(
            "missing_acquisition_order",
            missing_order,
            "empty_required_value:acquisition_order",
        )

        invalid_order = clone_rows(base_rows)
        invalid_order[0]["acquisition_order"] = "0"
        expect_input_error(
            "invalid_acquisition_order",
            invalid_order,
            "invalid_acquisition_order",
        )

        expect_finding(
            "rank_deficient_fixed_effect_design",
            preparation_scanner_alias,
            "rank_deficient_fixed_effect_design",
        )

        interaction_single_biology = [
            dict(row)
            for row in base_rows
            if not (
                row["biological_unit"] == "BIO_02"
                and row["preparation_condition"] == "PREP_B"
                and row["scanner"] == "SCN_2"
            )
        ]
        expect_finding(
            "interaction_without_cell_replication",
            interaction_single_biology,
            "interaction_insufficient_biological_replication",
            ("preparation_scanner",),
        )

        aliased_strings = clone_rows(base_rows)
        aliased_strings[0]["preparation_condition"] = "prep-a"
        expect_input_error(
            "string_aliasing_of_factor_levels",
            aliased_strings,
            "string_aliasing:preparation_condition",
        )

        empty_factor = clone_rows(base_rows)
        empty_factor[0]["scanner"] = ""
        expect_input_error(
            "empty_factor_value",
            empty_factor,
            "empty_required_value:scanner",
        )

        invalid_replicate = clone_rows(base_rows)
        invalid_replicate[0]["technical_replicate"] = "bad id!"
        expect_input_error(
            "invalid_replicate_identifier",
            invalid_replicate,
            "invalid_identifier:technical_replicate",
        )

        replicate_mapping = clone_rows(base_rows)
        replicate_mapping[4]["technical_replicate"] = replicate_mapping[0]["technical_replicate"]
        expect_input_error(
            "technical_replicate_mapping_conflict",
            replicate_mapping,
            "technical_replicate_mapping_conflict",
        )

        impossible_section = clone_rows(base_rows)
        impossible_section[0]["block_id"] = "BLK_99"
        expect_input_error(
            "impossible_section_block_relationship",
            impossible_section,
            "impossible_section_block_relationship",
        )

        missing_column_headers = tuple(field for field in REQUIRED_COLUMNS if field != "scan_batch")
        expect_input_error(
            "missing_required_column",
            base_rows,
            "missing_required_columns",
            missing_column_headers,
        )

        one_preparation_level = [
            dict(row) for row in base_rows if row["preparation_condition"] == "PREP_A"
        ]
        expect_input_error(
            "one_level_requested_factor",
            one_preparation_level,
            "factor_too_few_levels:preparation_condition",
        )

        try:
            matrix_rank([[1, 0], [1]])
        except RankCalculationError as exc:
            if exc.code != "rank_matrix_is_ragged":
                raise InputValidationError("self_test_wrong_rank_error", exc.code) from exc
        else:
            raise InputValidationError("self_test_rank_failure_not_detected")
        passed.append("rank_engine_failure")

    fixtures_removed = temporary_root is not None and not temporary_root.exists()
    missing = sorted(set(required_negative) - set(passed))
    if missing:
        raise InputValidationError("self_test_required_cases_missing", "|".join(missing))
    missing_patch_cases = sorted(set(required_patch_cases) - set(passed))
    if missing_patch_cases:
        raise InputValidationError(
            "self_test_required_patch_cases_missing",
            "|".join(missing_patch_cases),
        )
    if not fixtures_removed:
        raise InputValidationError("self_test_temporary_fixture_cleanup_failed")
    expected_total = len(required_negative) + 15 + len(required_patch_cases)
    if len(passed) != expected_total:
        raise InputValidationError(
            "self_test_count_mismatch",
            f"expected={expected_total}|observed={len(passed)}",
        )
    return {
        "status": "passed",
        "passed": len(passed),
        "total": expected_total,
        "core_regression_passed": len(required_negative) + 15,
        "core_regression_total": len(required_negative) + 15,
        "required_negative_passed": len(required_negative),
        "required_negative_total": len(required_negative),
        "required_patch_cases_passed": len(required_patch_cases),
        "required_patch_cases_total": len(required_patch_cases),
        "temporary_fixtures_removed": fixtures_removed,
        "tests": passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit structural identifiability of crossed preparation, scanner, "
            "and workflow designs without model training."
        )
    )
    parser.add_argument("--input", type=Path, help="Custom sampling-matrix CSV")
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        help="Print deterministic output without writing the checked report",
    )
    parser.add_argument(
        "--check-report",
        action="store_true",
        help="Compare deterministic report bytes and run self-tests",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run temporary-fixture fail-closed tests only",
    )
    parser.add_argument(
        "--requested-effects",
        default=",".join(DEFAULT_REQUESTED_EFFECTS),
        help=("Comma-separated requested effects from preparation, scanner, site_workflow"),
    )
    parser.add_argument(
        "--request-interaction",
        action="append",
        choices=tuple(INTERACTION_SPECS),
        default=[],
        help="Make a diagnostically audited interaction a blocking requirement",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        requested_effects = parse_requested_effects(args.requested_effects)
        if args.self_test:
            if (
                args.input
                or args.format
                or args.check_report
                or args.request_interaction
                or requested_effects != DEFAULT_REQUESTED_EFFECTS
            ):
                raise InputValidationError("self_test_option_conflict")
            tests = run_self_tests()
            print(
                "IDENTIFIABILITY_SELF_TEST_PASS "
                f"passed={tests['passed']}/{tests['total']} "
                f"core_regression={tests['core_regression_passed']}/"
                f"{tests['core_regression_total']} "
                f"patch_regression={tests['required_patch_cases_passed']}/"
                f"{tests['required_patch_cases_total']} "
                f"required_negative={tests['required_negative_passed']}/"
                f"{tests['required_negative_total']} "
                f"temporary_fixtures_removed={str(tests['temporary_fixtures_removed']).lower()}"
            )
            return 0

        requested_interactions = tuple(
            name for name in INTERACTION_SPECS if name in set(args.request_interaction)
        )
        if args.check_report and (
            args.input is not None
            or args.format is not None
            or requested_effects != DEFAULT_REQUESTED_EFFECTS
            or requested_interactions
        ):
            raise InputValidationError("check_report_option_conflict")

        input_path = args.input if args.input is not None else DEFAULT_INPUT
        input_label = "example_design_matrix.csv" if args.input is None else "<custom-input>"
        rows, input_digest, headers = load_design(input_path, requested_effects)
        audit = analyze_design(
            rows,
            input_digest,
            input_label,
            headers,
            requested_effects,
            requested_interactions,
        )
        tests = run_self_tests()
        finalized = finalize_audit(audit, tests)
        report = render_report(finalized)

        if args.check_report:
            try:
                current = DEFAULT_REPORT.read_bytes()
            except OSError as exc:
                print(
                    f"IDENTIFIABILITY_REPORT_CHECK_FAIL code=report_unreadable detail={exc}",
                    file=sys.stderr,
                )
                return 1
            expected = report.encode("utf-8")
            if current != expected:
                print(
                    "IDENTIFIABILITY_REPORT_CHECK_FAIL code=report_bytes_mismatch",
                    file=sys.stderr,
                )
                return 1
            if finalized["overall_status"] != "identifiable":
                print(
                    "IDENTIFIABILITY_REPORT_CHECK_FAIL code=default_design_not_identifiable",
                    file=sys.stderr,
                )
                return 1
            print(
                "IDENTIFIABILITY_REPORT_CHECK_PASS "
                f"observations={finalized['design_summary']['observations']} "
                f"fingerprint={finalized['audit_fingerprint_sha256']} "
                f"self_tests={tests['passed']}/{tests['total']} "
                f"core_regression={tests['core_regression_passed']}/"
                f"{tests['core_regression_total']} "
                f"patch_regression={tests['required_patch_cases_passed']}/"
                f"{tests['required_patch_cases_total']}"
            )
            return 0

        if args.format == "json":
            print(stable_json(finalized), end="")
        elif args.format == "markdown" or args.input is not None:
            print(report, end="")
        else:
            write_report(report)
            print(
                "IDENTIFIABILITY_REPORT_WRITTEN "
                f"observations={finalized['design_summary']['observations']} "
                f"fingerprint={finalized['audit_fingerprint_sha256']}"
            )
        return 0 if finalized["overall_status"] == "identifiable" else 1
    except InputValidationError as exc:
        suffix = f" detail={exc.detail}" if exc.detail else ""
        print(
            f"IDENTIFIABILITY_INPUT_ERROR code={exc.code}{suffix}",
            file=sys.stderr,
        )
        return 2
    except RankCalculationError as exc:
        suffix = f" detail={exc.detail}" if exc.detail else ""
        print(
            f"IDENTIFIABILITY_RANK_ERROR code={exc.code}{suffix}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
