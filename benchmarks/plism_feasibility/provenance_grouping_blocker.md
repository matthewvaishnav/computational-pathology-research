# PLISM registered-group provenance blocker

## Status

The metadata-only Figshare manifest remains usable. The executed crossing matrix is **blocked** until a provenance-bearing grouping key is recovered and validated.

The current public five-column image list exposes tissue category, stain, scanner/device label, coordinate, and image path. None of those fields, alone or in combination, is established as a unique physical-slide or specimen identifier. In particular, `tissue_type + coordinate` is unsafe because coordinates are local to an image/slide and tissue category is reused across many slides.

## Required hierarchy

Any executable crossing audit must preserve, when available:

1. specimen or source-block identifier;
2. physical slide / serial-section identifier;
3. registered field or tile-group identifier;
4. stain condition;
5. scanner domain;
6. image path.

A `registered_group_id` may be created only from a field whose semantics are documented by the source deposit or independently verified against archive structure. Filename coincidence is not sufficient.

## Fail-closed acceptance criteria

The crossing script must refuse to produce completeness claims unless all criteria below pass:

- a provenance-bearing slide, section, archive-group, or verified tile identifier is supplied;
- the identifier is stable across scanner rescans of the same stained section;
- serial sections with different stains remain distinguishable at the slide/section level;
- two different slides can share the same tissue label and coordinate without colliding;
- duplicate group–stain–scanner identities remain rejected;
- train/test grouping uses the highest available biological/provenance unit, not tile coordinates;
- the generated report records the grouping field and its source.

## Mandatory adversarial test

A deterministic fixture must include two distinct slides with the same tissue label and coordinate:

```csv
Slide ID,Tissue Type,Stain Type,Device Type,Coordinate,Image Path
slide_A,Liver,GV,S1,1000_500,slide_A/GV_S1/GV_S1_1000_500.png
slide_B,Liver,GV,S1,1000_500,slide_B/GV_S1/GV_S1_1000_500.png
```

The normalized output must contain two distinct registered groups. Any implementation that merges these rows is invalid.

## Safe implementation direction

Prefer an explicit required CLI option such as `--group-column "Slide ID"` or a separately generated provenance map. Do not silently infer a slide identifier from tissue category, coordinate, scanner, or stain. If the public archive cannot provide a validated grouping field, retain the package as a metadata inventory only and do not report registered-group completeness.

## Claim boundary

Until this blocker is resolved, no generated PLISM result may claim a complete stain × scanner crossing, a count of independent registered groups, or leakage-safe grouped splits. The global stain × scanner inventory can still be reported descriptively because it does not require asserting biological correspondence.