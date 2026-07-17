# Minimum prospective acquisition plan

When no existing dataset reaches confirmatory metadata readiness, collect a study designed around identifiability rather than attempting to repair provenance after acquisition.

## Biological material

Record stable identifiers for:

- biological unit;
- block or equivalent material anchor;
- section;
- matched serial-section relationship;
- section order and, where meaningful, section distance.

Preparation comparisons must use a physically valid anchor. Matched serial sections are not identical tissue or pixels.

## Preparation

Use at least two preparation conditions when the preparation contrast is requested. Record:

- preparation-condition identifier;
- preparation batch;
- protocol version;
- operator or operator pool when relevant;
- timing windows;
- deviations and exclusions.

Preparation must not be permanently nested within scanner, site, or batch.

## Scanner acquisition

Acquire every preparation level on multiple scanners or defensible scanner conditions. Record:

- scanner device ID;
- scanner model;
- scan batch;
- acquisition order;
- immutable source-event ID;
- file checksum;
- rescan/repeat identifier;
- quality-control outcome and exclusion reason.

Where the scanner contrast is requested, preserve explicit same-section paired scanning.

## Post-preparation workflow

Define the workflow prospectively rather than using site as a proxy. Record:

- workflow identifier;
- operator or operator pool;
- transfer/storage condition;
- handling and post-processing steps;
- exposure order;
- timing window;
- destructive/non-destructive status;
- carryover risk;
- physical bridge supporting the comparison.

A site label alone is insufficient.

## Counterbalancing and batching

Use randomized or counterbalanced acquisition order where physically feasible. Ensure preparation batches and scan batches are not one-to-one aliases. Record enough observations across batches to diagnose batch nesting and order confounding.

## Provenance

For every source acquisition, retain:

- immutable source-event ID;
- checksum;
- original file path or archive identifier;
- parent biological/block/section identifiers;
- preparation condition and batch;
- scanner device and scan batch;
- workflow exposure and order;
- QC/exclusion record.

## Minimum decision gate before analysis

Do not begin representation analysis until the executed sampling matrix passes the crossed-preparation identifiability audit and the metadata registry passes this readiness audit for the intended contrast.

This plan is a structural floor, not a power calculation or universal sample-size recommendation.
