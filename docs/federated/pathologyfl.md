# PathologyFL

PathologyFL is the custom federated learning framework for computational pathology in this repository.

Goals:

- privacy-preserving multi-site training
- pathology-specific federated workflows
- secure aggregation and differential privacy support
- robustness to client heterogeneity and site imbalance
- support for FAIR-WEIGHTS-H institutional weighting

## Coordinator components

Typical coordinator services include:

- orchestrator
- aggregator
- client registry
- privacy engine
- monitoring system
- byzantine detection

## Client-side workflow

```text
Local pathology data
    -> local training
    -> update serialization
    -> secure communication
    -> aggregation
```

## Validation status

Core federated integration tests pass and smoke tests execute on PCam-derived pathology data.

This validates end-to-end execution but not yet real multi-center clinical deployment.

For deeper configuration examples, see the legacy FL integration guide.
