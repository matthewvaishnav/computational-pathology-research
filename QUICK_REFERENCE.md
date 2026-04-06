# Quick Reference Card

## One-Minute Overview

**What**: Multimodal computational pathology framework with production deployment
**Status**: Working code with actual training results
**Purpose**: Demonstrate ML engineering capabilities for hiring

## Quick Commands

### Run Demos (3-5 minutes each)
```bash
python run_quick_demo.py              # Basic training demo
python run_missing_modality_demo.py   # Robustness test
python run_temporal_demo.py           # Temporal reasoning
```

### Docker Deployment
```bash
docker-compose up -d api              # Start API
curl http://localhost:8000/health     # Test health
docker-compose logs -f api            # View logs
docker-compose down                   # Stop service
```

### Testing
```bash
pytest tests/ -v                      # Run all tests
pytest tests/ --cov=src               # With coverage
```

### Development
```bash
# Setup environment
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Run Jupyter
jupyter notebook notebooks/
```

## Key Files

### Documentation (Start Here)
- `PORTFOLIO_SUMMARY.md` - Complete portfolio overview
- `README.md` - Main documentation
- `ARCHITECTURE.md` - System design
- `DOCKER.md` - Deployment guide

### Results (Proof It Works)
- `results/quick_demo/` - Training results
- `DEMO_RESULTS.md` - Results analysis
- `PERFORMANCE.md` - Benchmarks

### Code (Implementation)
- `src/models/multimodal.py` - Main fusion model
- `src/data/loaders.py` - Data pipeline
- `deploy/api.py` - REST API

### Deployment
- `Dockerfile` - Container definition
- `docker-compose.yml` - Service orchestration
- `deploy/README.md` - Deployment guide

## Architecture at a Glance

```
Input Modalities → Encoders → Fusion → Task Heads → Predictions
     ↓               ↓          ↓          ↓            ↓
  WSI/Genomic/   Attention   Cross-   Classifier/   Classes/
  Clinical Text   Pooling    Modal    Survival      Survival
                            Attention
```

## API Endpoints

```bash
GET  /health        # Health check
GET  /model-info    # Model metadata
POST /predict       # Single prediction
POST /batch-predict # Batch predictions
```

## Test Coverage

- Overall: 66%
- Data pipeline: 85%
- Models: 60%
- Pretraining: 70%

## Performance (Quick Demo)

- Training: 3 minutes (5 epochs, CPU)
- Validation: 93% accuracy
- Test: 83% accuracy
- Memory: ~2GB RAM

## What Makes This Special

1. ✅ **Actual Results**: Not just code, real training outputs
2. ✅ **Production Ready**: Docker, API, monitoring
3. ✅ **Well Tested**: 90+ tests, 66% coverage
4. ✅ **Documented**: 10+ documentation files
5. ✅ **Honest**: Clear about limitations

## Common Tasks

### Add New Modality
1. Create encoder in `src/models/encoders.py`
2. Update fusion in `src/models/fusion.py`
3. Add to dataset in `src/data/loaders.py`
4. Add tests in `tests/`

### Train New Model
1. Prepare data in `data/`
2. Configure in `experiments/configs/`
3. Run training script
4. Evaluate and save results

### Deploy to Production
1. Build Docker image
2. Configure environment variables
3. Deploy to cloud (AWS/GCP/Azure)
4. Setup monitoring and logging

### Run Experiments
1. Modify config in `experiments/configs/`
2. Run training script
3. Analyze results in `results/`
4. Update documentation

## Troubleshooting

### Import Errors
```bash
pip install -e .  # Install package in dev mode
```

### Out of Memory
```bash
# Reduce batch size in config
# Or use smaller model
# Or enable gradient checkpointing
```

### Docker Issues
```bash
docker-compose logs api  # Check logs
docker system prune      # Clean up
```

### Test Failures
```bash
pytest tests/ -v -s      # Verbose output
pytest tests/test_X.py   # Run specific test
```

## Resources

### Internal Documentation
- Architecture: `ARCHITECTURE.md`
- Performance: `PERFORMANCE.md`
- Deployment: `DOCKER.md`
- Tutorial: `notebooks/00_getting_started.ipynb`

### External Resources
- PyTorch: https://pytorch.org/docs/
- FastAPI: https://fastapi.tiangolo.com/
- Docker: https://docs.docker.com/

## Statistics

- **Code**: ~15,000 lines
- **Tests**: 90+ tests
- **Coverage**: 66%
- **Docs**: 10+ files
- **Demos**: 4 working demos
- **Deployment**: 6+ options

## For Reviewers

### 5-Minute Evaluation
1. Read `PORTFOLIO_SUMMARY.md`
2. Run `python run_quick_demo.py`
3. Check `results/quick_demo/`
4. Browse `src/models/`

### 15-Minute Evaluation
1. Read `README.md` and `ARCHITECTURE.md`
2. Run all demos
3. Review code structure
4. Run tests: `pytest tests/ -v`
5. Start Docker: `docker-compose up -d api`

### 30-Minute Evaluation
1. Complete 15-minute evaluation
2. Read `PERFORMANCE.md` and `DOCKER.md`
3. Review `deploy/api.py`
4. Go through `notebooks/00_getting_started.ipynb`
5. Check test coverage: `pytest tests/ --cov=src --cov-report=html`

## Contact

For questions about this repository or to discuss ML engineering opportunities, please reach out via GitHub issues or the contact information in the main README.

## License

MIT License - See LICENSE file for details.
