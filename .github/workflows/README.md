# GitHub Actions Workflows

This directory contains CI/CD workflows for automated testing, building, and deployment.

## Available Workflows

### 1. `ci.yml` - Continuous Integration
**Trigger**: Push to main/develop, Pull requests
**Purpose**: Main CI pipeline for testing and validation

**Jobs**:
- **test**: Run tests on multiple OS (Ubuntu, Windows, macOS) and Python versions (3.9, 3.10, 3.11) with parallel execution (pytest-xdist)
- **lint**: Code quality checks (flake8, black, isort)
- **type-check**: Static type checking with mypy
- **security**: Security scanning with bandit
- **docker**: Docker build test
- **docs**: Documentation validation and link checking
- **quick-demo**: Run quick demo to ensure end-to-end functionality
- **pacs-tests**: PACS integration property tests (40/48 properties, 83% coverage)
- **coverage-report**: Generate and upload coverage reports
- **all-checks-passed**: Final status check

**Artifacts**:
- Coverage reports (XML and HTML)
- Security scan results
- Quick demo results
- PACS test results and Hypothesis statistics

**Status Badge**:
```markdown
![CI](https://github.com/your-org/repo/workflows/CI/badge.svg)
```

### 2. `release.yml` - Release Automation
**Trigger**: Push tags matching `v*.*.*` (e.g., v1.0.0)
**Purpose**: Automated release creation and package publishing

**Jobs**:
- **create-release**: Build Python package and create GitHub release
- **docker-release**: Build and push Docker images with version tags

**Steps**:
1. Run full test suite
2. Build Python package (wheel and sdist)
3. Generate changelog from git commits
4. Create GitHub release with artifacts
5. Build and push Docker image to Docker Hub

**Usage**:
```bash
# Create a new release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### 3. `cd.yml` - Continuous Deployment
**Trigger**: Push to main (when src/k8s/Dockerfile/requirements change), Manual dispatch
**Purpose**: Automated deployment pipeline with environment promotion

**Jobs**:
- **test**: Run full test suite (skippable via manual trigger)
- **build**: Build and push Docker image to GitHub Container Registry
- **deploy-dev**: Deploy to development environment (automatic)
- **deploy-staging**: Deploy to staging environment (automatic after dev)
- **deploy-prod**: Deploy to production with blue-green strategy (manual only)
- **rollback**: Automatic rollback on failure
- **notify**: Deployment status summary

**Features**:
- Multi-environment deployment (dev → staging → prod)
- Blue-green deployment for production
- Automatic rollback on failure
- Smoke tests and integration tests
- Security scanning with Trivy
- Health check verification

**Manual Deployment**:
```bash
# Deploy to specific environment
gh workflow run cd.yml -f environment=prod
```

**Environments**:
- Development: https://dev.histocore.example.com
- Staging: https://staging.histocore.example.com
- Production: https://histocore.example.com

### 4. `docker-publish.yml` - Docker Image Publishing
**Trigger**: Manual dispatch only
**Purpose**: Standalone Docker image publishing to Docker Hub

**Jobs**:
- **build-and-push**: Build and push Docker image to Docker Hub

**Tags Created**:
- `latest` (for main branch)
- `main-<sha>` (commit-specific)
- Branch name (for other branches)

### 5. `codeql.yml` - Security Analysis
**Trigger**: Push, Pull requests, Weekly schedule (Monday midnight)
**Purpose**: Advanced security scanning with CodeQL

**Features**:
- Detects security vulnerabilities
- Identifies code quality issues
- Runs security-and-quality queries
- Integrates with GitHub Security tab

### 6. `dependency-review.yml` - Dependency Security
**Trigger**: Pull requests
**Purpose**: Review dependency changes for security issues

**Features**:
- Checks for vulnerable dependencies
- Validates license compatibility
- Fails on moderate+ severity issues
- Comments summary in PR

## Workflow Status

Check workflow status at: `https://github.com/your-org/repo/actions`

## Required Secrets

Configure these secrets in repository settings:

### Kubernetes (for cd.yml)
- `KUBECONFIG_DEV`: Base64-encoded kubeconfig for dev cluster
- `KUBECONFIG_STAGING`: Base64-encoded kubeconfig for staging cluster
- `KUBECONFIG_PROD`: Base64-encoded kubeconfig for prod cluster

### Docker Hub (for docker-publish.yml and release.yml)
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password or access token

### Optional: Codecov (for coverage reporting)
- `CODECOV_TOKEN`: Codecov upload token

### Optional: Weights & Biases (for experiment tracking)
- `WANDB_API_KEY`: W&B API key

## Setting Up Secrets

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each required secret

### Encoding kubeconfig for GitHub Secrets
```bash
# Encode kubeconfig file
cat ~/.kube/config | base64 -w 0 > kubeconfig.b64

# Copy content and add as secret
cat kubeconfig.b64
```

## Workflow Triggers

### Automatic Triggers
- **Push to main/develop**: Runs CI, Docker publish
- **Push to main (src/k8s changes)**: Runs CD pipeline (dev → staging)
- **Pull requests**: Runs CI, dependency review
- **Tag push (v*.*.*)**: Runs release workflow
- **Weekly (Monday)**: Runs CodeQL security scan

### Manual Triggers
All workflows support manual triggering via `workflow_dispatch`:
1. Go to Actions tab
2. Select workflow
3. Click "Run workflow"
4. Choose branch and run

**CD Pipeline Manual Deployment**:
```bash
# Using GitHub CLI
gh workflow run cd.yml -f environment=prod -f skip_tests=false

# Or via Actions UI
# 1. Go to Actions → CD
# 2. Click "Run workflow"
# 3. Select environment (dev/staging/prod)
# 4. Optionally skip tests
```

## Local Testing

### Run tests locally (mimics CI)
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
pip install pytest-xdist  # For parallel execution

# Run tests (parallel execution like CI)
pytest tests/ -v -n auto -m "not property and not slow" --cov=src --cov-report=term

# Run PACS tests
pytest tests/test_pacs_*.py -v --hypothesis-show-statistics -m "not slow"

# Run linting
flake8 src/ tests/
black --check src/ tests/
isort --check-only src/ tests/

# Run type checking
mypy src/ --ignore-missing-imports

# Run security scan
bandit -r src/

# Test Docker build
docker build -t pathology-api:test .
docker run --rm pathology-api:test python -c "import src"

# Run quick demo
python run_quick_demo.py
```

### Test CD pipeline locally
```bash
# Build Docker image
docker build -t histocore:test .

# Test with Helm (requires k8s cluster)
helm install histocore-test ./k8s/helm/histocore \
  --namespace test \
  --create-namespace \
  --values ./k8s/helm/histocore/values-dev.yaml \
  --set image.repository=histocore \
  --set image.tag=test \
  --dry-run --debug

# Deploy to local k8s (minikube/kind)
helm upgrade --install histocore-local ./k8s/helm/histocore \
  --namespace local \
  --create-namespace \
  --values ./k8s/helm/histocore/values-dev.yaml \
  --set image.repository=histocore \
  --set image.tag=test

# Verify deployment
kubectl get pods -n local
kubectl logs -n local -l app.kubernetes.io/name=histocore
```

## Workflow Optimization

### Caching
All workflows use caching to speed up builds:
- **pip cache**: Python dependencies
- **Docker layer cache**: Docker builds (GitHub Actions cache)

### Matrix Strategy
CI runs tests across multiple configurations:
- OS: Ubuntu, Windows, macOS
- Python: 3.9, 3.10, 3.11

This ensures compatibility across platforms.

### Parallel Execution
Jobs run in parallel when possible:
```
test ─┐
lint ─┼─→ all-checks-passed
docker─┤
docs ─┘
```

## Troubleshooting

### Issue: Tests fail on Windows but pass on Linux
**Cause**: Path separator differences or line ending issues
**Solution**: 
- Use `pathlib.Path` for cross-platform paths
- Configure git to handle line endings: `git config core.autocrlf true`

### Issue: Docker build fails
**Cause**: Missing dependencies or incorrect Dockerfile
**Solution**:
- Test locally: `docker build -t test .`
- Check Dockerfile syntax
- Verify all required files are included (check .dockerignore)

### Issue: Coverage upload fails
**Cause**: Missing CODECOV_TOKEN or network issues
**Solution**:
- Add CODECOV_TOKEN secret
- Set `fail_ci_if_error: false` in workflow (already configured)

### Issue: Quick demo times out
**Cause**: Demo takes longer than 10 minutes
**Solution**:
- Increase timeout in workflow: `timeout-minutes: 15`
- Optimize demo for faster execution

### Issue: Dependency review blocks PR
**Cause**: Vulnerable or incompatible dependencies
**Solution**:
- Update vulnerable packages: `pip install --upgrade <package>`
- Review license compatibility
- Request exception if necessary

### Issue: CD deployment fails
**Cause**: Missing kubeconfig secrets or cluster access issues
**Solution**:
- Verify kubeconfig secrets are base64-encoded correctly
- Test cluster access: `kubectl --kubeconfig=<file> get nodes`
- Check Helm chart syntax: `helm lint ./k8s/helm/histocore`
- Verify namespace exists or use `--create-namespace`

### Issue: Blue-green deployment stuck
**Cause**: Green deployment not healthy or traffic switch failed
**Solution**:
- Check pod status: `kubectl get pods -n prod -l version=green`
- View logs: `kubectl logs -n prod -l version=green`
- Manual rollback: `kubectl patch service histocore -n prod -p '{"spec":{"selector":{"version":"blue"}}}'`

### Issue: Trivy security scan fails
**Cause**: Critical vulnerabilities in Docker image
**Solution**:
- Update base image in Dockerfile
- Update vulnerable dependencies in requirements.txt
- Review Trivy report in GitHub Security tab

## Best Practices

1. **Always run tests locally** before pushing
2. **Keep workflows fast** - aim for <10 minutes total
3. **Use caching** to speed up builds
4. **Monitor workflow runs** and fix failures promptly
5. **Update dependencies** regularly to avoid security issues
6. **Use semantic versioning** for releases (v1.0.0, v1.1.0, etc.)
7. **Write meaningful commit messages** (used in changelog)
8. **Test Docker images** before releasing

## Workflow Badges

Add these badges to your README.md:

```markdown
![CI](https://github.com/your-org/repo/workflows/CI/badge.svg)
![CD](https://github.com/your-org/repo/workflows/CD/badge.svg)
![Docker](https://github.com/your-org/repo/workflows/Docker%20Publish/badge.svg)
![CodeQL](https://github.com/your-org/repo/workflows/CodeQL/badge.svg)
[![codecov](https://codecov.io/gh/your-org/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/repo)
```

## Customization

### Modify test matrix
Edit `ci.yml`:
```yaml
matrix:
  os: [ubuntu-latest, windows-latest]  # Remove macOS
  python-version: ['3.10', '3.11']     # Remove 3.9
```

### Change Docker registry
Edit `docker-publish.yml` to use GitHub Container Registry:
```yaml
- name: Log in to GitHub Container Registry
  uses: docker/login-action@v3
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}
```

### Add new workflow
1. Create new YAML file in `.github/workflows/`
2. Define trigger, jobs, and steps
3. Test with `workflow_dispatch` first
4. Enable for automatic triggers

## Monitoring

### View workflow runs
- Go to repository Actions tab
- Click on workflow name
- View run history and logs

### Set up notifications
1. Go to repository Settings → Notifications
2. Configure email/Slack notifications for workflow failures

### Monitor costs
- GitHub Actions is free for public repos
- Private repos: 2,000 minutes/month free
- Monitor usage: Settings → Billing → Actions

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [CodeQL](https://codeql.github.com/)
