# Docker research environment

The Docker files in this repository provide **local research and engineering infrastructure**. They are not evidence of clinical validation, hospital deployment, regulatory readiness, or production operation. See [`CLAIM_BOUNDARY.md`](../CLAIM_BOUNDARY.md).

## Two container paths

### Root `Dockerfile`

The root Dockerfile is the lightweight CI/research API image.

```bash
# Safe default: CPU image
docker build -t pathology-research-api .

# Explicit CUDA image
docker build --target gpu -t pathology-research-api:gpu .
```

For compatibility with older build scripts, `--target production` is an alias of the CPU research image. The name is historical; it does not indicate a deployed production service.

### `docker-compose.yml`

The Compose stack is for local development and integration testing. It currently contains:

- FastAPI research API;
- PostgreSQL;
- Redis;
- Nginx;
- Prometheus;
- Grafana.

Dead worker/DICOM demo services are intentionally excluded from the active stack.

## First-time local setup

Use the provided secret-free template:

```bash
cp .env.docker.example .env
```

Replace every `CHANGE_ME` value. The local launch scripts refuse to continue while placeholders remain.

Linux/macOS:

```bash
bash scripts/deploy.sh
```

Windows:

```powershell
scripts\deploy.bat
```

Both launchers validate the Compose configuration before starting services.

You can also run Compose directly after configuring `.env`:

```bash
docker compose config
docker compose up -d --build
```

Local endpoints:

- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

Stop the stack with:

```bash
docker compose down
```

## Secrets

Do not commit `.env`, database passwords, JWT keys, Grafana credentials, certificates, or tokens.

The tracked `.env.docker.example` is a template only. The active Compose file requires local values for:

- `POSTGRES_PASSWORD`;
- `DATABASE_URL`;
- `JWT_SECRET_KEY`;
- `GRAFANA_ADMIN_PASSWORD`.

The database initialization script creates schema only; it does **not** create a known default administrator account.

## API image behavior

`docker/Dockerfile.api`:

- uses Python 3.11 slim;
- installs runtime system libraries including OpenSlide;
- copies `src/` only;
- does not copy tests, experiment code, notebooks, or environment files;
- runs as an unprivileged user;
- exposes port 8000;
- uses `/health` for its container health check.

Runtime configuration is injected through environment variables rather than baked into the image.

## CI

The main CI workflow builds and imports the root Docker image after the Python test matrix. Docker configuration changes should therefore be reviewed together with CI results.

The larger Compose stack is development scaffolding and should be validated locally with:

```bash
docker compose config
docker compose build api
```

## Scope boundary

Cloud, Kubernetes, PACS, monitoring, and deployment-oriented directories include historical or experimental engineering work. Their presence in this repository does not mean those systems have been deployed or clinically validated. Treat the current claim boundary and research evidence ledger as authoritative.
