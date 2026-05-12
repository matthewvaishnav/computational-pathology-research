# Secrets Management Guide

## Overview

Never commit secrets to version control. Use environment variables, secrets managers, or encrypted vaults.

## Environment Variables

### Development

```bash
# .env (gitignored)
DB_PASSWORD=dev_password_here
SMTP_PASSWORD=smtp_password_here
TWILIO_AUTH_TOKEN=twilio_token_here
ADMIN_API_KEY=admin_key_here
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()

password = os.getenv('DB_PASSWORD')
```

### Production

Use secrets manager:
- **Kubernetes**: External Secrets Operator + Vault/AWS Secrets Manager
- **Docker**: Docker secrets
- **Cloud**: AWS Secrets Manager, Azure Key Vault, GCP Secret Manager

## Kubernetes Secrets

### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: histocore-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: histocore-secrets
  data:
    - secretKey: DB_PASSWORD
      remoteRef:
        key: histocore/database
        property: password
```

### Manual Secrets (Development Only)

```bash
# Create secret from literal
kubectl create secret generic histocore-secrets \
  --from-literal=DB_PASSWORD='your-password' \
  --from-literal=ADMIN_API_KEY='your-key'

# Create secret from file
kubectl create secret generic histocore-secrets \
  --from-env-file=.env.production
```

## HashiCorp Vault

### Setup

```bash
# Install Vault
helm install vault hashicorp/vault

# Initialize
kubectl exec -it vault-0 -- vault operator init

# Store secret
vault kv put secret/histocore/database \
  password="secure-password" \
  username="histocore_user"
```

### Access from App

```python
import hvac

client = hvac.Client(url='http://vault:8200')
client.token = os.getenv('VAULT_TOKEN')

secret = client.secrets.kv.v2.read_secret_version(
    path='histocore/database'
)
password = secret['data']['data']['password']
```

## AWS Secrets Manager

```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        raise e
```

## Azure Key Vault

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(
    vault_url="https://histocore-vault.vault.azure.net/",
    credential=credential
)

secret = client.get_secret("db-password")
password = secret.value
```

## Secret Rotation

### Automated Rotation

```yaml
# External Secrets with rotation
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: histocore-secrets
spec:
  refreshInterval: 15m  # Check every 15 minutes
  target:
    creationPolicy: Owner
    deletionPolicy: Retain
```

### Manual Rotation

```bash
# 1. Generate new secret
NEW_PASSWORD=$(openssl rand -base64 32)

# 2. Update in secrets manager
vault kv put secret/histocore/database password="$NEW_PASSWORD"

# 3. Restart pods to pick up new secret
kubectl rollout restart deployment/histocore
```

## Best Practices

### DO

- ✅ Use environment variables
- ✅ Use secrets managers (Vault, AWS Secrets Manager)
- ✅ Rotate secrets regularly
- ✅ Use different secrets per environment
- ✅ Encrypt secrets at rest
- ✅ Audit secret access
- ✅ Use least privilege access
- ✅ Set secret expiration

### DON'T

- ❌ Commit secrets to git
- ❌ Log secrets
- ❌ Share secrets via email/chat
- ❌ Use same secret across environments
- ❌ Store secrets in code
- ❌ Use weak/default passwords
- ❌ Share secrets between services unnecessarily

## Secret Detection

### Pre-commit Hooks

```bash
# Install
pip install pre-commit detect-secrets
pre-commit install

# Scan existing repo
detect-secrets scan > .secrets.baseline

# Update baseline
detect-secrets scan --baseline .secrets.baseline
```

### CI/CD Scanning

```yaml
# GitHub Actions
- name: Detect secrets
  uses: trufflesecurity/trufflehog@main
  with:
    path: ./
    base: ${{ github.event.repository.default_branch }}
    head: HEAD
```

## Emergency Response

### Secret Leaked to Git

```bash
# 1. Rotate secret immediately
# 2. Remove from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/secret" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Force push (coordinate with team)
git push origin --force --all

# 4. Audit access logs
# 5. Notify security team
```

### Compromised Secret

1. **Immediate**: Rotate secret
2. **Audit**: Check access logs
3. **Investigate**: Determine scope of compromise
4. **Notify**: Security team + affected parties
5. **Document**: Incident report

## Compliance

### HIPAA

- Encrypt secrets at rest (AES-256)
- Encrypt secrets in transit (TLS 1.2+)
- Audit all secret access
- Implement access controls
- Regular security assessments

### SOC 2

- Document secret management procedures
- Implement secret rotation
- Monitor secret access
- Incident response plan
- Regular audits

## Tools

### Recommended

- **HashiCorp Vault**: Enterprise secret management
- **AWS Secrets Manager**: AWS-native solution
- **Azure Key Vault**: Azure-native solution
- **External Secrets Operator**: Kubernetes integration
- **detect-secrets**: Pre-commit scanning
- **TruffleHog**: Git history scanning

### Encryption

```python
# Encrypt secret before storage
from cryptography.fernet import Fernet

key = Fernet.generate_key()
cipher = Fernet(key)

encrypted = cipher.encrypt(b"secret-password")
decrypted = cipher.decrypt(encrypted)
```

## References

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [CWE-798: Use of Hard-coded Credentials](https://cwe.mitre.org/data/definitions/798.html)
- [NIST SP 800-57: Key Management](https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final)
