# AWS & Azure Business Associate Agreement (BAA) Guide

## Overview

Before deploying the platform to production with hospital PHI data, you **MUST** execute Business Associate Agreements (BAAs) with your cloud providers. This is a **HIPAA requirement** - without BAAs, you cannot legally store or process PHI in the cloud.

---

## AWS BAA Process

### Prerequisites
- AWS account (free tier OK for initial setup)
- Credit card on file
- Estimated monthly spend: $500-2000 (depends on usage)

### Step-by-Step Process

#### 1. Sign Up for AWS Account (If Needed)
- Go to: https://aws.amazon.com/
- Click "Create an AWS Account"
- Provide: Email, password, account name
- Enter payment information
- Verify identity (phone verification)

#### 2. Request AWS BAA
AWS provides BAAs automatically for eligible services. No manual request needed.

**How to Access**:
1. Log into AWS Console: https://console.aws.amazon.com/
2. Navigate to: **AWS Artifact** (search in services)
3. Click: **Agreements** → **AWS Business Associate Addendum**
4. Click: **Download Agreement**
5. Review and **Accept Agreement** (electronic signature)

**Timeline**: Immediate (self-service)

**Cost**: Free (included with AWS account)

#### 3. Verify HIPAA-Eligible Services
Only certain AWS services are HIPAA-eligible. For the platform, use:

**Compute**:
- ✅ Amazon EC2 (virtual machines)
- ✅ Amazon ECS (container orchestration)
- ✅ Amazon EKS (Kubernetes)
- ✅ AWS Lambda (serverless functions)

**Storage**:
- ✅ Amazon S3 (object storage) - **MUST enable encryption**
- ✅ Amazon EBS (block storage)
- ✅ Amazon EFS (file storage)

**Database**:
- ✅ Amazon RDS (PostgreSQL, MySQL)
- ✅ Amazon DynamoDB

**Networking**:
- ✅ Amazon VPC (virtual private cloud)
- ✅ AWS PrivateLink
- ✅ Elastic Load Balancing

**Security**:
- ✅ AWS KMS (key management)
- ✅ AWS CloudTrail (audit logging)
- ✅ AWS CloudWatch (monitoring)

**NOT HIPAA-Eligible** (Do NOT use for PHI):
- ❌ Amazon Lightsail
- ❌ AWS Elastic Beanstalk (some configurations)
- ❌ Amazon WorkSpaces (personal)

Full list: https://aws.amazon.com/compliance/hipaa-eligible-services-reference/

#### 4. Configure HIPAA-Compliant Infrastructure

**Enable Encryption**:
```bash
# S3 bucket encryption (required for PHI)
aws s3api put-bucket-encryption \
  --bucket the platform-phi-data \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "aws:kms",
        "KMSMasterKeyID": "arn:aws:kms:us-east-1:123456789:key/abc-123"
      }
    }]
  }'

# EBS volume encryption (required for PHI)
aws ec2 create-volume \
  --size 100 \
  --encrypted \
  --kms-key-id arn:aws:kms:us-east-1:123456789:key/abc-123 \
  --availability-zone us-east-1a
```

**Enable Audit Logging**:
```bash
# CloudTrail (audit all API calls)
aws cloudtrail create-trail \
  --name the platform-audit-trail \
  --s3-bucket-name the platform-audit-logs \
  --is-multi-region-trail \
  --enable-log-file-validation

aws cloudtrail start-logging --name the platform-audit-trail
```

**Configure VPC** (network isolation):
```bash
# Create private VPC for PHI workloads
aws ec2 create-vpc --cidr-block 10.0.0.0/16 --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=the platform-phi-vpc}]'
```

#### 5. Document Compliance
Create file: `AWS_HIPAA_COMPLIANCE.md`

Document:
- BAA acceptance date
- Services used (all HIPAA-eligible)
- Encryption configuration
- Audit logging setup
- Access controls (IAM policies)
- Backup procedures
- Incident response plan

---

## Azure BAA Process

### Prerequisites
- Azure account
- Credit card on file
- Estimated monthly spend: $500-2000

### Step-by-Step Process

#### 1. Sign Up for Azure Account (If Needed)
- Go to: https://azure.microsoft.com/
- Click "Start Free" or "Pay As You Go"
- Provide: Email, password, organization details
- Enter payment information
- Verify identity

#### 2. Request Azure BAA
Azure requires **manual BAA request** (not self-service like AWS).

**How to Request**:

**Option A: Through Azure Portal**
1. Log into Azure Portal: https://portal.azure.com/
2. Navigate to: **Microsoft Trust Center** → **Compliance Manager**
3. Search for: "HIPAA BAA"
4. Click: **Request BAA**
5. Fill out form:
   - Organization name
   - Contact information
   - Intended use (AI pathology analysis)
   - Estimated PHI volume
6. Submit request

**Option B: Contact Azure Support**
1. Open support ticket: https://azure.microsoft.com/en-us/support/create-ticket/
2. Issue type: "Billing & Subscription"
3. Problem type: "HIPAA BAA Request"
4. Description: "Requesting Business Associate Agreement for HIPAA-compliant deployment of AI pathology system"
5. Include:
   - Organization name
   - Azure subscription ID
   - Contact email/phone
   - Intended use case

**Option C: Email Azure Compliance Team**
- Email: azurecompliance@microsoft.com
- Subject: "HIPAA BAA Request - [Your Organization]"
- Body: Include organization details, subscription ID, use case

**Timeline**: 5-10 business days (manual review)

**Cost**: Free (included with Azure account)

#### 3. Verify HIPAA-Eligible Services
For the platform, use:

**Compute**:
- ✅ Azure Virtual Machines
- ✅ Azure Kubernetes Service (AKS)
- ✅ Azure Container Instances
- ✅ Azure Functions

**Storage**:
- ✅ Azure Blob Storage (with encryption)
- ✅ Azure Files
- ✅ Azure Disk Storage

**Database**:
- ✅ Azure SQL Database
- ✅ Azure Database for PostgreSQL
- ✅ Azure Cosmos DB

**Networking**:
- ✅ Azure Virtual Network
- ✅ Azure Private Link
- ✅ Azure Load Balancer

**Security**:
- ✅ Azure Key Vault
- ✅ Azure Monitor
- ✅ Azure Security Center

Full list: https://docs.microsoft.com/en-us/azure/compliance/offerings/offering-hipaa-us

#### 4. Configure HIPAA-Compliant Infrastructure

**Enable Encryption**:
```bash
# Storage account encryption (required for PHI)
az storage account create \
  --name the platformphistorage \
  --resource-group the platform-rg \
  --location eastus \
  --sku Standard_LRS \
  --encryption-services blob file \
  --encryption-key-source Microsoft.Keyvault \
  --encryption-key-vault https://the platform-kv.vault.azure.net/ \
  --encryption-key-name phi-encryption-key
```

**Enable Audit Logging**:
```bash
# Azure Monitor (audit all operations)
az monitor diagnostic-settings create \
  --name the platform-audit-logs \
  --resource /subscriptions/{subscription-id}/resourceGroups/the platform-rg \
  --logs '[{"category": "Administrative", "enabled": true}]' \
  --workspace /subscriptions/{subscription-id}/resourceGroups/the platform-rg/providers/Microsoft.OperationalInsights/workspaces/the platform-logs
```

**Configure Virtual Network**:
```bash
# Create private VNet for PHI workloads
az network vnet create \
  --name the platform-phi-vnet \
  --resource-group the platform-rg \
  --address-prefix 10.0.0.0/16 \
  --subnet-name phi-subnet \
  --subnet-prefix 10.0.1.0/24
```

#### 5. Document Compliance
Create file: `AZURE_HIPAA_COMPLIANCE.md`

Document same items as AWS section.

---

## BAA Comparison: AWS vs Azure

| Feature | AWS | Azure |
|---------|-----|-------|
| **BAA Process** | Self-service (immediate) | Manual request (5-10 days) |
| **Cost** | Free | Free |
| **HIPAA Services** | 100+ services | 90+ services |
| **Encryption** | KMS (Key Management Service) | Key Vault |
| **Audit Logging** | CloudTrail | Azure Monitor |
| **Compliance Docs** | AWS Artifact | Trust Center |
| **Support** | 24/7 (paid plans) | 24/7 (paid plans) |

**Recommendation**: Start with **AWS** (faster BAA, easier setup). Add Azure later if needed for redundancy.

---

## Post-BAA Checklist

After executing BAAs, complete these steps:

### 1. Document BAA Execution
- [ ] Save signed BAA PDF to secure location
- [ ] Record BAA effective date
- [ ] Note BAA expiration date (if applicable)
- [ ] Add to regulatory documentation folder

### 2. Configure Cloud Infrastructure
- [ ] Enable encryption on all storage (S3/Blob)
- [ ] Enable audit logging (CloudTrail/Monitor)
- [ ] Configure VPC/VNet (network isolation)
- [ ] Set up IAM/RBAC (access controls)
- [ ] Enable MFA for all admin accounts
- [ ] Configure backup and disaster recovery

### 3. Deploy the platform
- [ ] Deploy to HIPAA-compliant services only
- [ ] Verify encryption at rest and in transit
- [ ] Test audit logging (verify logs captured)
- [ ] Configure monitoring and alerting
- [ ] Document deployment architecture

### 4. Security Assessment
- [ ] Run vulnerability scan
- [ ] Review IAM/RBAC policies
- [ ] Test backup and restore procedures
- [ ] Verify encryption keys secured
- [ ] Document security controls

### 5. Compliance Documentation
- [ ] Create HIPAA compliance document
- [ ] Document data flow diagrams
- [ ] List all cloud services used
- [ ] Document encryption configuration
- [ ] Document access controls
- [ ] Document audit logging setup
- [ ] Document incident response plan

---

## Cost Estimates

### AWS Monthly Costs (Estimated)

**Small Deployment** (1-2 hospitals, <100 slides/day):
- EC2 (t3.xlarge): $120/month
- S3 Storage (1TB): $23/month
- RDS PostgreSQL (db.t3.medium): $60/month
- Data Transfer: $50/month
- **Total**: ~$250/month

**Medium Deployment** (3-5 hospitals, 100-500 slides/day):
- EC2 (c5.2xlarge): $250/month
- S3 Storage (5TB): $115/month
- RDS PostgreSQL (db.m5.large): $150/month
- Data Transfer: $200/month
- **Total**: ~$700/month

**Large Deployment** (10+ hospitals, 1000+ slides/day):
- EKS Cluster (3 nodes, c5.4xlarge): $900/month
- S3 Storage (20TB): $460/month
- RDS PostgreSQL (db.m5.2xlarge): $500/month
- Data Transfer: $500/month
- **Total**: ~$2,400/month

### Azure Monthly Costs (Similar to AWS)

**Note**: Costs vary based on:
- Region (US East typically cheapest)
- Reserved instances (save 30-70% with 1-3 year commitment)
- Spot instances (save 70-90% for non-critical workloads)
- Data transfer (minimize cross-region transfers)

---

## Common Issues & Solutions

### Issue 1: "BAA not available for my account"
**Solution**: Ensure you have a paid AWS/Azure account (not free tier only). Add payment method.

### Issue 2: "Service not HIPAA-eligible"
**Solution**: Check official HIPAA-eligible services list. Use alternative service or request eligibility.

### Issue 3: "Encryption not enabled by default"
**Solution**: Manually enable encryption on all storage resources. Use KMS/Key Vault for key management.

### Issue 4: "Audit logs not capturing all events"
**Solution**: Enable CloudTrail/Monitor for all regions. Configure log retention (7+ years for HIPAA).

### Issue 5: "High data transfer costs"
**Solution**: Use VPC/VNet endpoints for internal traffic. Minimize cross-region transfers. Use CloudFront/CDN for static content.

---

## Next Steps

1. **This Week**: Execute AWS BAA (immediate)
2. **Next Week**: Request Azure BAA (5-10 day wait)
3. **Week 3**: Configure HIPAA-compliant infrastructure
4. **Week 4**: Deploy the platform to test environment
5. **Week 5**: Security assessment and compliance documentation
6. **Week 6**: Ready for hospital pilot deployment

---

## Resources

**AWS**:
- BAA: https://aws.amazon.com/compliance/hipaa-compliance/
- HIPAA Whitepaper: https://d1.awsstatic.com/whitepapers/compliance/AWS_HIPAA_Compliance_Whitepaper.pdf
- Eligible Services: https://aws.amazon.com/compliance/hipaa-eligible-services-reference/

**Azure**:
- BAA: https://www.microsoft.com/en-us/trust-center/compliance/hipaa
- HIPAA Blueprint: https://docs.microsoft.com/en-us/azure/governance/blueprints/samples/hipaa-hitrust-9-2
- Eligible Services: https://docs.microsoft.com/en-us/azure/compliance/offerings/offering-hipaa-us

**HIPAA Guidance**:
- HHS HIPAA: https://www.hhs.gov/hipaa/index.html
- OCR Guidance: https://www.hhs.gov/hipaa/for-professionals/security/guidance/index.html

---

**Contact**: If you encounter issues, email support@the platform-medical.ai or call +1 (650) 555-0199.
