@echo off
REM AWS deployment script for HistoCore (Windows)

setlocal enabledelayedexpansion

set ENVIRONMENT=%1
set AWS_REGION=%2

if "%ENVIRONMENT%"=="" set ENVIRONMENT=production
if "%AWS_REGION%"=="" set AWS_REGION=us-west-2

echo Deploying HistoCore to AWS EKS...
echo Environment: %ENVIRONMENT%
echo Region: %AWS_REGION%

REM Check tools
where terraform >nul 2>&1
if errorlevel 1 (
    echo terraform not found
    exit /b 1
)

where aws >nul 2>&1
if errorlevel 1 (
    echo AWS CLI not found
    exit /b 1
)

where kubectl >nul 2>&1
if errorlevel 1 (
    echo kubectl not found
    exit /b 1
)

REM Terraform init
cd terraform
echo Initializing Terraform...
terraform init

REM Plan
echo Planning infrastructure...
terraform plan -var="environment=%ENVIRONMENT%" -var="aws_region=%AWS_REGION%"

REM Apply
set /p response="Apply infrastructure? (y/N): "
if /i "%response%"=="y" (
    terraform apply -var="environment=%ENVIRONMENT%" -var="aws_region=%AWS_REGION%" -auto-approve
) else (
    echo Deployment cancelled
    exit /b 0
)

REM Get outputs
for /f "tokens=*" %%i in ('terraform output -raw cluster_name') do set CLUSTER_NAME=%%i
for /f "tokens=*" %%i in ('terraform output -raw redis_endpoint') do set REDIS_ENDPOINT=%%i
for /f "tokens=*" %%i in ('terraform output -raw s3_bucket_name') do set S3_BUCKET=%%i

REM Configure kubectl
echo Configuring kubectl...
aws eks update-kubeconfig --region %AWS_REGION% --name %CLUSTER_NAME%

REM Install NVIDIA device plugin
echo Installing NVIDIA device plugin...
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

REM Deploy HistoCore
cd ..\..\..\k8s
echo Deploying HistoCore to EKS...

REM Update ConfigMap with AWS resources
kubectl create configmap histocore-config --from-literal=REDIS_URL="redis://%REDIS_ENDPOINT%:6379" --from-literal=S3_BUCKET="%S3_BUCKET%" --from-literal=AWS_REGION="%AWS_REGION%" --dry-run=client -o yaml | kubectl apply -f -

REM Deploy
kubectl apply -f namespace.yaml
kubectl apply -f secret.yaml
kubectl apply -f streaming.yaml
kubectl apply -f redis.yaml
kubectl apply -f monitoring.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml

REM Wait for deployment
echo Waiting for deployment...
kubectl wait --for=condition=available --timeout=600s deployment/histocore-streaming -n histocore

REM Get ALB endpoint
for /f "tokens=*" %%i in ('kubectl get ingress histocore-ingress -n histocore -o jsonpath="{.status.loadBalancer.ingress[0].hostname}"') do set ALB_ENDPOINT=%%i

echo Deployment complete!
echo Cluster: %CLUSTER_NAME%
echo Redis: %REDIS_ENDPOINT%
echo S3: %S3_BUCKET%
echo ALB: %ALB_ENDPOINT%
echo.
echo Access via:
echo https://%ALB_ENDPOINT%
echo kubectl port-forward -n histocore svc/histocore-streaming 8000:8000