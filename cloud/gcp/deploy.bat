@echo off
REM GCP deployment script for HistoCore (Windows)

setlocal enabledelayedexpansion

set PROJECT_ID=%1
set ENVIRONMENT=%2
set GCP_REGION=%3

if "%PROJECT_ID%"=="" (
    echo Usage: %0 ^<PROJECT_ID^> [ENVIRONMENT] [REGION]
    echo Example: %0 my-histocore-project production us-central1
    exit /b 1
)

if "%ENVIRONMENT%"=="" set ENVIRONMENT=production
if "%GCP_REGION%"=="" set GCP_REGION=us-central1

echo Deploying HistoCore to Google Cloud...
echo Project: %PROJECT_ID%
echo Environment: %ENVIRONMENT%
echo Region: %GCP_REGION%

REM Check tools
where terraform >nul 2>&1
if errorlevel 1 (
    echo terraform not found
    exit /b 1
)

where gcloud >nul 2>&1
if errorlevel 1 (
    echo Google Cloud SDK not found
    exit /b 1
)

where kubectl >nul 2>&1
if errorlevel 1 (
    echo kubectl not found
    exit /b 1
)

REM Set project
gcloud config set project %PROJECT_ID%

REM Generate PostgreSQL password
for /f "tokens=*" %%i in ('powershell -command "[System.Web.Security.Membership]::GeneratePassword(32, 8)"') do set POSTGRES_PASSWORD=%%i

REM Terraform init
cd terraform
echo Initializing Terraform...
terraform init

REM Plan
echo Planning infrastructure...
terraform plan -var="project_id=%PROJECT_ID%" -var="environment=%ENVIRONMENT%" -var="gcp_region=%GCP_REGION%" -var="postgres_password=%POSTGRES_PASSWORD%"

REM Apply
set /p response="Apply infrastructure? (y/N): "
if /i "%response%"=="y" (
    terraform apply -var="project_id=%PROJECT_ID%" -var="environment=%ENVIRONMENT%" -var="gcp_region=%GCP_REGION%" -var="postgres_password=%POSTGRES_PASSWORD%" -auto-approve
) else (
    echo Deployment cancelled
    exit /b 0
)

REM Get outputs
for /f "tokens=*" %%i in ('terraform output -raw gke_cluster_name') do set CLUSTER_NAME=%%i
for /f "tokens=*" %%i in ('terraform output -raw redis_host') do set REDIS_HOST=%%i
for /f "tokens=*" %%i in ('terraform output -raw redis_port') do set REDIS_PORT=%%i
for /f "tokens=*" %%i in ('terraform output -raw redis_auth_string') do set REDIS_AUTH=%%i
for /f "tokens=*" %%i in ('terraform output -raw storage_bucket_name') do set STORAGE_BUCKET=%%i
for /f "tokens=*" %%i in ('terraform output -raw postgres_connection_name') do set POSTGRES_CONNECTION=%%i
for /f "tokens=*" %%i in ('terraform output -raw load_balancer_ip') do set LB_IP=%%i

REM Configure kubectl
echo Configuring kubectl...
gcloud container clusters get-credentials %CLUSTER_NAME% --region %GCP_REGION% --project %PROJECT_ID%

REM Install NVIDIA device plugin
echo Installing NVIDIA device plugin...
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

REM Deploy HistoCore
cd ..\..\..\k8s
echo Deploying HistoCore to GKE...

REM Update ConfigMap with GCP resources
kubectl create configmap histocore-config --from-literal=REDIS_URL="redis://:%REDIS_AUTH%@%REDIS_HOST%:%REDIS_PORT%" --from-literal=GCS_BUCKET="%STORAGE_BUCKET%" --from-literal=POSTGRES_CONNECTION="%POSTGRES_CONNECTION%" --from-literal=POSTGRES_PASSWORD="%POSTGRES_PASSWORD%" --from-literal=GCP_PROJECT="%PROJECT_ID%" --dry-run=client -o yaml | kubectl apply -f -

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

echo Deployment complete!
echo Project: %PROJECT_ID%
echo Cluster: %CLUSTER_NAME%
echo Redis: %REDIS_HOST%:%REDIS_PORT%
echo Storage: gs://%STORAGE_BUCKET%
echo Load Balancer IP: %LB_IP%
echo.
echo Access via:
echo http://%LB_IP%
echo kubectl port-forward -n histocore svc/histocore-streaming 8000:8000
echo.
echo PostgreSQL password saved in Terraform state. Retrieve with:
echo terraform output -raw postgres_password