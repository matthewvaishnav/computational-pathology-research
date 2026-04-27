@echo off
REM Azure deployment script for HistoCore (Windows)

setlocal enabledelayedexpansion

set ENVIRONMENT=%1
set AZURE_REGION=%2

if "%ENVIRONMENT%"=="" set ENVIRONMENT=production
if "%AZURE_REGION%"=="" set AZURE_REGION=East US

echo Deploying HistoCore to Azure AKS...
echo Environment: %ENVIRONMENT%
echo Region: %AZURE_REGION%

REM Check tools
where terraform >nul 2>&1
if errorlevel 1 (
    echo terraform not found
    exit /b 1
)

where az >nul 2>&1
if errorlevel 1 (
    echo Azure CLI not found
    exit /b 1
)

where kubectl >nul 2>&1
if errorlevel 1 (
    echo kubectl not found
    exit /b 1
)

REM Azure login check
az account show >nul 2>&1
if errorlevel 1 (
    echo Please login to Azure CLI first:
    echo az login
    exit /b 1
)

REM Terraform init
cd terraform
echo Initializing Terraform...
terraform init

REM Plan
echo Planning infrastructure...
terraform plan -var="environment=%ENVIRONMENT%" -var="azure_region=%AZURE_REGION%"

REM Apply
set /p response="Apply infrastructure? (y/N): "
if /i "%response%"=="y" (
    terraform apply -var="environment=%ENVIRONMENT%" -var="azure_region=%AZURE_REGION%" -auto-approve
) else (
    echo Deployment cancelled
    exit /b 0
)

REM Get outputs
for /f "tokens=*" %%i in ('terraform output -raw resource_group_name') do set RESOURCE_GROUP=%%i
for /f "tokens=*" %%i in ('terraform output -raw aks_cluster_name') do set CLUSTER_NAME=%%i
for /f "tokens=*" %%i in ('terraform output -raw redis_hostname') do set REDIS_HOSTNAME=%%i
for /f "tokens=*" %%i in ('terraform output -raw redis_port') do set REDIS_PORT=%%i
for /f "tokens=*" %%i in ('terraform output -raw storage_account_name') do set STORAGE_ACCOUNT=%%i
for /f "tokens=*" %%i in ('terraform output -raw storage_container_name') do set STORAGE_CONTAINER=%%i
for /f "tokens=*" %%i in ('terraform output -raw key_vault_name') do set KEY_VAULT=%%i

REM Configure kubectl
echo Configuring kubectl...
az aks get-credentials --resource-group %RESOURCE_GROUP% --name %CLUSTER_NAME% --overwrite-existing

REM Install NVIDIA device plugin
echo Installing NVIDIA device plugin...
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

REM Deploy HistoCore
cd ..\..\..\k8s
echo Deploying HistoCore to AKS...

REM Update ConfigMap with Azure resources
kubectl create configmap histocore-config --from-literal=REDIS_URL="redis://%REDIS_HOSTNAME%:%REDIS_PORT%" --from-literal=AZURE_STORAGE_ACCOUNT="%STORAGE_ACCOUNT%" --from-literal=AZURE_STORAGE_CONTAINER="%STORAGE_CONTAINER%" --from-literal=AZURE_KEY_VAULT="%KEY_VAULT%" --dry-run=client -o yaml | kubectl apply -f -

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

REM Get load balancer IP
for /f "tokens=*" %%i in ('kubectl get service histocore-streaming -n histocore -o jsonpath="{.status.loadBalancer.ingress[0].ip}"') do set LB_IP=%%i

echo Deployment complete!
echo Resource Group: %RESOURCE_GROUP%
echo Cluster: %CLUSTER_NAME%
echo Redis: %REDIS_HOSTNAME%:%REDIS_PORT%
echo Storage: %STORAGE_ACCOUNT%/%STORAGE_CONTAINER%
echo Load Balancer IP: %LB_IP%
echo.
echo Access via:
echo http://%LB_IP%
echo kubectl port-forward -n histocore svc/histocore-streaming 8000:8000