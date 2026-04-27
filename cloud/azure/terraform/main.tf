# Azure infrastructure for HistoCore streaming
terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "histocore" {
  name     = "histocore-${var.environment}"
  location = var.azure_region

  tags = {
    Environment = var.environment
    Project     = "HistoCore"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "histocore" {
  name                = "histocore-vnet"
  address_space       = [var.vnet_cidr]
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name

  tags = {
    Environment = var.environment
  }
}

# Subnets
resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = azurerm_resource_group.histocore.name
  virtual_network_name = azurerm_virtual_network.histocore.name
  address_prefixes     = [var.aks_subnet_cidr]
}

resource "azurerm_subnet" "services" {
  name                 = "services-subnet"
  resource_group_name  = azurerm_resource_group.histocore.name
  virtual_network_name = azurerm_virtual_network.histocore.name
  address_prefixes     = [var.services_subnet_cidr]
}

# Network Security Group
resource "azurerm_network_security_group" "aks" {
  name                = "histocore-aks-nsg"
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name

  security_rule {
    name                       = "AllowHTTPS"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  security_rule {
    name                       = "AllowHTTP"
    priority                   = 1002
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "80"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  tags = {
    Environment = var.environment
  }
}

resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks.id
  network_security_group_id = azurerm_network_security_group.aks.id
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "histocore" {
  name                = "histocore-${var.environment}"
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name
  dns_prefix          = "histocore-${var.environment}"
  kubernetes_version  = var.kubernetes_version

  default_node_pool {
    name           = "system"
    node_count     = var.system_node_count
    vm_size        = var.system_vm_size
    vnet_subnet_id = azurerm_subnet.aks.id
    
    upgrade_settings {
      max_surge = "10%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.histocore.id
  }

  tags = {
    Environment = var.environment
  }
}

# GPU Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.histocore.id
  vm_size               = var.gpu_vm_size
  node_count            = var.gpu_node_count
  vnet_subnet_id        = azurerm_subnet.aks.id

  node_taints = ["nvidia.com/gpu=true:NoSchedule"]

  upgrade_settings {
    max_surge = "10%"
  }

  tags = {
    Environment = var.environment
    NodeType    = "gpu"
  }
}

# CPU Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "cpu" {
  name                  = "cpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.histocore.id
  vm_size               = var.cpu_vm_size
  node_count            = var.cpu_node_count
  vnet_subnet_id        = azurerm_subnet.aks.id

  enable_auto_scaling = true
  min_count          = var.cpu_min_count
  max_count          = var.cpu_max_count

  upgrade_settings {
    max_surge = "10%"
  }

  tags = {
    Environment = var.environment
    NodeType    = "cpu"
  }
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "histocore" {
  name                = "histocore-redis-${var.environment}"
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"
  
  subnet_id = azurerm_subnet.services.id

  redis_configuration {
    enable_authentication = true
  }

  tags = {
    Environment = var.environment
  }
}

# Storage Account
resource "random_id" "storage_suffix" {
  byte_length = 4
}

resource "azurerm_storage_account" "histocore" {
  name                     = "histocore${var.environment}${random_id.storage_suffix.hex}"
  resource_group_name      = azurerm_resource_group.histocore.name
  location                 = azurerm_resource_group.histocore.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  blob_properties {
    versioning_enabled = true
  }

  tags = {
    Environment = var.environment
  }
}

resource "azurerm_storage_container" "data" {
  name                  = "histocore-data"
  storage_account_name  = azurerm_storage_account.histocore.name
  container_access_type = "private"
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "histocore" {
  name                = "histocore-logs-${var.environment}"
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = {
    Environment = var.environment
  }
}

# Application Insights
resource "azurerm_application_insights" "histocore" {
  name                = "histocore-insights-${var.environment}"
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name
  workspace_id        = azurerm_log_analytics_workspace.histocore.id
  application_type    = "web"

  tags = {
    Environment = var.environment
  }
}

# Public IP for Load Balancer
resource "azurerm_public_ip" "histocore" {
  name                = "histocore-lb-ip"
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name
  allocation_method   = "Static"
  sku                 = "Standard"

  tags = {
    Environment = var.environment
  }
}

# Key Vault
resource "azurerm_key_vault" "histocore" {
  name                = "histocore-kv-${var.environment}-${random_id.storage_suffix.hex}"
  location            = azurerm_resource_group.histocore.location
  resource_group_name = azurerm_resource_group.histocore.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    key_permissions = [
      "Get", "List", "Create", "Delete", "Update"
    ]

    secret_permissions = [
      "Get", "List", "Set", "Delete"
    ]
  }

  # AKS access policy
  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = azurerm_kubernetes_cluster.histocore.identity[0].principal_id

    secret_permissions = [
      "Get", "List"
    ]
  }

  tags = {
    Environment = var.environment
  }
}

# Store Redis connection string in Key Vault
resource "azurerm_key_vault_secret" "redis_connection" {
  name         = "redis-connection-string"
  value        = azurerm_redis_cache.histocore.primary_connection_string
  key_vault_id = azurerm_key_vault.histocore.id
}

# Store storage account key in Key Vault
resource "azurerm_key_vault_secret" "storage_key" {
  name         = "storage-account-key"
  value        = azurerm_storage_account.histocore.primary_access_key
  key_vault_id = azurerm_key_vault.histocore.id
}

# Data source for current client config
data "azurerm_client_config" "current" {}