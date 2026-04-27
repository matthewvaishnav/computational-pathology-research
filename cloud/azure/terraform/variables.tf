# Azure deployment variables
variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "azure_region" {
  description = "Azure region for resources"
  type        = string
  default     = "East US"
}

variable "vnet_cidr" {
  description = "CIDR block for the virtual network"
  type        = string
  default     = "10.0.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "CIDR block for AKS subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "services_subnet_cidr" {
  description = "CIDR block for services subnet"
  type        = string
  default     = "10.0.2.0/24"
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS cluster"
  type        = string
  default     = "1.28"
}

# System node pool
variable "system_node_count" {
  description = "Number of nodes in system node pool"
  type        = number
  default     = 2
}

variable "system_vm_size" {
  description = "VM size for system nodes"
  type        = string
  default     = "Standard_D2s_v3"
}

# CPU node pool
variable "cpu_node_count" {
  description = "Initial number of CPU nodes"
  type        = number
  default     = 2
}

variable "cpu_min_count" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_max_count" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_vm_size" {
  description = "VM size for CPU nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

# GPU node pool
variable "gpu_node_count" {
  description = "Number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_vm_size" {
  description = "VM size for GPU nodes (NC-series for NVIDIA GPUs)"
  type        = string
  default     = "Standard_NC6s_v3"
}

# Redis configuration
variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Redis cache family"
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "Redis cache SKU"
  type        = string
  default     = "Standard"
}