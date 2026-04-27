# GCP deployment variables
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "gcp_region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

# Network configuration
variable "gke_subnet_cidr" {
  description = "CIDR block for GKE subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "gke_pods_cidr" {
  description = "CIDR block for GKE pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "gke_services_cidr" {
  description = "CIDR block for GKE services"
  type        = string
  default     = "10.2.0.0/16"
}

# System node pool
variable "system_node_count" {
  description = "Number of nodes in system node pool"
  type        = number
  default     = 2
}

variable "system_machine_type" {
  description = "Machine type for system nodes"
  type        = string
  default     = "e2-standard-2"
}

# CPU node pool
variable "cpu_min_nodes" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_max_nodes" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_machine_type" {
  description = "Machine type for CPU nodes"
  type        = string
  default     = "e2-standard-4"
}

# GPU node pool
variable "gpu_node_count" {
  description = "Number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_machine_type" {
  description = "Machine type for GPU nodes"
  type        = string
  default     = "n1-standard-4"
}

variable "gpu_type" {
  description = "GPU type (nvidia-tesla-t4, nvidia-tesla-v100, etc.)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count_per_node" {
  description = "Number of GPUs per node"
  type        = number
  default     = 1
}

# Redis configuration
variable "redis_tier" {
  description = "Redis tier (BASIC or STANDARD_HA)"
  type        = string
  default     = "STANDARD_HA"
}

variable "redis_memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
}

variable "redis_reserved_ip_range" {
  description = "Reserved IP range for Redis"
  type        = string
  default     = "10.3.0.0/29"
}

# PostgreSQL configuration
variable "postgres_tier" {
  description = "PostgreSQL instance tier"
  type        = string
  default     = "db-f1-micro"
}

variable "postgres_password" {
  description = "PostgreSQL password"
  type        = string
  sensitive   = true
}