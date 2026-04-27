# Variables for AWS HistoCore deployment

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

# CPU Node Group
variable "cpu_instance_types" {
  description = "Instance types for CPU nodes"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "cpu_desired_capacity" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 2
}

variable "cpu_min_capacity" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_max_capacity" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

# GPU Node Group
variable "gpu_instance_types" {
  description = "Instance types for GPU nodes"
  type        = list(string)
  default     = ["p3.2xlarge", "g4dn.xlarge"]
}

variable "gpu_desired_capacity" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_min_capacity" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_max_capacity" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

# Redis
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

# Tags
variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default = {
    Project = "HistoCore"
    Owner   = "MLOps Team"
  }
}