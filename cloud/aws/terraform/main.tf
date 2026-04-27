# AWS infrastructure for HistoCore streaming
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC
resource "aws_vpc" "histocore" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "histocore-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "histocore" {
  vpc_id = aws_vpc.histocore.id

  tags = {
    Name        = "histocore-igw"
    Environment = var.environment
  }
}

# Subnets
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.histocore.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "histocore-public-${count.index + 1}"
    Environment = var.environment
    Type        = "public"
  }
}

resource "aws_subnet" "private" {
  count = 2

  vpc_id            = aws_vpc.histocore.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name        = "histocore-private-${count.index + 1}"
    Environment = var.environment
    Type        = "private"
  }
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.histocore.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.histocore.id
  }

  tags = {
    Name        = "histocore-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name        = "histocore-nat-eip"
    Environment = var.environment
  }
}

resource "aws_nat_gateway" "histocore" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name        = "histocore-nat"
    Environment = var.environment
  }

  depends_on = [aws_internet_gateway.histocore]
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.histocore.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.histocore.id
  }

  tags = {
    Name        = "histocore-private-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name_prefix = "histocore-eks-cluster-"
  vpc_id      = aws_vpc.histocore.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "histocore-eks-cluster-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "eks_nodes" {
  name_prefix = "histocore-eks-nodes-"
  vpc_id      = aws_vpc.histocore.id

  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "histocore-eks-nodes-sg"
    Environment = var.environment
  }
}

# EKS Cluster
resource "aws_eks_cluster" "histocore" {
  name     = "histocore-${var.environment}"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    security_group_ids      = [aws_security_group.eks_cluster.id]
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_service_policy,
  ]

  tags = {
    Name        = "histocore-eks"
    Environment = var.environment
  }
}

# EKS Node Group (CPU)
resource "aws_eks_node_group" "cpu_nodes" {
  cluster_name    = aws_eks_cluster.histocore.name
  node_group_name = "cpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  instance_types = var.cpu_instance_types
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = var.cpu_desired_capacity
    max_size     = var.cpu_max_capacity
    min_size     = var.cpu_min_capacity
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Name        = "histocore-cpu-nodes"
    Environment = var.environment
  }
}

# EKS Node Group (GPU)
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.histocore.name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  instance_types = var.gpu_instance_types
  capacity_type  = "ON_DEMAND"

  scaling_config {
    desired_size = var.gpu_desired_capacity
    max_size     = var.gpu_max_capacity
    min_size     = var.gpu_min_capacity
  }

  update_config {
    max_unavailable = 1
  }

  # GPU-specific taints
  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Name        = "histocore-gpu-nodes"
    Environment = var.environment
    NodeType    = "gpu"
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "histocore" {
  name       = "histocore-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "redis" {
  name_prefix = "histocore-redis-"
  vpc_id      = aws_vpc.histocore.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "histocore-redis-sg"
    Environment = var.environment
  }
}

resource "aws_elasticache_replication_group" "histocore" {
  replication_group_id       = "histocore-${var.environment}"
  description                = "Redis cluster for HistoCore caching"
  
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  
  num_cache_clusters         = 2
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name = aws_elasticache_subnet_group.histocore.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name        = "histocore-redis"
    Environment = var.environment
  }
}

# S3 Bucket for data storage
resource "aws_s3_bucket" "histocore_data" {
  bucket = "histocore-data-${var.environment}-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "histocore-data"
    Environment = var.environment
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "histocore_data" {
  bucket = aws_s3_bucket.histocore_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "histocore_data" {
  bucket = aws_s3_bucket.histocore_data.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "histocore_data" {
  bucket = aws_s3_bucket.histocore_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Application Load Balancer
resource "aws_security_group" "alb" {
  name_prefix = "histocore-alb-"
  vpc_id      = aws_vpc.histocore.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "histocore-alb-sg"
    Environment = var.environment
  }
}

resource "aws_lb" "histocore" {
  name               = "histocore-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Name        = "histocore-alb"
    Environment = var.environment
  }
}