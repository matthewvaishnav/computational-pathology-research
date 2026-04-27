# GCP infrastructure for HistoCore streaming
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.gcp_region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",
    "compute.googleapis.com",
    "redis.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com"
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = true
}

# VPC Network
resource "google_compute_network" "histocore" {
  name                    = "histocore-vpc"
  auto_create_subnetworks = false
  project                 = var.project_id

  depends_on = [google_project_service.required_apis]
}

# Subnet for GKE cluster
resource "google_compute_subnetwork" "gke" {
  name          = "histocore-gke-subnet"
  ip_cidr_range = var.gke_subnet_cidr
  region        = var.gcp_region
  network       = google_compute_network.histocore.id
  project       = var.project_id

  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = var.gke_pods_cidr
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = var.gke_services_cidr
  }
}

# Firewall rules
resource "google_compute_firewall" "allow_internal" {
  name    = "histocore-allow-internal"
  network = google_compute_network.histocore.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "icmp"
  }

  source_ranges = [var.gke_subnet_cidr, var.gke_pods_cidr, var.gke_services_cidr]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "histocore-allow-ssh"
  network = google_compute_network.histocore.name
  project = var.project_id

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh-allowed"]
}

# GKE Cluster
resource "google_container_cluster" "histocore" {
  name     = "histocore-${var.environment}"
  location = var.gcp_region
  project  = var.project_id

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.histocore.name
  subnetwork = google_compute_subnetwork.gke.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }

  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Enable network policy
  network_policy {
    enabled = true
  }

  addons_config {
    network_policy_config {
      disabled = false
    }
  }

  # Logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  depends_on = [google_project_service.required_apis]
}

# System node pool
resource "google_container_node_pool" "system" {
  name       = "system-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.histocore.name
  project    = var.project_id
  node_count = var.system_node_count

  node_config {
    preemptible  = false
    machine_type = var.system_machine_type

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node-type   = "system"
    }

    tags = ["gke-node", "histocore-${var.environment}"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# CPU node pool with autoscaling
resource "google_container_node_pool" "cpu" {
  name     = "cpu-pool"
  location = var.gcp_region
  cluster  = google_container_cluster.histocore.name
  project  = var.project_id

  autoscaling {
    min_node_count = var.cpu_min_nodes
    max_node_count = var.cpu_max_nodes
  }

  node_config {
    preemptible  = false
    machine_type = var.cpu_machine_type

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node-type   = "cpu"
    }

    tags = ["gke-node", "histocore-${var.environment}"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# GPU node pool
resource "google_container_node_pool" "gpu" {
  name       = "gpu-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.histocore.name
  project    = var.project_id
  node_count = var.gpu_node_count

  node_config {
    preemptible  = false
    machine_type = var.gpu_machine_type

    # Attach GPU
    guest_accelerator {
      type  = var.gpu_type
      count = var.gpu_count_per_node
    }

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      node-type   = "gpu"
    }

    tags = ["gke-node", "histocore-${var.environment}"]

    # GPU nodes need special taints
    taint {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "histocore-gke-nodes"
  display_name = "HistoCore GKE Nodes Service Account"
  project      = var.project_id
}

# IAM bindings for GKE nodes
resource "google_project_iam_member" "gke_nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Cloud Storage bucket
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "histocore_data" {
  name     = "histocore-data-${var.environment}-${random_id.bucket_suffix.hex}"
  location = var.gcp_region
  project  = var.project_id

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.histocore.id
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}

# Cloud KMS for encryption
resource "google_kms_key_ring" "histocore" {
  name     = "histocore-keyring"
  location = var.gcp_region
  project  = var.project_id
}

resource "google_kms_crypto_key" "histocore" {
  name     = "histocore-key"
  key_ring = google_kms_key_ring.histocore.id
  project  = var.project_id

  rotation_period = "7776000s" # 90 days
}

# Cloud Memorystore (Redis)
resource "google_redis_instance" "histocore" {
  name           = "histocore-redis-${var.environment}"
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_size_gb
  region         = var.gcp_region
  project        = var.project_id

  authorized_network = google_compute_network.histocore.id

  redis_version     = "REDIS_7_0"
  display_name      = "HistoCore Redis Cache"
  reserved_ip_range = var.redis_reserved_ip_range

  auth_enabled   = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"

  depends_on = [google_project_service.required_apis]
}

# Cloud SQL (PostgreSQL) for metadata
resource "google_sql_database_instance" "histocore" {
  name             = "histocore-db-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.gcp_region
  project          = var.project_id

  settings {
    tier = var.postgres_tier

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.histocore.id
    }

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }

    database_flags {
      name  = "log_connections"
      value = "on"
    }

    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
  }

  depends_on = [google_service_networking_connection.private_vpc_connection]
}

resource "google_sql_database" "histocore" {
  name     = "histocore"
  instance = google_sql_database_instance.histocore.name
  project  = var.project_id
}

resource "google_sql_user" "histocore" {
  name     = "histocore"
  instance = google_sql_database_instance.histocore.name
  password = var.postgres_password
  project  = var.project_id
}

# Private service connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.histocore.id
  project       = var.project_id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.histocore.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Static IP for load balancer
resource "google_compute_global_address" "histocore_lb" {
  name    = "histocore-lb-ip"
  project = var.project_id
}