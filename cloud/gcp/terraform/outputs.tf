# GCP deployment outputs
output "project_id" {
  description = "GCP project ID"
  value       = var.project_id
}

output "gke_cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.histocore.name
}

output "gke_cluster_endpoint" {
  description = "Endpoint of the GKE cluster"
  value       = google_container_cluster.histocore.endpoint
  sensitive   = true
}

output "gke_cluster_ca_certificate" {
  description = "CA certificate of the GKE cluster"
  value       = google_container_cluster.histocore.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.histocore.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.histocore.port
}

output "redis_auth_string" {
  description = "Redis auth string"
  value       = google_redis_instance.histocore.auth_string
  sensitive   = true
}

output "storage_bucket_name" {
  description = "Name of the Cloud Storage bucket"
  value       = google_storage_bucket.histocore_data.name
}

output "postgres_connection_name" {
  description = "PostgreSQL connection name"
  value       = google_sql_database_instance.histocore.connection_name
}

output "postgres_private_ip" {
  description = "PostgreSQL private IP address"
  value       = google_sql_database_instance.histocore.private_ip_address
}

output "load_balancer_ip" {
  description = "Static IP address for load balancer"
  value       = google_compute_global_address.histocore_lb.address
}

output "kms_key_id" {
  description = "KMS key ID for encryption"
  value       = google_kms_crypto_key.histocore.id
}