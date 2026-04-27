# Azure deployment outputs
output "resource_group_name" {
  description = "Name of the resource group"
  value       = azurerm_resource_group.histocore.name
}

output "aks_cluster_name" {
  description = "Name of the AKS cluster"
  value       = azurerm_kubernetes_cluster.histocore.name
}

output "aks_cluster_id" {
  description = "ID of the AKS cluster"
  value       = azurerm_kubernetes_cluster.histocore.id
}

output "redis_hostname" {
  description = "Redis cache hostname"
  value       = azurerm_redis_cache.histocore.hostname
}

output "redis_port" {
  description = "Redis cache port"
  value       = azurerm_redis_cache.histocore.port
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = azurerm_storage_account.histocore.name
}

output "storage_container_name" {
  description = "Name of the storage container"
  value       = azurerm_storage_container.data.name
}

output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = azurerm_key_vault.histocore.name
}

output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics workspace"
  value       = azurerm_log_analytics_workspace.histocore.id
}

output "application_insights_instrumentation_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.histocore.instrumentation_key
  sensitive   = true
}

output "public_ip_address" {
  description = "Public IP address for load balancer"
  value       = azurerm_public_ip.histocore.ip_address
}