use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HospitalMetadata {
    pub hospital_id: String,
    pub hospital_type: String,
    pub annual_cases: u32,
    pub cancer_specialties: Vec<String>,
    pub diagnostic_accuracy: f64,
    pub years_experience: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlideQuality {
    pub image_sharpness: f64,
    pub stain_consistency: f64,
    pub label_confidence: f64,
    pub artifact_level: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelUpdate {
    pub hospital_id: String,
    pub parameters: HashMap<String, Vec<f32>>,
    pub quality_metrics: SlideQuality,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregationRequest {
    pub round_number: u32,
    pub cancer_type: String,
    pub model_updates: Vec<ModelUpdate>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AggregationResponse {
    pub round_number: u32,
    pub aggregated_parameters: HashMap<String, Vec<f32>>,
    pub hospital_weights: HashMap<String, f64>,
}

pub struct PathologyFLCoordinator {
    hospitals: Arc<Mutex<HashMap<String, HospitalMetadata>>>,
    round_number: Arc<Mutex<u32>>,
}

impl PathologyFLCoordinator {
    pub fn new() -> Self {
        Self {
            hospitals: Arc::new(Mutex::new(HashMap::new())),
            round_number: Arc::new(Mutex::new(0)),
        }
    }

    pub fn register_hospital(&self, metadata: HospitalMetadata) {
        let mut hospitals = self.hospitals.lock().unwrap();
        hospitals.insert(metadata.hospital_id.clone(), metadata);
        println!("Registered hospital: {}", hospitals.len());
    }

    pub fn calculate_expertise_weight(&self, metadata: &HospitalMetadata, cancer_type: &str) -> f64 {
        let base_weight = match metadata.hospital_type.as_str() {
            "cancer_center" => 2.0,
            "teaching_hospital" => 1.5,
            "community_hospital" => 1.0,
            "rural_hospital" => 0.8,
            _ => 1.0,
        };

        let specialty_bonus = if metadata.cancer_specialties.contains(&cancer_type.to_string()) {
            1.5
        } else {
            1.0
        };

        let volume_factor = (1.0 + (metadata.annual_cases as f64 / 10000.0)).min(2.0);
        let accuracy_factor = metadata.diagnostic_accuracy;
        let experience_factor = (1.0 + (metadata.years_experience as f64 / 20.0)).min(1.5);

        base_weight * specialty_bonus * volume_factor * accuracy_factor * experience_factor
    }

    pub fn calculate_quality_weight(&self, quality: &SlideQuality) -> f64 {
        0.3 * quality.image_sharpness
            + 0.25 * quality.stain_consistency
            + 0.3 * quality.label_confidence
            + 0.15 * (1.0 - quality.artifact_level)
    }

    pub fn aggregate_updates(&self, request: AggregationRequest) -> AggregationResponse {
        let hospitals = self.hospitals.lock().unwrap();
        let mut hospital_weights = HashMap::new();
        let mut weighted_parameters: HashMap<String, Vec<f64>> = HashMap::new();
        let mut total_weight = 0.0;

        // Calculate weights for each hospital
        for update in &request.model_updates {
            if let Some(metadata) = hospitals.get(&update.hospital_id) {
                let expertise_weight = self.calculate_expertise_weight(metadata, &request.cancer_type);
                let quality_weight = self.calculate_quality_weight(&update.quality_metrics);
                let combined_weight = 0.5 * expertise_weight + 0.3 * quality_weight + 0.2;

                hospital_weights.insert(update.hospital_id.clone(), combined_weight);
                total_weight += combined_weight;

                // Weighted parameter accumulation
                for (param_name, param_values) in &update.parameters {
                    let weighted_params = weighted_parameters
                        .entry(param_name.clone())
                        .or_insert_with(|| vec![0.0; param_values.len()]);

                    for (i, &value) in param_values.iter().enumerate() {
                        weighted_params[i] += value as f64 * combined_weight;
                    }
                }
            }
        }

        // Normalize by total weight
        let mut aggregated_parameters = HashMap::new();
        for (param_name, weighted_values) in weighted_parameters {
            let normalized_values: Vec<f32> = weighted_values
                .iter()
                .map(|&v| (v / total_weight) as f32)
                .collect();
            aggregated_parameters.insert(param_name, normalized_values);
        }

        // Normalize hospital weights
        for weight in hospital_weights.values_mut() {
            *weight /= total_weight;
        }

        AggregationResponse {
            round_number: request.round_number,
            aggregated_parameters,
            hospital_weights,
        }
    }

    pub async fn handle_client(&self, mut stream: TcpStream) -> Result<(), Box<dyn std::error::Error>> {
        let mut buffer = vec![0; 4096];
        let n = stream.read(&mut buffer).await?;
        
        if n == 0 {
            return Ok(());
        }

        let request: AggregationRequest = serde_json::from_slice(&buffer[..n])?;
        println!("Processing FL round {} with {} hospitals", 
                request.round_number, request.model_updates.len());

        let response = self.aggregate_updates(request);
        let response_json = serde_json::to_vec(&response)?;

        stream.write_all(&response_json).await?;
        stream.flush().await?;

        Ok(())
    }

    pub async fn start_server(&self, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(addr).await?;
        println!("PathologyFL Coordinator listening on {}", addr);

        loop {
            let (stream, addr) = listener.accept().await?;
            println!("New connection from {}", addr);

            let coordinator = Arc::new(self);
            tokio::spawn(async move {
                if let Err(e) = coordinator.handle_client(stream).await {
                    eprintln!("Error handling client: {}", e);
                }
            });
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coordinator = PathologyFLCoordinator::new();

    // Register some test hospitals
    coordinator.register_hospital(HospitalMetadata {
        hospital_id: "mayo_clinic".to_string(),
        hospital_type: "cancer_center".to_string(),
        annual_cases: 15000,
        cancer_specialties: vec!["breast".to_string(), "lung".to_string()],
        diagnostic_accuracy: 0.96,
        years_experience: 20,
    });

    coordinator.register_hospital(HospitalMetadata {
        hospital_id: "community_hospital".to_string(),
        hospital_type: "community_hospital".to_string(),
        annual_cases: 3000,
        cancer_specialties: vec!["general".to_string()],
        diagnostic_accuracy: 0.87,
        years_experience: 8,
    });

    coordinator.start_server("127.0.0.1:8080").await?;
    Ok(())
}