package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/mux"
)

type HospitalMetadata struct {
	HospitalID         string   `json:"hospital_id"`
	HospitalType       string   `json:"hospital_type"`
	AnnualCases        int      `json:"annual_cases"`
	CancerSpecialties  []string `json:"cancer_specialties"`
	DiagnosticAccuracy float64  `json:"diagnostic_accuracy"`
	YearsExperience    int      `json:"years_experience"`
	LastSeen           time.Time `json:"last_seen"`
	Status             string   `json:"status"`
}

type HospitalRegistry struct {
	hospitals map[string]*HospitalMetadata
	mutex     sync.RWMutex
}

func NewHospitalRegistry() *HospitalRegistry {
	return &HospitalRegistry{
		hospitals: make(map[string]*HospitalMetadata),
	}
}

func (hr *HospitalRegistry) RegisterHospital(hospital *HospitalMetadata) {
	hr.mutex.Lock()
	defer hr.mutex.Unlock()
	
	hospital.LastSeen = time.Now()
	hospital.Status = "active"
	hr.hospitals[hospital.HospitalID] = hospital
	
	log.Printf("Registered hospital: %s (%s)", hospital.HospitalID, hospital.HospitalType)
}

func (hr *HospitalRegistry) GetHospital(hospitalID string) (*HospitalMetadata, bool) {
	hr.mutex.RLock()
	defer hr.mutex.RUnlock()
	
	hospital, exists := hr.hospitals[hospitalID]
	return hospital, exists
}

func (hr *HospitalRegistry) GetAllHospitals() []*HospitalMetadata {
	hr.mutex.RLock()
	defer hr.mutex.RUnlock()
	
	hospitals := make([]*HospitalMetadata, 0, len(hr.hospitals))
	for _, hospital := range hr.hospitals {
		hospitals = append(hospitals, hospital)
	}
	return hospitals
}

func (hr *HospitalRegistry) UpdateHeartbeat(hospitalID string) {
	hr.mutex.Lock()
	defer hr.mutex.Unlock()
	
	if hospital, exists := hr.hospitals[hospitalID]; exists {
		hospital.LastSeen = time.Now()
		hospital.Status = "active"
	}
}

func (hr *HospitalRegistry) CheckInactiveHospitals() {
	hr.mutex.Lock()
	defer hr.mutex.Unlock()
	
	threshold := time.Now().Add(-5 * time.Minute)
	
	for _, hospital := range hr.hospitals {
		if hospital.LastSeen.Before(threshold) {
			hospital.Status = "inactive"
		}
	}
}

type HospitalService struct {
	registry *HospitalRegistry
}

func NewHospitalService() *HospitalService {
	service := &HospitalService{
		registry: NewHospitalRegistry(),
	}
	
	// Start background task to check inactive hospitals
	go func() {
		ticker := time.NewTicker(1 * time.Minute)
		defer ticker.Stop()
		
		for range ticker.C {
			service.registry.CheckInactiveHospitals()
		}
	}()
	
	return service
}

func (hs *HospitalService) registerHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var hospital HospitalMetadata
	if err := json.NewDecoder(r.Body).Decode(&hospital); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Validate hospital data
	if hospital.HospitalID == "" {
		http.Error(w, "Hospital ID required", http.StatusBadRequest)
		return
	}
	
	if hospital.DiagnosticAccuracy < 0 || hospital.DiagnosticAccuracy > 1 {
		http.Error(w, "Diagnostic accuracy must be between 0 and 1", http.StatusBadRequest)
		return
	}
	
	hs.registry.RegisterHospital(&hospital)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "registered",
		"hospital_id": hospital.HospitalID,
	})
}

func (hs *HospitalService) getHospitalHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	hospitalID := vars["id"]
	
	hospital, exists := hs.registry.GetHospital(hospitalID)
	if !exists {
		http.Error(w, "Hospital not found", http.StatusNotFound)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(hospital)
}

func (hs *HospitalService) listHospitalsHandler(w http.ResponseWriter, r *http.Request) {
	hospitals := hs.registry.GetAllHospitals()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"hospitals": hospitals,
		"count": len(hospitals),
	})
}

func (hs *HospitalService) heartbeatHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	vars := mux.Vars(r)
	hospitalID := vars["id"]
	
	hs.registry.UpdateHeartbeat(hospitalID)
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "heartbeat_received",
		"hospital_id": hospitalID,
	})
}

func (hs *HospitalService) healthHandler(w http.ResponseWriter, r *http.Request) {
	hospitals := hs.registry.GetAllHospitals()
	
	activeCount := 0
	inactiveCount := 0
	
	for _, hospital := range hospitals {
		if hospital.Status == "active" {
			activeCount++
		} else {
			inactiveCount++
		}
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "healthy",
		"total_hospitals": len(hospitals),
		"active_hospitals": activeCount,
		"inactive_hospitals": inactiveCount,
		"uptime": time.Since(startTime).String(),
	})
}

var startTime = time.Now()

func main() {
	service := NewHospitalService()
	
	// Add some test hospitals
	service.registry.RegisterHospital(&HospitalMetadata{
		HospitalID:         "mayo_clinic",
		HospitalType:       "cancer_center",
		AnnualCases:        15000,
		CancerSpecialties:  []string{"breast", "lung", "prostate"},
		DiagnosticAccuracy: 0.96,
		YearsExperience:    20,
	})
	
	service.registry.RegisterHospital(&HospitalMetadata{
		HospitalID:         "community_hospital",
		HospitalType:       "community_hospital",
		AnnualCases:        3000,
		CancerSpecialties:  []string{"general"},
		DiagnosticAccuracy: 0.87,
		YearsExperience:    8,
	})
	
	r := mux.NewRouter()
	
	// API routes
	r.HandleFunc("/api/hospitals/register", service.registerHandler).Methods("POST")
	r.HandleFunc("/api/hospitals/{id}", service.getHospitalHandler).Methods("GET")
	r.HandleFunc("/api/hospitals", service.listHospitalsHandler).Methods("GET")
	r.HandleFunc("/api/hospitals/{id}/heartbeat", service.heartbeatHandler).Methods("POST")
	r.HandleFunc("/health", service.healthHandler).Methods("GET")
	
	// CORS middleware
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", "*")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	})
	
	fmt.Println("🏥 Hospital Registry Service starting on :8081")
	fmt.Println("📊 Health check: http://localhost:8081/health")
	fmt.Println("🔗 API docs: http://localhost:8081/api/hospitals")
	
	log.Fatal(http.ListenAndServe(":8081", r))
}