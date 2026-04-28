-- Medical AI Platform Database Initialization
-- This script sets up the initial database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Users and authentication
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'pathologist',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Organizations/Hospitals
CREATE TABLE IF NOT EXISTS organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100), -- hospital, clinic, lab, etc.
    address TEXT,
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    pacs_vendor VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- DICOM Studies
CREATE TABLE IF NOT EXISTS dicom_studies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    study_instance_uid VARCHAR(255) UNIQUE NOT NULL,
    patient_id VARCHAR(255),
    patient_name VARCHAR(255),
    study_date DATE,
    study_time TIME,
    modality VARCHAR(10),
    study_description TEXT,
    organization_id UUID REFERENCES organizations(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- DICOM Series
CREATE TABLE IF NOT EXISTS dicom_series (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_instance_uid VARCHAR(255) UNIQUE NOT NULL,
    study_id UUID REFERENCES dicom_studies(id) ON DELETE CASCADE,
    series_number INTEGER,
    series_description TEXT,
    modality VARCHAR(10),
    body_part_examined VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- WSI Images
CREATE TABLE IF NOT EXISTS wsi_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    series_id UUID REFERENCES dicom_series(id) ON DELETE CASCADE,
    file_path VARCHAR(500),
    file_size BIGINT,
    image_width INTEGER,
    image_height INTEGER,
    pixel_spacing_x FLOAT,
    pixel_spacing_y FLOAT,
    magnification FLOAT,
    compression VARCHAR(50),
    color_space VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI Analysis Results
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID REFERENCES wsi_images(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    model_version VARCHAR(100),
    analysis_type VARCHAR(100), -- classification, detection, segmentation
    confidence_score FLOAT,
    prediction_class VARCHAR(100),
    prediction_probability JSONB, -- stores class probabilities
    bounding_boxes JSONB, -- stores detection results
    segmentation_mask TEXT, -- path to segmentation mask file
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Annotations (manual/expert)
CREATE TABLE IF NOT EXISTS annotations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID REFERENCES wsi_images(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    annotation_type VARCHAR(50), -- point, rectangle, polygon, etc.
    coordinates JSONB, -- stores annotation coordinates
    label VARCHAR(255),
    confidence FLOAT,
    notes TEXT,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model Performance Metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(100),
    dataset_name VARCHAR(100),
    metric_name VARCHAR(100), -- accuracy, sensitivity, specificity, auc, etc.
    metric_value FLOAT,
    sample_count INTEGER,
    evaluation_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit Log
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100), -- login, analyze, annotate, etc.
    resource_type VARCHAR(100), -- image, study, annotation, etc.
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System Configuration
CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_dicom_studies_uid ON dicom_studies(study_instance_uid);
CREATE INDEX IF NOT EXISTS idx_dicom_studies_patient ON dicom_studies(patient_id);
CREATE INDEX IF NOT EXISTS idx_dicom_series_uid ON dicom_series(series_instance_uid);
CREATE INDEX IF NOT EXISTS idx_analysis_results_image ON analysis_results(image_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_user ON analysis_results(user_id);
CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id);
CREATE INDEX IF NOT EXISTS idx_annotations_user ON annotations(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_created ON audit_log(created_at);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_dicom_studies_search ON dicom_studies USING gin(to_tsvector('english', study_description));
CREATE INDEX IF NOT EXISTS idx_annotations_search ON annotations USING gin(to_tsvector('english', notes));

-- Insert default system configuration
INSERT INTO system_config (key, value, description) VALUES
    ('model_version', '"1.0.0"', 'Current AI model version'),
    ('max_upload_size_mb', '100', 'Maximum file upload size in MB'),
    ('supported_formats', '["svs", "ndpi", "tiff", "dcm"]', 'Supported image formats'),
    ('default_confidence_threshold', '0.8', 'Default confidence threshold for predictions'),
    ('enable_gpu_acceleration', 'true', 'Enable GPU acceleration for inference')
ON CONFLICT (key) DO NOTHING;

-- Create default admin user (password: admin123)
-- Note: In production, change this password immediately
INSERT INTO users (email, username, password_hash, full_name, role, is_active, is_verified) VALUES
    ('admin@medical-ai.com', 'admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBdXwtO5S5EM.S', 'System Administrator', 'admin', true, true)
ON CONFLICT (email) DO NOTHING;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_annotations_updated_at BEFORE UPDATE ON annotations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();