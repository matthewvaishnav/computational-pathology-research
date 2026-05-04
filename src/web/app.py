"""
HistoCore Web Interface
Browser-based WSI analysis
"""

import os
import json
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'svs', 'tiff', 'tif', 'ndpi', 'vms', 'vmu', 'scn'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'size': os.path.getsize(filepath)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Analysis configuration
    config = {
        'patch_size': data.get('patch_size', 256),
        'model': data.get('model', 'resnet50'),
        'tissue_threshold': data.get('tissue_threshold', 0.5),
        'use_gpu': data.get('use_gpu', True)
    }
    
    # Start analysis (demo mode)
    analysis_id = f"analysis_{int(time.time())}"
    
    try:
        # Demo analysis - replace with real processing
        result = run_demo_analysis(filepath, config)
        
        # Save results
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return jsonify({
            'success': True,
            'analysis_id': analysis_id,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status/<analysis_id>')
def get_status(analysis_id):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
    
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            result = json.load(f)
        return jsonify({'status': 'complete', 'result': result})
    
    return jsonify({'status': 'processing'})

@app.route('/download/<analysis_id>')
def download_results(analysis_id):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], f"{analysis_id}.json")
    
    if os.path.exists(result_path):
        return send_file(result_path, as_attachment=True)
    
    return jsonify({'error': 'Results not found'}), 404

def run_demo_analysis(filepath, config):
    """Demo analysis - replace with real HistoCore processing"""
    
    # Simulate processing time
    time.sleep(2)
    
    # Generate demo results
    result = {
        'file_path': filepath,
        'config': config,
        'prediction': np.random.choice(['Normal', 'Tumor']),
        'probability': float(np.random.random()),
        'confidence': float(np.random.uniform(0.7, 0.95)),
        'patches_analyzed': np.random.randint(500, 2000),
        'processing_time': 2.3,
        'attention_weights': np.random.random((10, 10)).tolist()
    }
    
    return result

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)