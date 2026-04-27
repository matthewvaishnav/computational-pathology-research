# Clinical Workflow Integration

This module integrates the annotation interface with the clinical workflow, connecting:
- Active learning system → Annotation queue
- PACS → Slide retrieval  
- WSI streaming → Real-time viewing
- Notification system → Pathologist alerts
- Expert feedback → Model retraining

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Clinical Workflow Integrator                   │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │ Active Learning  │  │  PACS Connector  │  │ Notification │ │
│  │   Connector      │  │                  │  │   Service    │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│           │                     │                     │         │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
    ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
    │   Active     │      │     PACS     │     │ Pathologists │
    │  Learning    │      │   System     │     │              │
    │   System     │      │              │     │              │
    └──────────────┘      └──────────────┘     └──────────────┘
```

## Components

### 1. ClinicalWorkflowIntegrator

Main orchestrator that coordinates all workflow components.

**Key Features:**
- Automatic case processing from PACS to annotation
- High-priority case monitoring and urgent notifications
- Retraining trigger detection
- End-to-end workflow testing

**Usage:**
```python
from src.annotation_interface.workflow import ClinicalWorkflowIntegrator
from src.continuous_learning.active_learning import ActiveLearningSystem
from src.clinical.pacs.pacs_adapter import PACSAdapter
from src.annotation_interface.workflow import NotificationService

# Initialize components
al_system = ActiveLearningSystem()
pacs_adapter = PACSAdapter()
notification_service = NotificationService()

# Create integrator
integrator = ClinicalWorkflowIntegrator(
    active_learning_system=al_system,
    pacs_adapter=pacs_adapter,
    notification_service=notification_service
)

# Start workflow
await integrator.start()

# Process new case
result = await integrator.process_new_case(
    study_uid="1.2.3.4.5",
    slide_id="slide_001",
    patient_id="PAT001",
    priority=0.8,
    expert_id="pathologist_1"
)
```

### 2. ActiveLearningConnector

Connects active learning system to annotation interface.

**Key Features:**
- Automatic polling for uncertain cases
- Case submission to annotation queue
- Expert feedback collection
- Retraining coordination

**Workflow:**
1. Poll active learning system for high-uncertainty cases
2. Convert AL cases to annotation queue items
3. Submit to annotation interface
4. Collect expert annotations
5. Send feedback to AL system for retraining

**Usage:**
```python
from src.annotation_interface.workflow import ActiveLearningConnector

connector = ActiveLearningConnector(
    active_learning_system=al_system,
    auto_queue_threshold=0.85,
    poll_interval=60.0
)

await connector.start()

# Submit expert feedback
success = await connector.submit_expert_feedback(
    task_id="task_123",
    expert_id="pathologist_1",
    diagnosis="malignant",
    confidence=0.95,
    annotation_time=120.0
)
```

### 3. PACSConnector

Integrates with PACS for slide retrieval.

**Key Features:**
- Automatic slide retrieval from PACS
- Local caching of retrieved slides
- Study querying for patients
- Integration with existing PACS adapter

**Workflow:**
1. Receive annotation task with study UID
2. Retrieve slide from PACS
3. Cache locally for fast access
4. Provide slide to annotation interface

**Usage:**
```python
from src.annotation_interface.workflow import PACSConnector

connector = PACSConnector(
    pacs_adapter=pacs_adapter,
    cache_directory="./pacs_cache"
)

# Retrieve slide for annotation
slide_info = await connector.retrieve_slide_for_annotation(
    study_uid="1.2.3.4.5",
    slide_id="slide_001"
)

# Query studies for patient
studies = await connector.query_studies_for_patient("PAT001")
```

### 4. NotificationService

Sends notifications to pathologists.

**Key Features:**
- Email notifications (SMTP)
- Webhook notifications
- Priority-based notification routing
- Notification history tracking

**Notification Types:**
- New annotation task
- Task assignment
- Urgent case alert

**Usage:**
```python
from src.annotation_interface.workflow import NotificationService

service = NotificationService(
    smtp_host="smtp.hospital.org",
    smtp_port=587,
    smtp_username="histocore",
    smtp_password="password"
)

# Notify about new task
await service.notify_new_annotation_task(
    expert_id="pathologist_1",
    task_id="task_123",
    slide_id="slide_001",
    priority=0.8,
    uncertainty_score=0.9
)

# Register webhook
service.register_webhook(
    expert_id="pathologist_1",
    webhook_url="https://hospital.org/webhook"
)

# Notify urgent case
await service.notify_urgent_case(
    expert_ids=["pathologist_1", "pathologist_2"],
    task_id="task_123",
    slide_id="slide_001",
    reason="Very high uncertainty (0.95)"
)
```

## Integration Points

### 1. Active Learning → Annotation Queue

**Flow:**
```
Active Learning System
    ↓ (identifies uncertain case)
ActiveLearningConnector
    ↓ (converts to annotation task)
Annotation Interface Queue
    ↓ (displays to pathologist)
Expert Annotation
    ↓ (feedback)
Active Learning System
    ↓ (triggers retraining)
Model Update
```

**Configuration:**
- `auto_queue_threshold`: Uncertainty threshold for automatic queuing (default: 0.85)
- `poll_interval`: Seconds between polling for new cases (default: 60.0)

### 2. PACS → Slide Retrieval

**Flow:**
```
Annotation Task Created
    ↓ (contains study UID)
PACSConnector
    ↓ (retrieves from PACS)
Local Cache
    ↓ (provides to interface)
WSI Streaming
    ↓ (serves tiles)
Annotation Interface
```

**Configuration:**
- `cache_directory`: Local directory for caching slides (default: "./pacs_cache")

### 3. WSI Streaming → Annotation Interface

**Integration:**
The annotation interface tile endpoint (`/api/slides/{slide_id}/tile/{z}/{x}/{y}`) should be connected to the existing WSI streaming infrastructure:

```python
from src.streaming.wsi_stream_reader import WSIStreamReader

# In annotation_api.py
@app.get("/api/slides/{slide_id}/tile/{z}/{x}/{y}")
async def get_slide_tile(slide_id: str, z: int, x: int, y: int):
    # Get slide path from database
    slide_info = slides_db.get(slide_id)
    
    # Use WSI streaming to get tile
    reader = WSIStreamReader(slide_info.image_path)
    tile = reader.get_tile(z, x, y)
    
    return Response(content=tile, media_type="image/jpeg")
```

### 4. Notifications → Pathologists

**Channels:**
- **Email**: SMTP-based email notifications
- **Webhook**: HTTP POST to registered URLs
- **SMS**: (Future) SMS notifications via Twilio
- **Push**: (Future) Mobile push notifications

**Priority Levels:**
- **URGENT**: High uncertainty (>0.9) + high priority (>0.8)
- **HIGH**: Uncertainty >0.85 or priority >0.6
- **NORMAL**: Uncertainty >0.7 or priority >0.4
- **LOW**: All other cases

## Configuration

### Environment Variables

```bash
# SMTP Configuration
SMTP_HOST=smtp.hospital.org
SMTP_PORT=587
SMTP_USERNAME=histocore
SMTP_PASSWORD=password
FROM_EMAIL=noreply@histocore.ai

# Active Learning
AL_UNCERTAINTY_THRESHOLD=0.85
AL_POLL_INTERVAL=60.0
AL_MIN_ANNOTATIONS_FOR_RETRAINING=50

# PACS
PACS_CACHE_DIRECTORY=./pacs_cache
PACS_CONFIG_PROFILE=production

# Workflow
MAX_CONCURRENT_WORKFLOWS=10
WORKFLOW_MONITOR_INTERVAL=30.0
```

### Configuration File

```yaml
# config/workflow.yaml
active_learning:
  uncertainty_threshold: 0.85
  poll_interval: 60.0
  min_annotations_for_retraining: 50

pacs:
  cache_directory: ./pacs_cache
  config_profile: production

notifications:
  smtp:
    host: smtp.hospital.org
    port: 587
    username: histocore
    password: ${SMTP_PASSWORD}
  from_email: noreply@histocore.ai

workflow:
  max_concurrent: 10
  monitor_interval: 30.0
  urgent_threshold: 0.95
```

## Testing

Run the integration tests:

```bash
# Run all workflow tests
pytest tests/test_clinical_workflow.py -v

# Run specific test class
pytest tests/test_clinical_workflow.py::TestClinicalWorkflowIntegrator -v

# Run with coverage
pytest tests/test_clinical_workflow.py --cov=src/annotation_interface/workflow
```

## Monitoring

### Workflow Statistics

```python
# Get comprehensive statistics
stats = integrator.get_workflow_statistics()

# Returns:
{
    'status': 'running',
    'active_learning': {
        'cases_identified': 150,
        'annotations_received': 75,
        'retraining_triggered': 3
    },
    'pacs': {
        'cached_slides': 50,
        'cache_directory': './pacs_cache'
    },
    'notifications': {
        'total_notifications_sent': 200,
        'registered_webhooks': 5
    }
}
```

### Health Check

```python
# Test all integrations
results = await integrator.test_workflow_integration()

# Returns:
{
    'timestamp': '2024-01-15T10:30:00',
    'tests': {
        'pacs_connection': {'success': True, 'message': 'Connected'},
        'active_learning': {'success': True, 'statistics': {...}},
        'notifications': {'success': True, 'statistics': {...}}
    },
    'overall_success': True
}
```

## Deployment

### Production Deployment

1. **Configure PACS Connection:**
   ```python
   pacs_adapter = PACSAdapter(
       config_profile="production",
       config_directory="/etc/histocore/pacs"
   )
   ```

2. **Configure SMTP:**
   ```python
   notification_service = NotificationService(
       smtp_host=os.getenv("SMTP_HOST"),
       smtp_port=int(os.getenv("SMTP_PORT")),
       smtp_username=os.getenv("SMTP_USERNAME"),
       smtp_password=os.getenv("SMTP_PASSWORD")
   )
   ```

3. **Start Integrator:**
   ```python
   integrator = ClinicalWorkflowIntegrator(
       active_learning_system=al_system,
       pacs_adapter=pacs_adapter,
       notification_service=notification_service,
       auto_start=True
   )
   ```

4. **Monitor Health:**
   ```python
   # Periodic health checks
   while True:
       results = await integrator.test_workflow_integration()
       if not results['overall_success']:
           logger.error("Workflow integration health check failed")
       await asyncio.sleep(300)  # Check every 5 minutes
   ```

## Troubleshooting

### Common Issues

**1. PACS Connection Failures**
- Check PACS adapter configuration
- Verify network connectivity
- Check DICOM credentials

**2. Notification Delivery Failures**
- Verify SMTP configuration
- Check email server logs
- Test webhook URLs manually

**3. Active Learning Not Queuing Cases**
- Check uncertainty threshold
- Verify AL system is running
- Check poll interval

**4. Slide Retrieval Timeouts**
- Increase PACS retrieval timeout
- Check network bandwidth
- Verify PACS server performance

## Future Enhancements

1. **Real-time WebSocket Notifications**: Push notifications to web interface
2. **SMS Notifications**: Integrate with Twilio for SMS alerts
3. **Mobile Push Notifications**: iOS/Android push notifications
4. **Advanced Priority Algorithms**: ML-based priority scoring
5. **Workflow Analytics Dashboard**: Real-time workflow monitoring
6. **Multi-site Coordination**: Coordinate annotation across multiple hospitals
7. **Automated Quality Control**: Automatic annotation quality checks
8. **Expert Consensus**: Multi-expert annotation for difficult cases

## License

Copyright (c) 2024 HistoCore. All rights reserved.
