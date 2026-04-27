# iOS Native Components

CoreML inference native module for React Native.

## Architecture

### CoreMLInference Module
- **CoreMLInference.swift**: Swift implementation
- **CoreMLInference.m**: Objective-C bridge
- Uses Vision framework for image processing
- Uses CoreML for on-device inference

## Features

- **Model Loading**: Load `.mlmodel` from app bundle
- **Image Preprocessing**: Resize and convert to CVPixelBuffer
- **Inference**: Run CoreML inference on-device
- **Postprocessing**: Sort and return top predictions
- **Performance**: Optimized for Neural Engine

## Usage

### JavaScript/TypeScript

```typescript
import { NativeModules } from 'react-native';

const { CoreMLInference } = NativeModules;

// Load model
await CoreMLInference.loadModel('medical_model');

// Run inference
const result = await CoreMLInference.predict('file:///path/to/image.jpg');

console.log(result.predictions);
// [
//   { label: 'Benign', confidence: 0.85 },
//   { label: 'Malignant', confidence: 0.12 },
//   { label: 'Uncertain', confidence: 0.03 }
// ]

console.log(result.inference_time_ms); // 45.2
```

### Model Integration

1. **Convert PyTorch to CoreML**:
```python
from src.mobile_edge.optimization.coreml_converter import convert_medical_model_coreml

convert_medical_model_coreml(
    model=pytorch_model,
    input_shape=(1, 3, 224, 224),
    save_path='medical_model.mlmodel',
    class_labels=['Benign', 'Malignant', 'Uncertain']
)
```

2. **Add to Xcode Project**:
   - Drag `medical_model.mlmodel` into Xcode
   - Ensure "Target Membership" includes MedicalAI
   - Xcode will compile to `.mlmodelc`

3. **Verify Model**:
   - Open `.mlmodel` in Xcode
   - Check input/output specifications
   - Test with sample data

## Performance

### Target Metrics
- Model load time: <1s
- Inference time: <100ms
- Memory usage: <150MB
- Neural Engine utilization: >80%

### Optimization
- FP16 precision (2x compression)
- Neural Engine optimization
- Batch size = 1 (single image)
- Async inference on background queue

## Testing

### Unit Tests
```bash
cd ios
xcodebuild test -workspace MedicalAI.xcworkspace -scheme MedicalAI -destination 'platform=iOS Simulator,name=iPhone 14'
```

### Manual Testing
1. Run app in simulator
2. Capture/select image
3. Verify inference runs
4. Check console for timing

## Troubleshooting

### Model Not Found
- Verify `.mlmodel` is in Xcode project
- Check "Target Membership"
- Clean build folder (Cmd+Shift+K)

### Inference Fails
- Check input image format
- Verify model input shape (224x224)
- Check console for errors

### Slow Performance
- Ensure Neural Engine is used (check model compatibility)
- Verify FP16 precision
- Profile with Instruments

## Requirements

- iOS 14.0+
- Xcode 13.0+
- CoreML framework
- Vision framework

## License

Medical Use Only - Research Purposes
