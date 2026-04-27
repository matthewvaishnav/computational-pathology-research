# Android Native Components

TensorFlow Lite inference native module for React Native.

## Architecture

### TFLiteInference Module
- **TFLiteInference.java**: Java implementation
- **TFLiteInferencePackage.java**: React Native package
- Uses TensorFlow Lite for on-device inference
- Uses NNAPI for hardware acceleration

## Features

- **Model Loading**: Load `.tflite` from assets
- **Image Preprocessing**: Resize and normalize
- **Inference**: Run TFLite inference on-device
- **Postprocessing**: Sort and return top predictions
- **Hardware Acceleration**: NNAPI, GPU delegate

## Usage

### JavaScript/TypeScript

```typescript
import { NativeModules } from 'react-native';

const { TFLiteInference } = NativeModules;

// Load model
await TFLiteInference.loadModel('medical_model');

// Run inference
const result = await TFLiteInference.predict('file:///path/to/image.jpg');

console.log(result.predictions);
// [
//   { label: 'Benign', confidence: 0.85 },
//   { label: 'Malignant', confidence: 0.12 },
//   { label: 'Uncertain', confidence: 0.03 }
// ]

console.log(result.inference_time_ms); // 52.3
```

### Model Integration

1. **Convert PyTorch to TFLite**:
```python
from src.mobile_edge.optimization.mobile_inference import convert_pytorch_to_tflite

convert_pytorch_to_tflite(
    model=pytorch_model,
    input_shape=(1, 3, 224, 224),
    save_path='medical_model.tflite',
    quantize=False  # Set True for INT8 quantization
)
```

2. **Add to Android Project**:
   - Copy `medical_model.tflite` to `android/app/src/main/assets/`
   - Model will be loaded from assets at runtime

3. **Verify Model**:
```bash
# Check model info
python -c "
import tensorflow as tf
interpreter = tf.lite.Interpreter('medical_model.tflite')
print(interpreter.get_input_details())
print(interpreter.get_output_details())
"
```

## Performance

### Target Metrics
- Model load time: <1s
- Inference time: <150ms (CPU), <50ms (GPU/NNAPI)
- Memory usage: <200MB
- NNAPI utilization: >70%

### Optimization
- NNAPI acceleration (Android 8.1+)
- GPU delegate (optional)
- INT8 quantization (4x compression)
- 4 CPU threads

## Hardware Acceleration

### NNAPI (Android Neural Networks API)
- Enabled by default
- Uses device-specific accelerators (NPU, DSP, GPU)
- Fallback to CPU if unavailable

### GPU Delegate
```java
// Enable GPU delegate
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(new GpuDelegate());
interpreter = new Interpreter(modelBuffer, options);
```

### Hexagon Delegate (Qualcomm)
```java
// Enable Hexagon DSP
Interpreter.Options options = new Interpreter.Options();
options.addDelegate(new HexagonDelegate());
interpreter = new Interpreter(modelBuffer, options);
```

## Testing

### Unit Tests
```bash
cd android
./gradlew test
```

### Instrumentation Tests
```bash
./gradlew connectedAndroidTest
```

### Manual Testing
1. Run app on device/emulator
2. Capture/select image
3. Verify inference runs
4. Check logcat for timing

## Troubleshooting

### Model Not Found
- Verify `.tflite` is in `assets/` folder
- Check file name matches `loadModel()` call
- Clean and rebuild project

### Inference Fails
- Check input image format
- Verify model input shape (224x224x3)
- Check logcat for errors

### Slow Performance
- Enable NNAPI acceleration
- Try GPU delegate
- Profile with Android Profiler

### NNAPI Not Working
- Requires Android 8.1+ (API 27+)
- Check device support
- Fallback to CPU if unavailable

## Requirements

- Android 8.0+ (API 26+)
- Android Studio 2022.1+
- TensorFlow Lite 2.13+
- NNAPI (optional, Android 8.1+)

## Dependencies

```gradle
implementation 'org.tensorflow:tensorflow-lite:2.13.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
```

## Alternative: ONNX Runtime Mobile

For ONNX models, use ONNX Runtime Mobile:

```gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.15.1'
```

See `ONNXInference.java` for implementation.

## License

Medical Use Only - Research Purposes
