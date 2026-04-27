# Medical AI Mobile Application

Cross-platform React Native mobile app for medical pathology AI inference.

## Features

- **Offline-First**: All inference runs on-device
- **Privacy-Focused**: No data leaves the device
- **Medical-Grade**: Optimized for medical imaging
- **Cross-Platform**: iOS and Android support

## Architecture

### Screens
- **HomeScreen**: Main landing page
- **CameraScreen**: Image capture/selection
- **InferenceScreen**: On-device AI processing
- **ResultsScreen**: Display predictions
- **HistoryScreen**: View past results
- **SettingsScreen**: App configuration

### Services
- **InferenceService**: On-device AI inference
  - iOS: CoreML backend
  - Android: TFLite/ONNX Runtime Mobile
- **StorageService**: Local data persistence
  - AsyncStorage for settings
  - SQLite for history (optional)

### Native Modules (TODO)
- **CoreMLInference** (iOS): CoreML model inference
- **TFLiteInference** (Android): TensorFlow Lite inference
- **ONNXInference** (Android): ONNX Runtime Mobile

## Setup

### Prerequisites
- Node.js >= 16
- React Native CLI
- Xcode (iOS)
- Android Studio (Android)

### Installation

```bash
cd mobile
npm install

# iOS
cd ios && pod install && cd ..

# Android
# No additional steps
```

### Run

```bash
# iOS
npm run ios

# Android
npm run android
```

## Model Integration

### iOS (CoreML)

1. Convert PyTorch → CoreML:
```python
from src.mobile_edge.optimization.coreml_converter import convert_medical_model_coreml

convert_medical_model_coreml(
    model=pytorch_model,
    input_shape=(1, 3, 224, 224),
    save_path='medical_model.mlmodel'
)
```

2. Add `.mlmodel` to Xcode project
3. Implement native module in `ios/CoreMLInference.swift`

### Android (TFLite)

1. Convert PyTorch → TFLite:
```python
from src.mobile_edge.optimization.mobile_inference import convert_pytorch_to_tflite

convert_pytorch_to_tflite(
    model=pytorch_model,
    input_shape=(1, 3, 224, 224),
    save_path='medical_model.tflite'
)
```

2. Add `.tflite` to `android/app/src/main/assets/`
3. Implement native module in `android/app/src/main/java/TFLiteInference.java`

### Android (ONNX Runtime Mobile)

1. Export PyTorch → ONNX:
```python
from src.mobile_edge.optimization.onnx_exporter import export_medical_model_onnx

export_medical_model_onnx(
    model=pytorch_model,
    input_shape=(1, 3, 224, 224),
    save_path='medical_model.onnx'
)
```

2. Add `.onnx` to `android/app/src/main/assets/`
3. Add ONNX Runtime Mobile dependency
4. Implement inference in Java/Kotlin

## Performance

### Target Metrics
- Inference time: <500ms on-device
- Model size: <50MB
- Memory usage: <200MB
- Battery impact: Minimal

### Optimization Techniques
- Model compression (pruning + quantization)
- Knowledge distillation
- Platform-specific optimization (CoreML, TFLite)
- Efficient preprocessing

## Privacy & Security

- **On-Device Processing**: All inference runs locally
- **No Network Calls**: No data transmission
- **Local Storage**: Encrypted storage for history
- **HIPAA Considerations**: Designed for medical data

## Testing

```bash
npm test
```

## Build

### iOS

```bash
npm run build:ios
```

### Android

```bash
npm run build:android
```

## Deployment

### iOS App Store
1. Configure signing in Xcode
2. Archive and upload to App Store Connect
3. Submit for review

### Google Play Store
1. Generate signed APK/AAB
2. Upload to Google Play Console
3. Submit for review

## License

Medical Use Only - Research Purposes

## Disclaimer

⚠️ **For research purposes only. Not for clinical diagnosis.**

This application is intended for research and educational purposes. It should not be used for clinical diagnosis or treatment decisions without proper validation and regulatory approval.
