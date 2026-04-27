package com.medicalai;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableNativeArray;
import com.facebook.react.bridge.WritableNativeMap;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * TensorFlow Lite inference native module for React Native
 */
public class TFLiteInference extends ReactContextBaseJavaModule {

    private static final String MODULE_NAME = "TFLiteInference";
    private static final int INPUT_SIZE = 224;
    private static final int NUM_CLASSES = 3;
    private static final String[] CLASS_LABELS = {"Benign", "Malignant", "Uncertain"};

    private Interpreter interpreter;
    private boolean isModelLoaded = false;
    private final ReactApplicationContext reactContext;

    public TFLiteInference(ReactApplicationContext reactContext) {
        super(reactContext);
        this.reactContext = reactContext;
    }

    @Override
    public String getName() {
        return MODULE_NAME;
    }

    /**
     * Load TFLite model from assets
     */
    @ReactMethod
    public void loadModel(String modelName, Promise promise) {
        try {
            // Load model from assets
            ByteBuffer modelBuffer = FileUtil.loadMappedFile(reactContext, modelName + ".tflite");

            // Create interpreter
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            options.setUseNNAPI(true); // Use Android Neural Networks API

            interpreter = new Interpreter(modelBuffer, options);
            isModelLoaded = true;

            WritableMap result = new WritableNativeMap();
            result.putBoolean("success", true);
            result.putString("modelName", modelName);
            result.putString("backend", "TFLite");

            promise.resolve(result);

        } catch (IOException e) {
            promise.reject("MODEL_LOAD_ERROR", "Failed to load model: " + e.getMessage(), e);
        }
    }

    /**
     * Run inference on image
     */
    @ReactMethod
    public void predict(String imageUri, Promise promise) {
        if (!isModelLoaded || interpreter == null) {
            promise.reject("MODEL_NOT_LOADED", "Model not loaded. Call loadModel first.");
            return;
        }

        try {
            long startTime = System.currentTimeMillis();

            // Load image
            Bitmap bitmap = loadImage(imageUri);
            if (bitmap == null) {
                promise.reject("IMAGE_LOAD_ERROR", "Failed to load image from URI");
                return;
            }

            // Preprocess
            long preprocessStart = System.currentTimeMillis();
            ByteBuffer inputBuffer = preprocessImage(bitmap);
            long preprocessTime = System.currentTimeMillis() - preprocessStart;

            // Run inference
            long inferenceStart = System.currentTimeMillis();
            float[][] output = new float[1][NUM_CLASSES];
            interpreter.run(inputBuffer, output);
            long inferenceTime = System.currentTimeMillis() - inferenceStart;

            // Postprocess
            long postprocessStart = System.currentTimeMillis();
            List<Prediction> predictions = postprocessPredictions(output[0]);
            long postprocessTime = System.currentTimeMillis() - postprocessStart;

            long totalTime = System.currentTimeMillis() - startTime;

            // Build result
            WritableMap result = new WritableNativeMap();
            result.putArray("predictions", predictionsToWritableArray(predictions));
            result.putDouble("inference_time_ms", inferenceTime);
            result.putDouble("preprocessing_time_ms", preprocessTime);
            result.putDouble("postprocessing_time_ms", postprocessTime);
            result.putDouble("total_time_ms", totalTime);

            promise.resolve(result);

        } catch (Exception e) {
            promise.reject("INFERENCE_ERROR", "Inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Get model info
     */
    @ReactMethod
    public void getModelInfo(Promise promise) {
        WritableMap info = new WritableNativeMap();
        info.putString("name", "Medical AI Foundation");
        info.putString("version", "1.0.0");
        info.putString("backend", "TFLite");
        info.putBoolean("isLoaded", isModelLoaded);

        promise.resolve(info);
    }

    // MARK: - Image Processing

    private Bitmap loadImage(String uri) {
        try {
            // Handle file:// URIs
            if (uri.startsWith("file://")) {
                String path = uri.replace("file://", "");
                return BitmapFactory.decodeFile(path);
            }

            // Handle content:// URIs
            if (uri.startsWith("content://")) {
                return BitmapFactory.decodeStream(
                    reactContext.getContentResolver().openInputStream(Uri.parse(uri))
                );
            }

            return null;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        // Resize to 224x224
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);

        // Convert to ByteBuffer
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        // Normalize to [0, 1]
        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;

            buffer.putFloat(r);
            buffer.putFloat(g);
            buffer.putFloat(b);
        }

        return buffer;
    }

    private List<Prediction> postprocessPredictions(float[] output) {
        List<Prediction> predictions = new ArrayList<>();

        for (int i = 0; i < output.length && i < CLASS_LABELS.length; i++) {
            predictions.add(new Prediction(CLASS_LABELS[i], output[i]));
        }

        // Sort by confidence
        Collections.sort(predictions, new Comparator<Prediction>() {
            @Override
            public int compare(Prediction p1, Prediction p2) {
                return Float.compare(p2.confidence, p1.confidence);
            }
        });

        // Take top 3
        return predictions.subList(0, Math.min(3, predictions.size()));
    }

    private WritableArray predictionsToWritableArray(List<Prediction> predictions) {
        WritableArray array = new WritableNativeArray();

        for (Prediction pred : predictions) {
            WritableMap map = new WritableNativeMap();
            map.putString("label", pred.label);
            map.putDouble("confidence", pred.confidence);
            array.pushMap(map);
        }

        return array;
    }

    // MARK: - Helper Classes

    private static class Prediction {
        String label;
        float confidence;

        Prediction(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }
}
