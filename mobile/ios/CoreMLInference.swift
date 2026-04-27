//
//  CoreMLInference.swift
//  MedicalAI
//
//  CoreML inference native module for React Native
//

import Foundation
import CoreML
import Vision
import UIKit

@objc(CoreMLInference)
class CoreMLInference: NSObject {
  
  private var model: VNCoreMLModel?
  private var isModelLoaded = false
  
  // MARK: - Model Loading
  
  @objc
  func loadModel(_ modelName: String,
                 resolver resolve: @escaping RCTPromiseResolveBlock,
                 rejecter reject: @escaping RCTPromiseRejectBlock) {
    
    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
      guard let self = self else { return }
      
      do {
        // Load CoreML model from bundle
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
          reject("MODEL_NOT_FOUND", "Model file not found: \(modelName)", nil)
          return
        }
        
        let mlModel = try MLModel(contentsOf: modelURL)
        self.model = try VNCoreMLModel(for: mlModel)
        self.isModelLoaded = true
        
        DispatchQueue.main.async {
          resolve([
            "success": true,
            "modelName": modelName,
            "backend": "CoreML"
          ])
        }
        
      } catch {
        DispatchQueue.main.async {
          reject("MODEL_LOAD_ERROR", "Failed to load model: \(error.localizedDescription)", error)
        }
      }
    }
  }
  
  // MARK: - Inference
  
  @objc
  func predict(_ imageUri: String,
               resolver resolve: @escaping RCTPromiseResolveBlock,
               rejecter reject: @escaping RCTPromiseRejectBlock) {
    
    guard isModelLoaded, let model = self.model else {
      reject("MODEL_NOT_LOADED", "Model not loaded. Call loadModel first.", nil)
      return
    }
    
    DispatchQueue.global(qos: .userInitiated).async { [weak self] in
      guard let self = self else { return }
      
      do {
        // Load image
        guard let image = self.loadImage(from: imageUri) else {
          reject("IMAGE_LOAD_ERROR", "Failed to load image from URI", nil)
          return
        }
        
        // Preprocess
        let startTime = Date()
        guard let preprocessedImage = self.preprocessImage(image) else {
          reject("PREPROCESS_ERROR", "Failed to preprocess image", nil)
          return
        }
        let preprocessTime = Date().timeIntervalSince(startTime) * 1000
        
        // Run inference
        let inferenceStart = Date()
        let predictions = try self.runInference(on: preprocessedImage, using: model)
        let inferenceTime = Date().timeIntervalSince(inferenceStart) * 1000
        
        // Postprocess
        let postprocessStart = Date()
        let results = self.postprocessPredictions(predictions)
        let postprocessTime = Date().timeIntervalSince(postprocessStart) * 1000
        
        let totalTime = Date().timeIntervalSince(startTime) * 1000
        
        DispatchQueue.main.async {
          resolve([
            "predictions": results,
            "inference_time_ms": inferenceTime,
            "preprocessing_time_ms": preprocessTime,
            "postprocessing_time_ms": postprocessTime,
            "total_time_ms": totalTime
          ])
        }
        
      } catch {
        DispatchQueue.main.async {
          reject("INFERENCE_ERROR", "Inference failed: \(error.localizedDescription)", error)
        }
      }
    }
  }
  
  // MARK: - Image Processing
  
  private func loadImage(from uri: String) -> UIImage? {
    // Handle file:// URIs
    if uri.hasPrefix("file://") {
      let path = uri.replacingOccurrences(of: "file://", with: "")
      return UIImage(contentsOfFile: path)
    }
    
    // Handle ph:// (Photos framework) URIs
    if uri.hasPrefix("ph://") {
      // TODO: Load from Photos framework
      return nil
    }
    
    return nil
  }
  
  private func preprocessImage(_ image: UIImage) -> CVPixelBuffer? {
    // Resize to 224x224 (model input size)
    let targetSize = CGSize(width: 224, height: 224)
    
    guard let resizedImage = image.resize(to: targetSize) else {
      return nil
    }
    
    // Convert to CVPixelBuffer
    return resizedImage.pixelBuffer()
  }
  
  private func runInference(on pixelBuffer: CVPixelBuffer,
                           using model: VNCoreMLModel) throws -> [VNClassificationObservation] {
    
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .centerCrop
    
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
    try handler.perform([request])
    
    guard let results = request.results as? [VNClassificationObservation] else {
      throw NSError(domain: "CoreMLInference", code: -1, userInfo: [
        NSLocalizedDescriptionKey: "Failed to get classification results"
      ])
    }
    
    return results
  }
  
  private func postprocessPredictions(_ observations: [VNClassificationObservation]) -> [[String: Any]] {
    // Sort by confidence
    let sorted = observations.sorted { $0.confidence > $1.confidence }
    
    // Take top 3
    let top3 = Array(sorted.prefix(3))
    
    // Convert to dictionary
    return top3.map { observation in
      return [
        "label": observation.identifier,
        "confidence": Double(observation.confidence)
      ]
    }
  }
  
  // MARK: - Model Info
  
  @objc
  func getModelInfo(_ resolver resolve: @escaping RCTPromiseResolveBlock,
                    rejecter reject: @escaping RCTPromiseRejectBlock) {
    
    resolve([
      "name": "Medical AI Foundation",
      "version": "1.0.0",
      "backend": "CoreML",
      "isLoaded": isModelLoaded
    ])
  }
  
  // MARK: - React Native Bridge
  
  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}

// MARK: - UIImage Extensions

extension UIImage {
  
  func resize(to size: CGSize) -> UIImage? {
    UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
    defer { UIGraphicsEndImageContext() }
    
    draw(in: CGRect(origin: .zero, size: size))
    return UIGraphicsGetImageFromCurrentImageContext()
  }
  
  func pixelBuffer() -> CVPixelBuffer? {
    let width = Int(size.width)
    let height = Int(size.height)
    
    let attrs = [
      kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
      kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
    ] as CFDictionary
    
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      width,
      height,
      kCVPixelFormatType_32ARGB,
      attrs,
      &pixelBuffer
    )
    
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      return nil
    }
    
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    
    let pixelData = CVPixelBufferGetBaseAddress(buffer)
    
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
      data: pixelData,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
      space: rgbColorSpace,
      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
    ) else {
      return nil
    }
    
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)
    
    UIGraphicsPushContext(context)
    draw(in: CGRect(x: 0, y: 0, width: width, height: height))
    UIGraphicsPopContext()
    
    return buffer
  }
}
