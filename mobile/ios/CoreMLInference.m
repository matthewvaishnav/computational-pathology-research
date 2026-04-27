//
//  CoreMLInference.m
//  MedicalAI
//
//  React Native bridge for CoreML inference
//

#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(CoreMLInference, NSObject)

RCT_EXTERN_METHOD(loadModel:(NSString *)modelName
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(predict:(NSString *)imageUri
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(getModelInfo:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
