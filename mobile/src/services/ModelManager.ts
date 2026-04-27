/**
 * Model Manager
 * 
 * Manages model lifecycle and updates
 */

import { Platform } from 'react-native';
import RNFS from 'react-native-fs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { getFromCache, setInCache } from './CacheService';

export interface ModelInfo {
  name: string;
  version: string;
  size: number;
  backend: string;
  downloadUrl?: string;
  localPath?: string;
  isLoaded: boolean;
  lastUpdated: number;
}

const MODELS_DIR = `${RNFS.DocumentDirectoryPath}/models`;
const MODEL_INFO_KEY = '@medical_ai_model_info';

/**
 * Model manager for offline-first architecture
 * 
 * Features:
 * - Model versioning
 * - Automatic updates
 * - Fallback to bundled model
 * - Delta updates (future)
 */
class ModelManager {
  private currentModel: ModelInfo | null = null;

  /**
   * Initialize model manager
   */
  async initialize(): Promise<void> {
    // Create models directory
    const dirExists = await RNFS.exists(MODELS_DIR);
    if (!dirExists) {
      await RNFS.mkdir(MODELS_DIR);
    }

    // Load current model info
    await this.loadModelInfo();

    // Check for bundled model
    await this.checkBundledModel();
  }

  /**
   * Get current model
   */
  async getCurrentModel(): Promise<ModelInfo | null> {
    if (!this.currentModel) {
      await this.loadModelInfo();
    }
    return this.currentModel;
  }

  /**
   * Check for model updates
   */
  async checkForUpdates(): Promise<boolean> {
    // TODO: Implement server check
    // For now, return false (no updates)
    return false;
  }

  /**
   * Download model update
   */
  async downloadUpdate(modelInfo: ModelInfo): Promise<void> {
    if (!modelInfo.downloadUrl) {
      throw new Error('No download URL provided');
    }

    const localPath = `${MODELS_DIR}/${modelInfo.name}_${modelInfo.version}.${this.getModelExtension()}`;

    // Download model
    const download = RNFS.downloadFile({
      fromUrl: modelInfo.downloadUrl,
      toFile: localPath,
      progressDivider: 10,
      progress: (res) => {
        const progress = (res.bytesWritten / res.contentLength) * 100;
        console.log(`Download progress: ${progress.toFixed(1)}%`);
      },
    });

    await download.promise;

    // Update model info
    this.currentModel = {
      ...modelInfo,
      localPath,
      isLoaded: false,
      lastUpdated: Date.now(),
    };

    await this.saveModelInfo();
  }

  /**
   * Load model into memory
   */
  async loadModel(): Promise<void> {
    if (!this.currentModel) {
      throw new Error('No model available');
    }

    // Model loading is handled by native modules
    // This just updates the state

    this.currentModel.isLoaded = true;
    await this.saveModelInfo();
  }

  /**
   * Unload model from memory
   */
  async unloadModel(): Promise<void> {
    if (this.currentModel) {
      this.currentModel.isLoaded = false;
      await this.saveModelInfo();
    }
  }

  /**
   * Delete old models
   */
  async cleanupOldModels(): Promise<void> {
    const files = await RNFS.readDir(MODELS_DIR);

    for (const file of files) {
      // Keep current model
      if (this.currentModel && file.path === this.currentModel.localPath) {
        continue;
      }

      // Delete old models
      await RNFS.unlink(file.path);
    }
  }

  /**
   * Get model size
   */
  async getModelSize(): Promise<number> {
    if (!this.currentModel || !this.currentModel.localPath) {
      return 0;
    }

    try {
      const stat = await RNFS.stat(this.currentModel.localPath);
      return stat.size;
    } catch (error) {
      return 0;
    }
  }

  /**
   * Check for bundled model
   */
  private async checkBundledModel(): Promise<void> {
    // Check if bundled model exists
    const bundledPath = this.getBundledModelPath();

    if (Platform.OS === 'ios') {
      // iOS: Model is in app bundle, no need to copy
      if (!this.currentModel) {
        this.currentModel = {
          name: 'medical_model',
          version: '1.0.0',
          size: 45 * 1024 * 1024, // 45MB
          backend: 'CoreML',
          localPath: bundledPath,
          isLoaded: false,
          lastUpdated: Date.now(),
        };
        await this.saveModelInfo();
      }
    } else if (Platform.OS === 'android') {
      // Android: Model is in assets, no need to copy
      if (!this.currentModel) {
        this.currentModel = {
          name: 'medical_model',
          version: '1.0.0',
          size: 38 * 1024 * 1024, // 38MB
          backend: 'TFLite',
          localPath: bundledPath,
          isLoaded: false,
          lastUpdated: Date.now(),
        };
        await this.saveModelInfo();
      }
    }
  }

  /**
   * Get bundled model path
   */
  private getBundledModelPath(): string {
    if (Platform.OS === 'ios') {
      return 'medical_model'; // CoreML model name
    } else {
      return 'medical_model'; // TFLite model name (in assets)
    }
  }

  /**
   * Get model extension
   */
  private getModelExtension(): string {
    if (Platform.OS === 'ios') {
      return 'mlmodel';
    } else {
      return 'tflite';
    }
  }

  /**
   * Load model info
   */
  private async loadModelInfo(): Promise<void> {
    try {
      const data = await AsyncStorage.getItem(MODEL_INFO_KEY);
      if (data) {
        this.currentModel = JSON.parse(data);
      }
    } catch (error) {
      console.error('Failed to load model info:', error);
    }
  }

  /**
   * Save model info
   */
  private async saveModelInfo(): Promise<void> {
    try {
      if (this.currentModel) {
        await AsyncStorage.setItem(MODEL_INFO_KEY, JSON.stringify(this.currentModel));
      }
    } catch (error) {
      console.error('Failed to save model info:', error);
    }
  }
}

// Singleton instance
const modelManager = new ModelManager();

export default modelManager;

// Export convenience functions
export const initializeModelManager = () => modelManager.initialize();
export const getCurrentModel = () => modelManager.getCurrentModel();
export const checkForModelUpdates = () => modelManager.checkForUpdates();
export const downloadModelUpdate = (modelInfo: ModelInfo) =>
  modelManager.downloadUpdate(modelInfo);
export const loadModel = () => modelManager.loadModel();
export const unloadModel = () => modelManager.unloadModel();
export const cleanupOldModels = () => modelManager.cleanupOldModels();
export const getModelSize = () => modelManager.getModelSize();
