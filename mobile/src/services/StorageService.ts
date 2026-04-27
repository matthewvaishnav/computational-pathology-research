/**
 * Storage Service
 * 
 * Local data persistence using AsyncStorage and SQLite
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

export interface HistoryItem {
  id: string;
  imageUri: string;
  predictions: any[];
  timestamp: number;
}

export interface Settings {
  offlineMode: boolean;
  saveHistory: boolean;
  highQualityInference: boolean;
  autoSync: boolean;
}

const HISTORY_KEY = '@medical_ai_history';
const SETTINGS_KEY = '@medical_ai_settings';

/**
 * Save inference result to history
 */
export async function saveToHistory(
  imageUri: string,
  predictions: any[]
): Promise<void> {
  try {
    const history = await getHistory();
    
    const item: HistoryItem = {
      id: Date.now().toString(),
      imageUri,
      predictions,
      timestamp: Date.now(),
    };
    
    history.unshift(item);
    
    // Keep only last 100 items
    const trimmed = history.slice(0, 100);
    
    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
  } catch (error) {
    console.error('Failed to save history:', error);
  }
}

/**
 * Get inference history
 */
export async function getHistory(): Promise<HistoryItem[]> {
  try {
    const data = await AsyncStorage.getItem(HISTORY_KEY);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error('Failed to load history:', error);
    return [];
  }
}

/**
 * Clear all history
 */
export async function clearHistory(): Promise<void> {
  try {
    await AsyncStorage.removeItem(HISTORY_KEY);
  } catch (error) {
    console.error('Failed to clear history:', error);
  }
}

/**
 * Get settings
 */
export async function getSettings(): Promise<Settings> {
  try {
    const data = await AsyncStorage.getItem(SETTINGS_KEY);
    return data ? JSON.parse(data) : {
      offlineMode: true,
      saveHistory: true,
      highQualityInference: false,
      autoSync: false,
    };
  } catch (error) {
    console.error('Failed to load settings:', error);
    return {
      offlineMode: true,
      saveHistory: true,
      highQualityInference: false,
      autoSync: false,
    };
  }
}

/**
 * Save settings
 */
export async function saveSettings(settings: Settings): Promise<void> {
  try {
    await AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (error) {
    console.error('Failed to save settings:', error);
  }
}

/**
 * Clear all data
 */
export async function clearAllData(): Promise<void> {
  try {
    await AsyncStorage.multiRemove([HISTORY_KEY, SETTINGS_KEY]);
  } catch (error) {
    console.error('Failed to clear data:', error);
  }
}

/**
 * Get storage usage
 */
export async function getStorageUsage(): Promise<{
  history_items: number;
  total_size_mb: number;
}> {
  try {
    const history = await getHistory();
    const historyData = JSON.stringify(history);
    const sizeBytes = new Blob([historyData]).size;
    const sizeMB = sizeBytes / (1024 * 1024);
    
    return {
      history_items: history.length,
      total_size_mb: sizeMB,
    };
  } catch (error) {
    console.error('Failed to get storage usage:', error);
    return {
      history_items: 0,
      total_size_mb: 0,
    };
  }
}
