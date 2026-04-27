/**
 * Cache Service
 * 
 * Intelligent caching for models and results
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import RNFS from 'react-native-fs';

export interface CacheEntry {
  key: string;
  data: any;
  timestamp: number;
  size: number;
  ttl: number; // Time to live in ms
}

export interface CacheStats {
  totalEntries: number;
  totalSizeMB: number;
  hitRate: number;
  missRate: number;
}

const CACHE_DIR = `${RNFS.DocumentDirectoryPath}/cache`;
const CACHE_INDEX_KEY = '@medical_ai_cache_index';
const CACHE_STATS_KEY = '@medical_ai_cache_stats';
const MAX_CACHE_SIZE_MB = 500; // 500MB max cache

/**
 * Cache service for offline-first architecture
 * 
 * Features:
 * - LRU eviction policy
 * - TTL support
 * - Size limits
 * - Hit/miss tracking
 */
class CacheService {
  private cacheIndex: Map<string, CacheEntry> = new Map();
  private hits = 0;
  private misses = 0;

  /**
   * Initialize cache
   */
  async initialize(): Promise<void> {
    // Create cache directory
    const dirExists = await RNFS.exists(CACHE_DIR);
    if (!dirExists) {
      await RNFS.mkdir(CACHE_DIR);
    }

    // Load cache index
    await this.loadCacheIndex();

    // Clean expired entries
    await this.cleanExpiredEntries();
  }

  /**
   * Get from cache
   */
  async get<T>(key: string): Promise<T | null> {
    const entry = this.cacheIndex.get(key);

    if (!entry) {
      this.misses++;
      return null;
    }

    // Check TTL
    if (Date.now() - entry.timestamp > entry.ttl) {
      await this.delete(key);
      this.misses++;
      return null;
    }

    // Read from disk
    try {
      const filePath = this.getFilePath(key);
      const data = await RNFS.readFile(filePath, 'utf8');
      this.hits++;
      return JSON.parse(data) as T;
    } catch (error) {
      console.error(`Failed to read cache entry ${key}:`, error);
      this.misses++;
      return null;
    }
  }

  /**
   * Set cache entry
   */
  async set(key: string, data: any, ttl: number = 24 * 60 * 60 * 1000): Promise<void> {
    const serialized = JSON.stringify(data);
    const size = new Blob([serialized]).size;

    // Check cache size limit
    await this.ensureCacheSpace(size);

    // Write to disk
    const filePath = this.getFilePath(key);
    await RNFS.writeFile(filePath, serialized, 'utf8');

    // Update index
    const entry: CacheEntry = {
      key,
      data: null, // Don't store data in index
      timestamp: Date.now(),
      size,
      ttl,
    };

    this.cacheIndex.set(key, entry);
    await this.saveCacheIndex();
  }

  /**
   * Delete cache entry
   */
  async delete(key: string): Promise<void> {
    const filePath = this.getFilePath(key);

    try {
      await RNFS.unlink(filePath);
    } catch (error) {
      // File might not exist
    }

    this.cacheIndex.delete(key);
    await this.saveCacheIndex();
  }

  /**
   * Clear all cache
   */
  async clear(): Promise<void> {
    // Delete all files
    const files = await RNFS.readDir(CACHE_DIR);
    for (const file of files) {
      await RNFS.unlink(file.path);
    }

    // Clear index
    this.cacheIndex.clear();
    await this.saveCacheIndex();
  }

  /**
   * Get cache stats
   */
  async getStats(): Promise<CacheStats> {
    const totalEntries = this.cacheIndex.size;
    const totalSize = Array.from(this.cacheIndex.values()).reduce(
      (sum, entry) => sum + entry.size,
      0
    );
    const totalSizeMB = totalSize / (1024 * 1024);

    const totalRequests = this.hits + this.misses;
    const hitRate = totalRequests > 0 ? this.hits / totalRequests : 0;
    const missRate = totalRequests > 0 ? this.misses / totalRequests : 0;

    return {
      totalEntries,
      totalSizeMB,
      hitRate,
      missRate,
    };
  }

  /**
   * Ensure cache has space
   */
  private async ensureCacheSpace(requiredSize: number): Promise<void> {
    const stats = await this.getStats();
    const requiredSizeMB = requiredSize / (1024 * 1024);

    if (stats.totalSizeMB + requiredSizeMB <= MAX_CACHE_SIZE_MB) {
      return;
    }

    // Evict LRU entries
    const entries = Array.from(this.cacheIndex.values()).sort(
      (a, b) => a.timestamp - b.timestamp
    );

    let freedSpace = 0;
    for (const entry of entries) {
      await this.delete(entry.key);
      freedSpace += entry.size;

      if (freedSpace >= requiredSize) {
        break;
      }
    }
  }

  /**
   * Clean expired entries
   */
  private async cleanExpiredEntries(): Promise<void> {
    const now = Date.now();
    const expiredKeys: string[] = [];

    for (const [key, entry] of this.cacheIndex.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        expiredKeys.push(key);
      }
    }

    for (const key of expiredKeys) {
      await this.delete(key);
    }
  }

  /**
   * Load cache index
   */
  private async loadCacheIndex(): Promise<void> {
    try {
      const data = await AsyncStorage.getItem(CACHE_INDEX_KEY);
      if (data) {
        const entries = JSON.parse(data) as CacheEntry[];
        this.cacheIndex = new Map(entries.map(e => [e.key, e]));
      }
    } catch (error) {
      console.error('Failed to load cache index:', error);
    }
  }

  /**
   * Save cache index
   */
  private async saveCacheIndex(): Promise<void> {
    try {
      const entries = Array.from(this.cacheIndex.values());
      await AsyncStorage.setItem(CACHE_INDEX_KEY, JSON.stringify(entries));
    } catch (error) {
      console.error('Failed to save cache index:', error);
    }
  }

  /**
   * Get file path for key
   */
  private getFilePath(key: string): string {
    const hash = this.hashKey(key);
    return `${CACHE_DIR}/${hash}.json`;
  }

  /**
   * Hash key for filename
   */
  private hashKey(key: string): string {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      const char = key.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
  }
}

// Singleton instance
const cacheService = new CacheService();

export default cacheService;

// Export convenience functions
export const initializeCache = () => cacheService.initialize();
export const getFromCache = <T>(key: string) => cacheService.get<T>(key);
export const setInCache = (key: string, data: any, ttl?: number) =>
  cacheService.set(key, data, ttl);
export const deleteFromCache = (key: string) => cacheService.delete(key);
export const clearCache = () => cacheService.clear();
export const getCacheStats = () => cacheService.getStats();
