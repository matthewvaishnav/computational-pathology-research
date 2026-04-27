# Offline-First Architecture

Comprehensive offline-first architecture for Medical AI mobile app.

## Overview

The app is designed to work completely offline with optional cloud sync when available.

## Core Principles

1. **Offline by Default**: All features work without network
2. **Local-First**: Data stored locally first, synced later
3. **Eventual Consistency**: Sync when network available
4. **Graceful Degradation**: Fallback to cached/bundled resources

## Architecture Components

### 1. Offline Service (`OfflineService.ts`)

Manages offline operations and sync queue.

**Features**:
- Sync queue with retry logic
- Network state monitoring
- Automatic sync when online
- Exponential backoff for retries

**Usage**:
```typescript
import offlineService from './services/OfflineService';

// Initialize
await offlineService.initialize();

// Save result (offline-first)
await offlineService.saveInferenceResult(imageUri, predictions);

// Get sync status
const status = await offlineService.getSyncStatus();
console.log(`Pending items: ${status.pendingItems}`);

// Force sync
await offlineService.forceSyncNow();
```

### 2. Cache Service (`CacheService.ts`)

Intelligent caching with LRU eviction.

**Features**:
- LRU eviction policy
- TTL support
- Size limits (500MB max)
- Hit/miss tracking

**Usage**:
```typescript
import cacheService from './services/CacheService';

// Initialize
await cacheService.initialize();

// Set cache
await cacheService.set('model_predictions', predictions, 24 * 60 * 60 * 1000);

// Get from cache
const cached = await cacheService.get('model_predictions');

// Get stats
const stats = await cacheService.getStats();
console.log(`Hit rate: ${(stats.hitRate * 100).toFixed(1)}%`);
```

### 3. Model Manager (`ModelManager.ts`)

Manages model lifecycle and updates.

**Features**:
- Model versioning
- Automatic updates
- Fallback to bundled model
- Delta updates (future)

**Usage**:
```typescript
import modelManager from './services/ModelManager';

// Initialize
await modelManager.initialize();

// Get current model
const model = await modelManager.getCurrentModel();
console.log(`Model: ${model.name} v${model.version}`);

// Check for updates
const hasUpdate = await modelManager.checkForUpdates();

// Download update
if (hasUpdate) {
  await modelManager.downloadUpdate(newModelInfo);
}

// Cleanup old models
await modelManager.cleanupOldModels();
```

### 4. Storage Service (`StorageService.ts`)

Local data persistence.

**Features**:
- AsyncStorage for settings
- SQLite for history (optional)
- Encrypted storage (future)

## Data Flow

### Inference Flow (Offline)

```
User captures image
  ↓
Image saved locally
  ↓
Preprocessing (on-device)
  ↓
Inference (on-device)
  ↓
Results saved locally
  ↓
Added to sync queue
  ↓
[When online] Sync to server
```

### Sync Flow

```
Network becomes available
  ↓
OfflineService detects
  ↓
Process sync queue
  ↓
For each item:
  - Try to sync
  - If success: remove from queue
  - If fail: increment retry count
  - If max retries: drop item
  ↓
Update last sync time
```

## Storage Structure

### AsyncStorage Keys
- `@medical_ai_history`: Inference history
- `@medical_ai_settings`: App settings
- `@medical_ai_sync_queue`: Sync queue
- `@medical_ai_last_sync`: Last sync timestamp
- `@medical_ai_cache_index`: Cache index
- `@medical_ai_model_info`: Model metadata

### File System
```
DocumentDirectory/
├── cache/
│   ├── <hash1>.json
│   ├── <hash2>.json
│   └── ...
├── models/
│   ├── medical_model_1.0.0.mlmodel (iOS)
│   ├── medical_model_1.0.0.tflite (Android)
│   └── ...
└── images/
    ├── captured/
    └── processed/
```

## Network Handling

### Network State Detection
```typescript
import NetInfo from '@react-native-community/netinfo';

// Listen for changes
NetInfo.addEventListener(state => {
  if (state.isConnected) {
    // Trigger sync
  }
});

// Check current state
const netInfo = await NetInfo.fetch();
console.log(`Online: ${netInfo.isConnected}`);
```

### Retry Logic

Exponential backoff with max retries:
- Retry 1: Immediate
- Retry 2: 2s delay
- Retry 3: 4s delay
- Max retries: 3
- After max: Drop item

## Conflict Resolution

### Strategy: Last Write Wins (LWW)

When syncing:
1. Send local timestamp
2. Server compares with server timestamp
3. If local > server: Accept update
4. If local < server: Reject (conflict)
5. If conflict: Notify user

### Future: Operational Transform (OT)

For collaborative features:
- Track operations, not states
- Transform operations for concurrent edits
- Merge without conflicts

## Performance Optimization

### 1. Lazy Loading
- Load models on-demand
- Unload when not in use
- Preload on app startup (optional)

### 2. Batch Operations
- Batch sync requests
- Batch database writes
- Reduce I/O overhead

### 3. Background Sync
- Use background tasks (iOS/Android)
- Sync during idle time
- Respect battery/data limits

## Security Considerations

### 1. Local Data Encryption
```typescript
// Future: Encrypt sensitive data
import { encrypt, decrypt } from './crypto';

const encrypted = await encrypt(data, key);
await AsyncStorage.setItem(key, encrypted);
```

### 2. Secure Sync
- HTTPS only
- Certificate pinning
- Token-based auth

### 3. Data Retention
- Auto-delete old data
- User-controlled retention
- HIPAA compliance

## Testing

### Unit Tests
```bash
npm test -- OfflineService.test.ts
npm test -- CacheService.test.ts
npm test -- ModelManager.test.ts
```

### Integration Tests
```bash
npm test -- offline-integration.test.ts
```

### Manual Testing
1. Enable airplane mode
2. Capture image
3. Run inference
4. Verify results saved
5. Disable airplane mode
6. Verify sync occurs

## Monitoring

### Metrics to Track
- Sync queue size
- Sync success rate
- Cache hit rate
- Model load time
- Inference time
- Storage usage

### Logging
```typescript
console.log('[OfflineService] Syncing 5 items...');
console.log('[CacheService] Hit rate: 85.2%');
console.log('[ModelManager] Model loaded: 1.2s');
```

## Future Enhancements

1. **Delta Sync**: Only sync changes, not full data
2. **Compression**: Compress data before sync
3. **P2P Sync**: Sync between devices directly
4. **Conflict UI**: Show conflicts to user
5. **Selective Sync**: User chooses what to sync
6. **Background Fetch**: iOS/Android background tasks

## Best Practices

1. **Always save locally first**
2. **Queue for sync, don't block**
3. **Handle network errors gracefully**
4. **Provide offline indicators**
5. **Test offline scenarios**
6. **Monitor sync queue size**
7. **Clean up old data**

## Troubleshooting

### Sync Not Working
- Check network state
- Verify sync queue not empty
- Check retry count
- Review error logs

### Cache Not Working
- Check cache size limits
- Verify TTL not expired
- Check file permissions
- Review cache stats

### Model Not Loading
- Verify model exists
- Check file path
- Review model info
- Check native module logs

## License

Medical Use Only - Research Purposes
