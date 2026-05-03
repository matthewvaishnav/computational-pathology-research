// Checkpointing utilities for Opus Delegation System

import * as fs from 'fs';
import * as path from 'path';
import type { DelegationSession } from '../types/core.js';

export interface Checkpoint {
  sessionId: string;
  timestamp: Date;
  roundNumber: number;
  state: Partial<DelegationSession>;
  checksum: string;
}

export class CheckpointManager {
  private checkpointDir: string;

  constructor(baseDir: string = '.opus-delegation') {
    this.checkpointDir = path.join(baseDir, 'checkpoints');
    this.ensureCheckpointDir();
  }

  private ensureCheckpointDir(): void {
    if (!fs.existsSync(this.checkpointDir)) {
      fs.mkdirSync(this.checkpointDir, { recursive: true });
    }
  }

  private calculateChecksum(data: string): string {
    // Simple checksum using hash
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(16);
  }

  createCheckpoint(
    sessionId: string,
    roundNumber: number,
    state: Partial<DelegationSession>
  ): Checkpoint {
    const checkpoint: Checkpoint = {
      sessionId,
      timestamp: new Date(),
      roundNumber,
      state,
      checksum: '',
    };

    const data = JSON.stringify(checkpoint.state);
    checkpoint.checksum = this.calculateChecksum(data);

    const checkpointPath = this.getCheckpointPath(sessionId, roundNumber);
    fs.writeFileSync(checkpointPath, JSON.stringify(checkpoint, null, 2));

    return checkpoint;
  }

  loadCheckpoint(sessionId: string, roundNumber?: number): Checkpoint | null {
    try {
      let checkpointPath: string;

      if (roundNumber !== undefined) {
        checkpointPath = this.getCheckpointPath(sessionId, roundNumber);
      } else {
        // Load latest checkpoint
        const checkpoints = this.listCheckpoints(sessionId);
        if (checkpoints.length === 0) {
          return null;
        }
        checkpointPath = checkpoints[checkpoints.length - 1];
      }

      if (!fs.existsSync(checkpointPath)) {
        return null;
      }

      const data = fs.readFileSync(checkpointPath, 'utf-8');
      const checkpoint: Checkpoint = JSON.parse(data);

      // Verify checksum
      const stateData = JSON.stringify(checkpoint.state);
      const calculatedChecksum = this.calculateChecksum(stateData);

      if (calculatedChecksum !== checkpoint.checksum) {
        throw new Error('Checkpoint checksum mismatch - data may be corrupted');
      }

      return checkpoint;
    } catch (error) {
      console.error(`Failed to load checkpoint: ${error}`);
      return null;
    }
  }

  listCheckpoints(sessionId: string): string[] {
    const sessionDir = path.join(this.checkpointDir, sessionId);

    if (!fs.existsSync(sessionDir)) {
      return [];
    }

    return fs
      .readdirSync(sessionDir)
      .filter((file) => file.endsWith('.json'))
      .map((file) => path.join(sessionDir, file))
      .sort();
  }

  deleteCheckpoint(sessionId: string, roundNumber: number): void {
    const checkpointPath = this.getCheckpointPath(sessionId, roundNumber);

    if (fs.existsSync(checkpointPath)) {
      fs.unlinkSync(checkpointPath);
    }
  }

  deleteAllCheckpoints(sessionId: string): void {
    const sessionDir = path.join(this.checkpointDir, sessionId);

    if (fs.existsSync(sessionDir)) {
      fs.rmSync(sessionDir, { recursive: true, force: true });
    }
  }

  private getCheckpointPath(sessionId: string, roundNumber: number): string {
    const sessionDir = path.join(this.checkpointDir, sessionId);

    if (!fs.existsSync(sessionDir)) {
      fs.mkdirSync(sessionDir, { recursive: true });
    }

    return path.join(sessionDir, `checkpoint_round_${roundNumber}.json`);
  }

  resumeFromCheckpoint(sessionId: string): Partial<DelegationSession> | null {
    const checkpoint = this.loadCheckpoint(sessionId);

    if (!checkpoint) {
      return null;
    }

    return checkpoint.state;
  }
}
