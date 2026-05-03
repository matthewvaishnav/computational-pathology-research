/**
 * Session History Manager Component
 * Implements Task 15 - Session History and Context Management
 * Requirements: 8.1-8.7
 */

import {
  DelegationSession,
  DelegationRound,
  ParsedArtifact,
  DelegationType,
  SessionComplexity,
} from '../types/core.js';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Session search criteria
 */
export interface SessionSearchCriteria {
  problemType?: DelegationType;
  keywords?: string[];
  dateFrom?: Date;
  dateTo?: Date;
  status?: 'active' | 'completed' | 'abandoned';
}

/**
 * Session summary report
 */
export interface SessionReport {
  sessionId: string;
  problemTitle: string;
  problemType: DelegationType;
  complexity: SessionComplexity;
  createdAt: Date;
  completedAt?: Date;
  roundCount: number;
  artifactCount: number;
  implementationStatus: 'not_started' | 'in_progress' | 'completed';
  completenessScore: number;
}

/**
 * Session History Manager Component
 * Manages delegation session storage, retrieval, and reuse
 */
export class SessionHistoryManager {
  private sessionsDir: string;
  private sessionIndex: Map<string, DelegationSession>;

  constructor(baseDir: string = '.opus-delegation') {
    this.sessionsDir = path.join(baseDir, 'sessions');
    this.sessionIndex = new Map();
    this.ensureDirectoryStructure();
    this.loadSessionIndex();
  }

  /**
   * Ensure directory structure exists
   * Requirement 8.1: Define session directory structure
   * Task 15.1: Create session storage
   */
  private ensureDirectoryStructure(): void {
    if (!fs.existsSync(this.sessionsDir)) {
      fs.mkdirSync(this.sessionsDir, { recursive: true });
    }
  }

  /**
   * Load session index from disk
   */
  private loadSessionIndex(): void {
    if (!fs.existsSync(this.sessionsDir)) {
      return;
    }

    const sessionDirs = fs.readdirSync(this.sessionsDir);

    for (const sessionId of sessionDirs) {
      const sessionPath = path.join(this.sessionsDir, sessionId, 'session.json');
      if (fs.existsSync(sessionPath)) {
        try {
          const sessionData = fs.readFileSync(sessionPath, 'utf-8');
          const session = JSON.parse(sessionData) as DelegationSession;
          
          // Convert date strings back to Date objects
          session.createdAt = new Date(session.createdAt);
          session.updatedAt = new Date(session.updatedAt);
          if (session.completedAt) {
            session.completedAt = new Date(session.completedAt);
          }
          
          session.rounds = session.rounds.map((round) => ({
            ...round,
            timestamp: new Date(round.timestamp),
          }));

          this.sessionIndex.set(sessionId, session);
        } catch (error) {
          console.error(`Failed to load session ${sessionId}:`, error);
        }
      }
    }
  }

  /**
   * Create new delegation session
   * Requirement 8.1: Maintain session history recording all delegation requests
   */
  public createSession(
    problemTitle: string,
    problemDescription: string,
    problemType: DelegationType,
    complexity: SessionComplexity
  ): DelegationSession {
    const sessionId = this.generateSessionId();
    const now = new Date();

    const session: DelegationSession = {
      id: sessionId,
      createdAt: now,
      updatedAt: now,
      problem: {
        title: problemTitle,
        description: problemDescription,
        type: problemType,
        complexity,
      },
      rounds: [],
      finalArtifacts: [],
      metrics: {
        totalTime: 0,
        contextSize: 0,
        roundCount: 0,
        finalCompleteness: 0,
      },
      status: 'active',
    };

    this.sessionIndex.set(sessionId, session);
    this.saveSession(session);

    return session;
  }

  /**
   * Add round to session
   * Requirement 8.2: Store timestamps, problem descriptions, context bundles, artifact links
   */
  public addRound(
    sessionId: string,
    request: string,
    response: string,
    artifacts: ParsedArtifact[],
    contextSize: number
  ): DelegationRound {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    const roundNumber = session.rounds.length + 1;
    const round: DelegationRound = {
      roundNumber,
      request,
      response,
      artifacts,
      timestamp: new Date(),
      contextSize,
    };

    session.rounds.push(round);
    session.updatedAt = new Date();
    session.metrics.roundCount = roundNumber;
    session.metrics.contextSize = Math.max(session.metrics.contextSize, contextSize);

    this.saveSession(session);
    this.saveRound(sessionId, round);

    return round;
  }

  /**
   * Update session with final artifacts
   */
  public updateFinalArtifacts(sessionId: string, artifacts: ParsedArtifact[]): void {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    session.finalArtifacts = artifacts;
    session.updatedAt = new Date();

    this.saveSession(session);
  }

  /**
   * Mark session as completed
   */
  public completeSession(sessionId: string, completenessScore: number): void {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    session.completedAt = new Date();
    session.status = 'completed';
    session.metrics.finalCompleteness = completenessScore;
    session.metrics.totalTime = session.completedAt.getTime() - session.createdAt.getTime();

    this.saveSession(session);
  }

  /**
   * Get session by ID
   * Requirement 8.7: Stored context and artifacts shall be retrievable
   */
  public getSession(sessionId: string): DelegationSession | undefined {
    return this.sessionIndex.get(sessionId);
  }

  /**
   * Search sessions by criteria
   * Requirement 8.3: Support searching session history by problem type, keywords, date range
   * Task 15.2: Implement session search and retrieval
   */
  public searchSessions(criteria: SessionSearchCriteria): DelegationSession[] {
    let results = Array.from(this.sessionIndex.values());

    // Filter by problem type
    if (criteria.problemType) {
      results = results.filter((s) => s.problem.type === criteria.problemType);
    }

    // Filter by keywords
    if (criteria.keywords && criteria.keywords.length > 0) {
      results = results.filter((s) => {
        const searchText = `${s.problem.title} ${s.problem.description}`.toLowerCase();
        return criteria.keywords!.some((keyword) => searchText.includes(keyword.toLowerCase()));
      });
    }

    // Filter by date range
    if (criteria.dateFrom) {
      results = results.filter((s) => s.createdAt >= criteria.dateFrom!);
    }

    if (criteria.dateTo) {
      results = results.filter((s) => s.createdAt <= criteria.dateTo!);
    }

    // Filter by status
    if (criteria.status) {
      results = results.filter((s) => s.status === criteria.status);
    }

    // Sort by creation date (newest first)
    results.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

    return results;
  }

  /**
   * Get all sessions
   */
  public getAllSessions(): DelegationSession[] {
    return Array.from(this.sessionIndex.values()).sort(
      (a, b) => b.createdAt.getTime() - a.createdAt.getTime()
    );
  }

  /**
   * Reuse context bundle from previous session
   * Requirement 8.4: Allow reusing context bundles from previous sessions
   * Task 15.3: Implement context reuse
   */
  public reuseContextBundle(sessionId: string, roundNumber?: number): string | null {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      return null;
    }

    if (roundNumber !== undefined) {
      const round = session.rounds.find((r) => r.roundNumber === roundNumber);
      return round?.request || null;
    }

    // Return context from first round
    return session.rounds[0]?.request || null;
  }

  /**
   * Track artifact usage
   * Requirement 8.5: Track artifact usage (which artifacts were implemented, modified)
   */
  public trackArtifactUsage(
    sessionId: string,
    artifactId: string,
    status: 'implemented' | 'modified' | 'rejected'
  ): void {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }

    // Find artifact and update metadata
    for (const round of session.rounds) {
      const artifact = round.artifacts.find((a) => a.id === artifactId);
      if (artifact) {
        if (!artifact.metadata.usage) {
          artifact.metadata.usage = { status, updatedAt: new Date() };
        } else {
          artifact.metadata.usage.status = status;
          artifact.metadata.usage.updatedAt = new Date();
        }
        break;
      }
    }

    session.updatedAt = new Date();
    this.saveSession(session);
  }

  /**
   * Generate session summary report
   * Requirement 8.6: Generate session reports summarizing delegation outcomes
   * Task 15.4: Create session reporting
   */
  public generateSessionReport(sessionId: string): SessionReport | null {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      return null;
    }

    // Determine implementation status
    let implementationStatus: 'not_started' | 'in_progress' | 'completed' = 'not_started';
    const implementedCount = this.countArtifactsByStatus(session, 'implemented');
    const totalArtifacts = session.finalArtifacts.length;

    if (implementedCount > 0 && implementedCount < totalArtifacts) {
      implementationStatus = 'in_progress';
    } else if (implementedCount === totalArtifacts && totalArtifacts > 0) {
      implementationStatus = 'completed';
    }

    return {
      sessionId: session.id,
      problemTitle: session.problem.title,
      problemType: session.problem.type,
      complexity: session.problem.complexity,
      createdAt: session.createdAt,
      completedAt: session.completedAt,
      roundCount: session.metrics.roundCount,
      artifactCount: totalArtifacts,
      implementationStatus,
      completenessScore: session.metrics.finalCompleteness,
    };
  }

  /**
   * Generate summary report for all sessions
   */
  public generateAllSessionsReport(): SessionReport[] {
    const reports: SessionReport[] = [];

    for (const session of this.sessionIndex.values()) {
      const report = this.generateSessionReport(session.id);
      if (report) {
        reports.push(report);
      }
    }

    return reports.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  /**
   * Export session as JSON
   */
  public exportSession(sessionId: string): string | null {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      return null;
    }

    return JSON.stringify(session, null, 2);
  }

  /**
   * Delete session
   */
  public deleteSession(sessionId: string): boolean {
    const session = this.sessionIndex.get(sessionId);
    if (!session) {
      return false;
    }

    // Delete session directory
    const sessionDir = path.join(this.sessionsDir, sessionId);
    if (fs.existsSync(sessionDir)) {
      fs.rmSync(sessionDir, { recursive: true, force: true });
    }

    this.sessionIndex.delete(sessionId);
    return true;
  }

  /**
   * Save session to disk
   * Requirement 8.2: Implement session persistence (JSON format)
   */
  private saveSession(session: DelegationSession): void {
    const sessionDir = path.join(this.sessionsDir, session.id);
    if (!fs.existsSync(sessionDir)) {
      fs.mkdirSync(sessionDir, { recursive: true });
    }

    const sessionPath = path.join(sessionDir, 'session.json');
    fs.writeFileSync(sessionPath, JSON.stringify(session, null, 2), 'utf-8');
  }

  /**
   * Save round data to disk
   */
  private saveRound(sessionId: string, round: DelegationRound): void {
    const roundsDir = path.join(this.sessionsDir, sessionId, 'rounds');
    if (!fs.existsSync(roundsDir)) {
      fs.mkdirSync(roundsDir, { recursive: true });
    }

    const roundNumber = round.roundNumber.toString().padStart(2, '0');

    // Save request
    const requestPath = path.join(roundsDir, `${roundNumber}_request.md`);
    fs.writeFileSync(requestPath, round.request, 'utf-8');

    // Save response
    const responsePath = path.join(roundsDir, `${roundNumber}_response.md`);
    fs.writeFileSync(responsePath, round.response, 'utf-8');

    // Save artifacts
    const artifactsPath = path.join(roundsDir, `${roundNumber}_artifacts.json`);
    fs.writeFileSync(artifactsPath, JSON.stringify(round.artifacts, null, 2), 'utf-8');
  }

  /**
   * Count artifacts by usage status
   */
  private countArtifactsByStatus(
    session: DelegationSession,
    status: 'implemented' | 'modified' | 'rejected'
  ): number {
    let count = 0;

    for (const round of session.rounds) {
      for (const artifact of round.artifacts) {
        if (artifact.metadata.usage?.status === status) {
          count++;
        }
      }
    }

    return count;
  }

  /**
   * Generate unique session ID
   */
  private generateSessionId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 9);
    return `session-${timestamp}-${random}`;
  }

  /**
   * Get session statistics
   */
  public getStatistics(): {
    totalSessions: number;
    activeSessions: number;
    completedSessions: number;
    averageRounds: number;
    averageCompleteness: number;
  } {
    const sessions = Array.from(this.sessionIndex.values());
    const activeSessions = sessions.filter((s) => s.status === 'active').length;
    const completedSessions = sessions.filter((s) => s.status === 'completed').length;

    const totalRounds = sessions.reduce((sum, s) => sum + s.metrics.roundCount, 0);
    const averageRounds = sessions.length > 0 ? totalRounds / sessions.length : 0;

    const completedSessionsWithScore = sessions.filter(
      (s) => s.status === 'completed' && s.metrics.finalCompleteness > 0
    );
    const totalCompleteness = completedSessionsWithScore.reduce(
      (sum, s) => sum + s.metrics.finalCompleteness,
      0
    );
    const averageCompleteness =
      completedSessionsWithScore.length > 0
        ? totalCompleteness / completedSessionsWithScore.length
        : 0;

    return {
      totalSessions: sessions.length,
      activeSessions,
      completedSessions,
      averageRounds: Math.round(averageRounds * 10) / 10,
      averageCompleteness: Math.round(averageCompleteness * 10) / 10,
    };
  }
}
