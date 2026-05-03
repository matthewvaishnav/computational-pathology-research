// Cost Tracking component for Opus Delegation System

export interface TimeMetrics {
  phaseStart: Date;
  phaseEnd?: Date;
  phaseDuration?: number; // milliseconds
  phaseName: string;
}

export interface CostEstimate {
  contextSize: number;
  estimatedTokens: number;
  estimatedCost: number; // USD
  model: string;
}

export interface EfficiencyMetrics {
  artifactsPerHour: number;
  qualityPerRound: number[];
  averageQuality: number;
  timeToCompletion: number;
}

export interface CostBenefitReport {
  totalTime: number;
  totalCost: number;
  manualEstimate: number;
  timeSaved: number;
  costSaved: number;
  roi: number;
}

export class CostTracking {
  private timeMetrics: Map<string, TimeMetrics[]> = new Map();
  private costEstimates: Map<string, CostEstimate[]> = new Map();

  // Time tracking
  startPhase(sessionId: string, phaseName: string): void {
    const metrics: TimeMetrics = {
      phaseStart: new Date(),
      phaseName,
    };

    if (!this.timeMetrics.has(sessionId)) {
      this.timeMetrics.set(sessionId, []);
    }

    this.timeMetrics.get(sessionId)!.push(metrics);
  }

  endPhase(sessionId: string, phaseName: string): number {
    const metrics = this.timeMetrics.get(sessionId);
    if (!metrics) {
      throw new Error(`No metrics found for session ${sessionId}`);
    }

    const phaseMetric = metrics.find(
      (m) => m.phaseName === phaseName && !m.phaseEnd
    );

    if (!phaseMetric) {
      throw new Error(`No active phase ${phaseName} for session ${sessionId}`);
    }

    phaseMetric.phaseEnd = new Date();
    phaseMetric.phaseDuration =
      phaseMetric.phaseEnd.getTime() - phaseMetric.phaseStart.getTime();

    return phaseMetric.phaseDuration;
  }

  getTotalTime(sessionId: string): number {
    const metrics = this.timeMetrics.get(sessionId);
    if (!metrics) {
      return 0;
    }

    return metrics.reduce((total, m) => total + (m.phaseDuration || 0), 0);
  }

  getPhaseTime(sessionId: string, phaseName: string): number {
    const metrics = this.timeMetrics.get(sessionId);
    if (!metrics) {
      return 0;
    }

    return metrics
      .filter((m) => m.phaseName === phaseName)
      .reduce((total, m) => total + (m.phaseDuration || 0), 0);
  }

  // Cost estimation
  estimateCost(
    contextSize: number,
    model: string = 'claude-opus-4.5'
  ): CostEstimate {
    // Rough token estimation: ~4 chars per token
    const estimatedTokens = Math.ceil(contextSize / 4);

    // Cost per 1M tokens (approximate)
    const costPer1MTokens = model.includes('opus') ? 15.0 : 3.0;

    const estimatedCost = (estimatedTokens / 1_000_000) * costPer1MTokens;

    return {
      contextSize,
      estimatedTokens,
      estimatedCost,
      model,
    };
  }

  trackCost(sessionId: string, contextSize: number, model?: string): void {
    const estimate = this.estimateCost(contextSize, model);

    if (!this.costEstimates.has(sessionId)) {
      this.costEstimates.set(sessionId, []);
    }

    this.costEstimates.get(sessionId)!.push(estimate);
  }

  getTotalCost(sessionId: string): number {
    const estimates = this.costEstimates.get(sessionId);
    if (!estimates) {
      return 0;
    }

    return estimates.reduce((total, e) => total + e.estimatedCost, 0);
  }

  // Efficiency metrics
  calculateEfficiencyMetrics(
    sessionId: string,
    artifactCount: number,
    qualityScores: number[]
  ): EfficiencyMetrics {
    const totalTime = this.getTotalTime(sessionId);
    const hours = totalTime / (1000 * 60 * 60);

    const artifactsPerHour = hours > 0 ? artifactCount / hours : 0;

    const averageQuality =
      qualityScores.length > 0
        ? qualityScores.reduce((sum, q) => sum + q, 0) / qualityScores.length
        : 0;

    return {
      artifactsPerHour,
      qualityPerRound: qualityScores,
      averageQuality,
      timeToCompletion: totalTime,
    };
  }

  // Cost-benefit analysis
  generateCostBenefitReport(
    sessionId: string,
    manualEstimateHours: number = 8,
    hourlyRate: number = 150
  ): CostBenefitReport {
    const totalTime = this.getTotalTime(sessionId);
    const totalCost = this.getTotalCost(sessionId);

    const manualEstimate = manualEstimateHours * hourlyRate;
    const actualHours = totalTime / (1000 * 60 * 60);
    const actualLaborCost = actualHours * hourlyRate;

    const timeSaved = manualEstimateHours - actualHours;
    const costSaved = manualEstimate - (actualLaborCost + totalCost);

    const roi = manualEstimate > 0 ? (costSaved / manualEstimate) * 100 : 0;

    return {
      totalTime,
      totalCost: actualLaborCost + totalCost,
      manualEstimate,
      timeSaved,
      costSaved,
      roi,
    };
  }

  // Export metrics
  exportMetrics(sessionId: string): {
    timeMetrics: TimeMetrics[];
    costEstimates: CostEstimate[];
    totalTime: number;
    totalCost: number;
  } {
    return {
      timeMetrics: this.timeMetrics.get(sessionId) || [],
      costEstimates: this.costEstimates.get(sessionId) || [],
      totalTime: this.getTotalTime(sessionId),
      totalCost: this.getTotalCost(sessionId),
    };
  }

  // Clear session data
  clearSession(sessionId: string): void {
    this.timeMetrics.delete(sessionId);
    this.costEstimates.delete(sessionId);
  }
}
