// Error handling utilities for Opus Delegation System

export class DelegationError extends Error {
  constructor(
    message: string,
    public code: string,
    public recoverable: boolean = true,
    public context?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'DelegationError';
  }
}

export class ContextExtractionError extends DelegationError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'CONTEXT_EXTRACTION_ERROR', true, context);
    this.name = 'ContextExtractionError';
  }
}

export class ParseError extends DelegationError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'PARSE_ERROR', true, context);
    this.name = 'ParseError';
  }
}

export class ValidationError extends DelegationError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'VALIDATION_ERROR', true, context);
    this.name = 'ValidationError';
  }
}

export class SessionError extends DelegationError {
  constructor(message: string, context?: Record<string, unknown>) {
    super(message, 'SESSION_ERROR', false, context);
    this.name = 'SessionError';
  }
}

export interface ErrorRecovery {
  error: DelegationError;
  recoveryAction: string;
  partialResult?: unknown;
  suggestions: string[];
}

export class ErrorHandler {
  private errorLog: Array<{
    timestamp: Date;
    error: DelegationError;
    recovery?: ErrorRecovery;
  }> = [];

  handleContextExtractionError(
    error: Error,
    missingFiles: string[],
    extractedFiles: unknown[]
  ): ErrorRecovery {
    const delegationError = new ContextExtractionError(
      `Context extraction failed: ${error.message}`,
      {
        missingFiles,
        extractedCount: Array.isArray(extractedFiles) ? extractedFiles.length : 0,
      }
    );

    const recovery: ErrorRecovery = {
      error: delegationError,
      recoveryAction: 'partial_extraction',
      partialResult: extractedFiles,
      suggestions: [
        'Check file permissions',
        'Verify file paths exist',
        'Reduce extraction scope',
        `Missing files: ${missingFiles.join(', ')}`,
      ],
    };

    this.logError(delegationError, recovery);
    return recovery;
  }

  handleParseError(
    error: Error,
    response: string,
    location?: { start: number; end: number }
  ): ErrorRecovery {
    const delegationError = new ParseError(`Parse failed: ${error.message}`, {
      location,
      responseLength: response.length,
    });

    const problematicSection = location
      ? response.substring(location.start, Math.min(location.end, location.start + 200))
      : response.substring(0, 200);

    const recovery: ErrorRecovery = {
      error: delegationError,
      recoveryAction: 'identify_problematic_section',
      partialResult: { problematicSection, location },
      suggestions: [
        'Check markdown formatting',
        'Verify code block syntax',
        'Ensure YAML/JSON validity',
        `Problematic section: ${problematicSection}...`,
      ],
    };

    this.logError(delegationError, recovery);
    return recovery;
  }

  handleValidationError(
    artifactId: string,
    issues: Array<{ severity: string; message: string }>
  ): ErrorRecovery {
    const delegationError = new ValidationError(
      `Validation failed for artifact ${artifactId}`,
      {
        artifactId,
        issueCount: issues.length,
      }
    );

    const recovery: ErrorRecovery = {
      error: delegationError,
      recoveryAction: 'generate_followup',
      suggestions: issues.map((issue) => `${issue.severity}: ${issue.message}`),
    };

    this.logError(delegationError, recovery);
    return recovery;
  }

  handleSessionError(sessionId: string, error: Error): ErrorRecovery {
    const delegationError = new SessionError(`Session error: ${error.message}`, {
      sessionId,
    });

    const recovery: ErrorRecovery = {
      error: delegationError,
      recoveryAction: 'restore_from_checkpoint',
      suggestions: [
        'Check session data integrity',
        'Restore from last checkpoint',
        'Create new session if corruption detected',
      ],
    };

    this.logError(delegationError, recovery);
    return recovery;
  }

  private logError(error: DelegationError, recovery?: ErrorRecovery): void {
    this.errorLog.push({
      timestamp: new Date(),
      error,
      recovery,
    });
  }

  getErrorLog(): Array<{
    timestamp: Date;
    error: DelegationError;
    recovery?: ErrorRecovery;
  }> {
    return [...this.errorLog];
  }

  clearErrorLog(): void {
    this.errorLog = [];
  }
}
