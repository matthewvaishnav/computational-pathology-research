// Configuration management for Opus Delegation System

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

export interface Config {
  context: {
    max_size: number;
    extraction: {
      include_patterns: string[];
      exclude_patterns: string[];
    };
    summarization: {
      enabled: boolean;
      max_doc_size: number;
    };
  };
  validation: {
    completeness_threshold: number;
    quality_threshold: number;
    auto_followup: boolean;
  };
  export: {
    diagram_format: 'svg' | 'png' | 'pdf';
    api_docs_generator: 'redoc' | 'swagger-ui';
  };
  spec_integration: {
    enabled: boolean;
    output_dir: string;
    templates: {
      requirements: string;
      design: string;
      tasks: string;
    };
  };
}

const DEFAULT_CONFIG: Config = {
  context: {
    max_size: 50000,
    extraction: {
      include_patterns: ['src/**/*.ts', 'src/**/*.py', 'docs/**/*.md'],
      exclude_patterns: ['**/*.test.ts', '**/node_modules/**', '**/__pycache__/**'],
    },
    summarization: {
      enabled: true,
      max_doc_size: 5000,
    },
  },
  validation: {
    completeness_threshold: 80,
    quality_threshold: 70,
    auto_followup: true,
  },
  export: {
    diagram_format: 'svg',
    api_docs_generator: 'redoc',
  },
  spec_integration: {
    enabled: true,
    output_dir: 'specs/',
    templates: {
      requirements: 'templates/requirements.md.j2',
      design: 'templates/design.md.j2',
      tasks: 'templates/tasks.md.j2',
    },
  },
};

export class ConfigManager {
  private config: Config;
  private configPath: string;

  constructor(configPath: string = '.opus-delegation/config.yaml') {
    this.configPath = configPath;
    this.config = this.loadConfig();
  }

  private loadConfig(): Config {
    try {
      if (fs.existsSync(this.configPath)) {
        const fileContent = fs.readFileSync(this.configPath, 'utf-8');
        
        // Validate size
        const MAX_SIZE = 1024 * 1024; // 1MB
        if (fileContent.length > MAX_SIZE) {
          throw new Error('Config file too large');
        }
        
        // Check for YAML bomb patterns
        const anchorCount = (fileContent.match(/&/g) || []).length;
        if (anchorCount > 100) {
          throw new Error('Too many YAML anchors in config (potential bomb)');
        }
        
        const loadedConfig = yaml.load(fileContent) as Partial<Config>;

        // Merge with defaults
        return this.mergeWithDefaults(loadedConfig);
      }
    } catch (error) {
      console.warn(`Failed to load config from ${this.configPath}: ${error}`);
    }

    // Return defaults if file doesn't exist or load fails
    return { ...DEFAULT_CONFIG };
  }

  private mergeWithDefaults(partial: Partial<Config>): Config {
    return {
      context: {
        ...DEFAULT_CONFIG.context,
        ...partial.context,
        extraction: {
          ...DEFAULT_CONFIG.context.extraction,
          ...partial.context?.extraction,
        },
        summarization: {
          ...DEFAULT_CONFIG.context.summarization,
          ...partial.context?.summarization,
        },
      },
      validation: {
        ...DEFAULT_CONFIG.validation,
        ...partial.validation,
      },
      export: {
        ...DEFAULT_CONFIG.export,
        ...partial.export,
      },
      spec_integration: {
        ...DEFAULT_CONFIG.spec_integration,
        ...partial.spec_integration,
        templates: {
          ...DEFAULT_CONFIG.spec_integration.templates,
          ...partial.spec_integration?.templates,
        },
      },
    };
  }

  getConfig(): Config {
    return { ...this.config };
  }

  get(key: string): unknown {
    const keys = key.split('.');
    let value: unknown = this.config;

    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = (value as Record<string, unknown>)[k];
      } else {
        return undefined;
      }
    }

    return value;
  }

  set(key: string, value: unknown): void {
    const keys = key.split('.');
    let current: Record<string, unknown> = this.config as unknown as Record<string, unknown>;

    for (let i = 0; i < keys.length - 1; i++) {
      const k = keys[i];
      if (!(k in current) || typeof current[k] !== 'object') {
        current[k] = {};
      }
      current = current[k] as Record<string, unknown>;
    }

    current[keys[keys.length - 1]] = value;
  }

  save(): void {
    try {
      const dir = path.dirname(this.configPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      const yamlContent = yaml.dump(this.config, {
        indent: 2,
        lineWidth: 100,
      });

      fs.writeFileSync(this.configPath, yamlContent, 'utf-8');
    } catch (error) {
      throw new Error(`Failed to save config to ${this.configPath}: ${error}`);
    }
  }

  initializeDefaultConfig(): void {
    this.config = { ...DEFAULT_CONFIG };
    this.save();
  }

  validateConfig(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate context settings
    if (this.config.context.max_size <= 0) {
      errors.push('context.max_size must be positive');
    }

    if (this.config.context.summarization.max_doc_size <= 0) {
      errors.push('context.summarization.max_doc_size must be positive');
    }

    // Validate validation thresholds
    if (
      this.config.validation.completeness_threshold < 0 ||
      this.config.validation.completeness_threshold > 100
    ) {
      errors.push('validation.completeness_threshold must be between 0 and 100');
    }

    if (
      this.config.validation.quality_threshold < 0 ||
      this.config.validation.quality_threshold > 100
    ) {
      errors.push('validation.quality_threshold must be between 0 and 100');
    }

    // Validate export settings
    const validDiagramFormats = ['svg', 'png', 'pdf'];
    if (!validDiagramFormats.includes(this.config.export.diagram_format)) {
      errors.push(
        `export.diagram_format must be one of: ${validDiagramFormats.join(', ')}`
      );
    }

    const validApiDocsGenerators = ['redoc', 'swagger-ui'];
    if (!validApiDocsGenerators.includes(this.config.export.api_docs_generator)) {
      errors.push(
        `export.api_docs_generator must be one of: ${validApiDocsGenerators.join(', ')}`
      );
    }

    return {
      valid: errors.length === 0,
      errors,
    };
  }

  // Environment variable overrides
  applyEnvironmentOverrides(): void {
    const envMappings: Record<string, string> = {
      OPUS_CONTEXT_MAX_SIZE: 'context.max_size',
      OPUS_COMPLETENESS_THRESHOLD: 'validation.completeness_threshold',
      OPUS_QUALITY_THRESHOLD: 'validation.quality_threshold',
      OPUS_DIAGRAM_FORMAT: 'export.diagram_format',
      OPUS_SPEC_OUTPUT_DIR: 'spec_integration.output_dir',
    };

    for (const [envVar, configKey] of Object.entries(envMappings)) {
      const value = process.env[envVar];
      if (value !== undefined) {
        // Parse numeric values
        if (configKey.includes('size') || configKey.includes('threshold')) {
          this.set(configKey, parseInt(value, 10));
        } else {
          this.set(configKey, value);
        }
      }
    }
  }
}

export function loadConfig(configPath?: string): Config {
  const manager = new ConfigManager(configPath);
  manager.applyEnvironmentOverrides();
  return manager.getConfig();
}
