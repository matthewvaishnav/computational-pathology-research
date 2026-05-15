import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Platform',
      collapsed: false,
      items: [
        'DOCS_INDEX',
        'GETTING_STARTED',
        'FRAMEWORK_OVERVIEW',
        'ARCHITECTURE',
        'API_REFERENCE',
      ],
    },
    {
      type: 'category',
      label: 'Modeling',
      items: [
        'FOUNDATION_MODELS',
        'INFERENCE_OPTIMIZATION',
        'BENCHMARK_SYSTEM',
        'PERFORMANCE_COMPARISON',
        'TRANSNNMIL_IMPLEMENTATION',
        'MODEL_INTERPRETABILITY',
      ],
    },
    {
      type: 'category',
      label: 'Production',
      items: [
        'DEPLOYMENT',
        'SECURITY_HARDENING',
        'CLINICAL_VALIDATION',
        'TESTING',
        'PACS_INTEGRATION',
        'QUICK_REFERENCE',
        'TROUBLESHOOTING',
      ],
    },
    {
      type: 'category',
      label: 'Project',
      items: [
        'PROJECT_STATUS',
        'ROADMAP',
        'CURRENT_STATUS_2026-05-14',
        'CHANGELOG',
        'CONTRIBUTING',
        'STYLE_GUIDE',
      ],
    },
    {
      type: 'category',
      label: 'Analysis',
      items: [
        'CODE_OPTIMIZATIONS',
        'OPTIMIZATION_SUMMARY',
        'PERFORMANCE',
        'GPU_AND_TRAINING_FIXES',
        'MISSING_ITEMS_ANALYSIS',
        'UNPUBLISHED_BENCHMARKS_INVENTORY',
      ],
    },
    {
      type: 'category',
      label: 'Resources',
      items: [
        'DEPENDENCIES',
        'SECURITY_AUDIT',
        'HistoCore_Presentation',
        'PRESENTATION_ABSTRACT',
        'LINKEDIN_UPDATES',
      ],
    },
  ],
};

export default sidebars;
