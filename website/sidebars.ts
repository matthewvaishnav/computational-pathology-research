import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'GETTING_STARTED',
        'REPOSITORY_OVERVIEW',
        'PLATFORM_OVERVIEW',
        'DOCS_INDEX',
        'QUICK_REFERENCE',
      ],
    },
    {
      type: 'category',
      label: 'Results & Benchmarks',
      items: [
        'PCAM_REAL_RESULTS',
        'BENCHMARK_SYSTEM',
        'CURRENT_STATUS_2026-05-14',
        'PERFORMANCE_COMPARISON',
      ],
    },
    {
      type: 'category',
      label: 'Federated Learning',
      items: [
        'FAIR_WEIGHTS_HYBRID_PROTOCOL',
        'FAIR_WEIGHTS_H_IMPLEMENTATION_STATUS',
        'FAIR_WEIGHTS_H_SYNTHETIC_REPORT',
        'ROADMAP_TO_REAL_DATASETS',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'FRAMEWORK_OVERVIEW',
        'ARCHITECTURE',
        'API_REFERENCE',
      ],
    },
    {
      type: 'category',
      label: 'Models & Training',
      items: [
        'FOUNDATION_MODELS',
        'MODEL_INTERPRETABILITY',
        'INFERENCE_OPTIMIZATION',
      ],
    },
    {
      type: 'category',
      label: 'Clinical Integration',
      items: [
        'CLINICAL_WORKFLOW_INTEGRATION',
        'PACS_INTEGRATION',
        'CLINICAL_VALIDATION',
      ],
    },
    {
      type: 'category',
      label: 'Deployment & Security',
      items: [
        'DEPLOYMENT',
        'SECURITY_HARDENING',
        'TESTING',
        'TROUBLESHOOTING',
      ],
    },
  ],
};

export default sidebars;
