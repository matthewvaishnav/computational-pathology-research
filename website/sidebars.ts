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
        'QUICK_REFERENCE',
        'TROUBLESHOOTING',
      ],
    },
  ],
};

export default sidebars;
