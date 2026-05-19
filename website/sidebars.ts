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
        'DOCS_INDEX',
        'QUICK_REFERENCE',
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
      label: 'Models & Performance',
      items: [
        'FOUNDATION_MODELS',
        'INFERENCE_OPTIMIZATION',
        'BENCHMARK_SYSTEM',
        'PERFORMANCE_COMPARISON',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [
        'DEPLOYMENT',
        'SECURITY_HARDENING',
        'CLINICAL_VALIDATION',
        'TESTING',
        'TROUBLESHOOTING',
      ],
    },
  ],
};

export default sidebars;
