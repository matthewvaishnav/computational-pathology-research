import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Current Public Record',
      collapsed: false,
      items: [
        'CURRENT_STATUS',
        'DOCS_INDEX',
        'PLATFORM_OVERVIEW',
        'PORTFOLIO_SUMMARY',
        'PRESENTATION_ABSTRACT',
        'DATA_PROVENANCE',
      ],
    },
    {
      type: 'category',
      label: 'Research Evidence',
      items: [
        'PCAM_REAL_RESULTS',
        'TESTING',
      ],
    },
    {
      type: 'category',
      label: 'Engineering References',
      items: [
        'REPOSITORY_OVERVIEW',
        'FOUNDATION_MODELS',
        'MODEL_INTERPRETABILITY',
        'QUICK_REFERENCE',
        'TROUBLESHOOTING',
      ],
    },
  ],
};

export default sidebars;
