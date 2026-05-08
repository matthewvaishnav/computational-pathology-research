import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'HistoCore Research Platform',
  tagline: 'Computational Pathology Research Platform for Clinical AI Deployment',
  favicon: 'img/favicon.svg',
  future: {
    v4: true,
  },
  url: 'https://matthewvaishnav.github.io',
  baseUrl: '/computational-pathology-research/',
  organizationName: 'matthewvaishnav',
  projectName: 'computational-pathology-research',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
  onBrokenLinks: 'warn',
  staticDirectories: ['static'],
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenMarkdownImages: 'warn',
    },
  },
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  themes: ['@easyops-cn/docusaurus-search-local'],
  plugins: [
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {
            from: ['/docs.html'],
            to: '/docs/',
          },
        ],
        createRedirects(existingPath: string) {
          if (!existingPath.startsWith('/docs/')) {
            return undefined;
          }

          const legacyPath = existingPath.replace(/^\/docs/, '');
          if (!legacyPath || legacyPath === '/') {
            return undefined;
          }

          const redirects = new Set<string>([legacyPath, `${legacyPath}.html`]);

          if (legacyPath.endsWith('/README')) {
            const directoryPath = legacyPath.slice(0, -'/README'.length) || '/';
            redirects.add(directoryPath);
            redirects.add(`${directoryPath}.html`);
            if (directoryPath !== '/') {
              redirects.add(`${directoryPath}/index.html`);
            }
          }

          return [...redirects].filter((redirectPath) => redirectPath !== existingPath);
        },
      },
    ],
  ],
  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          exclude: [
            '**/_config.yml',
            '**/.gitkeep',
          ],
          showLastUpdateTime: true,
          editUrl:
            'https://github.com/matthewvaishnav/computational-pathology-research/tree/main/',
        },
        blog: false,
        pages: {
          path: 'src/pages',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  themeConfig: {
    image: 'img/og-card.png',
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      hideOnScroll: false,
      title: 'HistoCore Research',
      items: [
        {to: '/docs/GETTING_STARTED', label: 'Documentation', position: 'left'},
        {to: '/docs/FOUNDATION_MODELS', label: 'Models', position: 'left'},
        {to: '/docs/DEPLOYMENT', label: 'Deployment', position: 'left'},
        {to: '/docs/BENCHMARK_SYSTEM', label: 'Benchmarks', position: 'left'},
        {type: 'search', position: 'right'},
        {
          href: 'https://github.com/matthewvaishnav/computational-pathology-research',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    announcementBar: {
      id: 'research-platform',
      content:
        '📚 Research Platform - Comprehensive computational pathology framework for clinical AI deployment',
      backgroundColor: '#e3f2fd',
      textColor: '#0d47a1',
      isCloseable: true,
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Getting Started', to: '/docs/GETTING_STARTED'},
            {label: 'Architecture', to: '/docs/ARCHITECTURE'},
            {label: 'API Reference', to: '/docs/API_REFERENCE'},
            {label: 'Troubleshooting', to: '/docs/TROUBLESHOOTING'},
          ],
        },
        {
          title: 'Platform',
          items: [
            {label: 'Foundation Models', to: '/docs/FOUNDATION_MODELS'},
            {label: 'Inference', to: '/docs/INFERENCE_OPTIMIZATION'},
            {label: 'Deployment', to: '/docs/DEPLOYMENT'},
            {label: 'Security', to: '/docs/SECURITY_HARDENING'},
          ],
        },
        {
          title: 'Resources',
          items: [
            {label: 'Benchmarks', to: '/docs/BENCHMARK_SYSTEM'},
            {label: 'Testing', to: '/docs/TESTING'},
            {label: 'Clinical Validation', to: '/docs/CLINICAL_VALIDATION'},
            {label: 'GitHub', href: 'https://github.com/matthewvaishnav/computational-pathology-research'},
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} HistoCore Research Platform. Built for computational pathology research.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ['bash', 'json', 'yaml', 'toml', 'diff', 'python'],
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    metadata: [
      {
        name: 'keywords',
        content:
          'computational pathology, pytorch, histopathology, WSI, foundation models, clinical AI, PACS, federated learning',
      },
      {
        name: 'description',
        content:
          'HistoCore is a production-grade computational pathology platform with foundation model integration, security compliance, and clinical deployment capabilities.',
      },
    ],
  } satisfies Preset.ThemeConfig,
};

export default config;
