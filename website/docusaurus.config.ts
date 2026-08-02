import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Computational Pathology Research',
  tagline: 'Paired-acquisition neural research and reproducible evidence auditing',
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
          exclude: ['**/_config.yml', '**/.gitkeep'],
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
      title: 'Computational Pathology Research',
      items: [
        {to: '/docs/CURRENT_STATUS', label: 'Current Status', position: 'left'},
        {to: '/docs/DOCS_INDEX', label: 'Documentation', position: 'left'},
        {to: '/docs/PORTFOLIO_SUMMARY', label: 'Portfolio', position: 'left'},
        {type: 'search', position: 'right'},
        {
          href: 'https://github.com/matthewvaishnav/computational-pathology-research/blob/main/CLAIM_BOUNDARY.md',
          label: 'Claim Boundary',
          position: 'right',
        },
        {
          href: 'https://github.com/matthewvaishnav/computational-pathology-research',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    announcementBar: {
      id: 'scientific-audit-20260802',
      content:
        'Scientific audit active: the claim boundary and current-status page override older platform, benchmark, manuscript, and deployment language.',
      backgroundColor: '#fff3cd',
      textColor: '#664d03',
      isCloseable: true,
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Current Record',
          items: [
            {label: 'Current Status', to: '/docs/CURRENT_STATUS'},
            {label: 'Documentation Index', to: '/docs/DOCS_INDEX'},
            {label: 'Repository Overview', to: '/docs/PLATFORM_OVERVIEW'},
            {label: 'Portfolio Summary', to: '/docs/PORTFOLIO_SUMMARY'},
          ],
        },
        {
          title: 'Research & Evidence',
          items: [
            {label: 'Presentation Abstract', to: '/docs/PRESENTATION_ABSTRACT'},
            {label: 'Data Provenance', to: '/docs/DATA_PROVENANCE'},
            {label: 'PCam Record', to: '/docs/PCAM_REAL_RESULTS'},
            {label: 'Testing', to: '/docs/TESTING'},
          ],
        },
        {
          title: 'External',
          items: [
            {
              label: 'Claim Boundary',
              href: 'https://github.com/matthewvaishnav/computational-pathology-research/blob/main/CLAIM_BOUNDARY.md',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/matthewvaishnav/computational-pathology-research',
            },
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Computational Pathology Research. Research use only; no clinical or deployment claim.`,
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
          'computational pathology, neural networks, paired acquisition, histopathology, representation learning, reproducibility, WSI',
      },
      {
        name: 'description',
        content:
          'Independent computational pathology research on paired acquisition, representation auditing, whole-slide neural models, and reproducible experiment engineering.',
      },
    ],
  } satisfies Preset.ThemeConfig,
};

export default config;
