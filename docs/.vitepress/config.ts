import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Computational Pathology Research',
  description: 'Computational pathology, TransnnMIL v2.0, PathologyFL, FAIR-WEIGHTS-H, and validation reports.',
  base: '/computational-pathology-research/',
  cleanUrls: true,
  lastUpdated: true,
  themeConfig: {
    logo: undefined,
    siteTitle: 'Computational Pathology Research',
    nav: [
      { text: 'Overview', link: '/overview/' },
      { text: 'Models', link: '/models/' },
      { text: 'Federated', link: '/federated/pathologyfl' },
      { text: 'Validation', link: '/validation/' },
      { text: 'Results', link: '/results/pcam-results' },
      { text: 'Roadmap', link: '/roadmap/' }
    ],
    sidebar: [
      {
        text: 'Overview',
        items: [
          { text: 'Home', link: '/' },
          { text: 'What this project is', link: '/overview/' },
          { text: 'Repository structure', link: '/repository-overview' },
          { text: 'Platform overview', link: '/platform-overview' },
          { text: 'Claim status', link: '/overview/claim-status' }
        ]
      },
      {
        text: 'Quickstart',
        items: [
          { text: 'Getting started', link: '/getting-started' },
          { text: 'Reproducing experiments', link: '/quickstart/reproducing-experiments' },
          { text: 'Running benchmarks', link: '/quickstart/running-benchmarks' }
        ]
      },
      {
        text: 'Models',
        items: [
          { text: 'Models overview', link: '/models/' },
          { text: 'TransnnMIL v2.0', link: '/models/transnnmil-v2' },
          { text: 'Foundation encoders', link: '/models/foundation-encoders' },
          { text: 'Model cards', link: '/models/model-cards' }
        ]
      },
      {
        text: 'Federated Learning',
        items: [
          { text: 'PathologyFL', link: '/federated/pathologyfl' },
          { text: 'FAIR-WEIGHTS-H', link: '/theory/fair-weights-h' },
          { text: 'Implementation status', link: '/theory/implementation-status' },
          { text: 'Privacy and DP', link: '/federated/privacy-dp' },
          { text: 'Secure aggregation', link: '/federated/secure-aggregation' },
          { text: 'Byzantine robustness', link: '/federated/byzantine-robustness' }
        ]
      },
      {
        text: 'Validation',
        items: [
          { text: 'Validation overview', link: '/validation/' },
          { text: 'Synthetic validation', link: '/validation/synthetic-report' },
          { text: 'PCam smoke report', link: '/validation/pcam-smoke-report' },
          { text: 'PCam benchmark plan', link: '/validation/PCAM_FEDERATED_BENCHMARK_PLAN' },
          { text: 'Camelyon17 plan', link: '/validation/camelyon17-plan' }
        ]
      },
      {
        text: 'Results',
        items: [
          { text: 'PCam results', link: '/results/pcam-results' },
          { text: 'Performance comparison', link: '/results/performance-comparison' }
        ]
      },
      {
        text: 'Engineering',
        items: [
          { text: 'Architecture', link: '/engineering/architecture' },
          { text: 'Testing status', link: '/engineering/testing-status' },
          { text: 'Benchmark system', link: '/engineering/benchmark-system' },
          { text: 'Security hardening', link: '/engineering/security-hardening' },
          { text: 'Deployment', link: '/engineering/deployment' }
        ]
      },
      {
        text: 'Roadmap',
        items: [
          { text: 'Roadmap', link: '/roadmap/' },
          { text: 'Limitations', link: '/roadmap/limitations' },
          { text: 'Changelog', link: '/changelog' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/matthewvaishnav/computational-pathology-research' }
    ],
    search: {
      provider: 'local'
    },
    footer: {
      message: 'Research documentation. Not clinical validation or regulatory clearance.',
      copyright: 'Computational Pathology Research'
    }
  },
  markdown: {
    math: true
  }
})
