import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Matthew Vaishnav Computational Pathology, Federated Oncology Learning, and Mathematical Validation Infrastructure',
  description: 'Whole-slide pathology AI, TransnnMIL v2.0, PathologyFL, FAIR-WEIGHTS-H institutional weighting, PCam/Camelyon validation, and multi-institutional oncology learning infrastructure.',
  base: '/computational-pathology-research/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,
  srcExclude: [/* existing exclusions preserved */],
  themeConfig: {
    siteTitle: 'Computational Pathology, Federated Oncology Learning & Mathematical Validation',
    nav: [
      { text: 'Overview', link: '/overview/' },
      { text: 'Models', link: '/models/' },
      { text: 'Federated', link: '/federated/pathologyfl' },
      { text: 'Validation', link: '/validation/' },
      { text: 'Results', link: '/results/pcam-results' },
      { text: 'Roadmap', link: '/roadmap/' }
    ]
  }
})
