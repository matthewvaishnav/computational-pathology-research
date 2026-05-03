/**
 * Script to generate built-in templates
 * Run with: npx tsx scripts/generate-templates.ts
 */

import { TemplateLibrary } from '../src/components/TemplateLibrary.js';

const library = new TemplateLibrary('./templates');
library.createBuiltInTemplates();

console.log('✅ Built-in templates generated successfully!');
console.log('\nGenerated templates:');
const templates = library.listTemplates();
templates.forEach(t => {
  console.log(`  - ${t.template_id} (${t.name}) v${t.version}`);
});
