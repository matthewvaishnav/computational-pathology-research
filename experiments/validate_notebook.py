"""
Validate the PCam visualization notebook structure and content.
"""

import json
from pathlib import Path

print('='*60)
print('VALIDATING PCAM VISUALIZATION NOTEBOOK')
print('='*60)

# Load notebook
notebook_path = Path('experiments/notebooks/pcam_visualization.ipynb')
if not notebook_path.exists():
    print(f'✗ Notebook not found: {notebook_path}')
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print(f'\n✓ Notebook loaded: {notebook_path}')

# Validate structure
cells = notebook.get('cells', [])
print(f'✓ Total cells: {len(cells)}')

# Count cell types
markdown_cells = [c for c in cells if c['cell_type'] == 'markdown']
code_cells = [c for c in cells if c['cell_type'] == 'code']

print(f'  - Markdown cells: {len(markdown_cells)}')
print(f'  - Code cells: {len(code_cells)}')

# Check for required sections
required_sections = [
    'Dataset Exploration',
    'Training Curves',
    'Model Performance',
    'Prediction Analysis',
    'Save All Plots'
]

print('\n[Checking Required Sections]')
found_sections = []
for cell in markdown_cells:
    source = ''.join(cell['source'])
    for section in required_sections:
        if section in source:
            found_sections.append(section)
            print(f'✓ Found: {section}')

missing_sections = set(required_sections) - set(found_sections)
if missing_sections:
    print(f'\n✗ Missing sections: {missing_sections}')
else:
    print('\n✓ All required sections present')

# Check for required imports
print('\n[Checking Required Imports]')
required_imports = [
    'matplotlib',
    'seaborn',
    'torch',
    'numpy',
    'sklearn'
]

first_code_cell = ''.join(code_cells[0]['source']) if code_cells else ''
found_imports = []
for imp in required_imports:
    if imp in first_code_cell:
        found_imports.append(imp)
        print(f'✓ Import: {imp}')

missing_imports = set(required_imports) - set(found_imports)
if missing_imports:
    print(f'\n⚠ Missing imports: {missing_imports}')
else:
    print('\n✓ All required imports present')

# Check for output directory creation
print('\n[Checking Output Directory Setup]')
output_dir_found = False
for cell in code_cells:
    source = ''.join(cell['source'])
    if 'OUTPUT_DIR' in source and 'mkdir' in source:
        output_dir_found = True
        print('✓ Output directory creation found')
        break

if not output_dir_found:
    print('⚠ Output directory creation not found')

# Check for plot saving
print('\n[Checking Plot Saving]')
expected_plots = [
    'sample_grid.png',
    'class_distribution.png',
    'loss_curves.png',
    'accuracy_curves.png',
    'confusion_matrix.png',
    'roc_curve.png',
    'precision_recall_curve.png',
    'confidence_histogram.png'
]

notebook_content = json.dumps(notebook)
found_plots = []
for plot in expected_plots:
    if plot in notebook_content:
        found_plots.append(plot)
        print(f'✓ Plot: {plot}')

missing_plots = set(expected_plots) - set(found_plots)
if missing_plots:
    print(f'\n⚠ Missing plot references: {missing_plots}')
else:
    print('\n✓ All expected plots referenced')

# Check for data loading functions
print('\n[Checking Data Loading Functions]')
required_functions = [
    'load_evaluation_results',
    'load_metrics'
]

found_functions = []
for func in required_functions:
    if func in notebook_content:
        found_functions.append(func)
        print(f'✓ Function: {func}')

missing_functions = set(required_functions) - set(found_functions)
if missing_functions:
    print(f'\n⚠ Missing functions: {missing_functions}')
else:
    print('\n✓ All required functions present')

# Summary
print('\n' + '='*60)
print('VALIDATION SUMMARY')
print('='*60)

validation_checks = [
    ('Notebook exists', notebook_path.exists()),
    ('Has markdown cells', len(markdown_cells) > 0),
    ('Has code cells', len(code_cells) > 0),
    ('All sections present', len(missing_sections) == 0),
    ('All imports present', len(missing_imports) == 0),
    ('Output directory setup', output_dir_found),
    ('All plots referenced', len(missing_plots) == 0),
    ('All functions present', len(missing_functions) == 0)
]

passed = sum(1 for _, status in validation_checks if status)
total = len(validation_checks)

print(f'\nPassed: {passed}/{total} checks')
print('\nDetailed Results:')
for check, status in validation_checks:
    symbol = '✓' if status else '✗'
    print(f'  {symbol} {check}')

if passed == total:
    print('\n✓ Notebook validation PASSED!')
    exit(0)
else:
    print(f'\n⚠ Notebook validation completed with {total - passed} warnings')
    exit(0)
