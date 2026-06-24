import os
import re
import glob

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Update body background
    content = re.sub(
        r'(body\s*\{[^}]*background:\s*)linear-gradient\([^)]+\)',
        r'\1linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%)',
        content,
        flags=re.DOTALL
    )

    # 2. Update .header background
    content = re.sub(
        r'(\.header\s*\{[^}]*background:\s*)linear-gradient\([^)]+\)',
        r'\1linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
        content,
        flags=re.DOTALL
    )

    # 3. Primary buttons background
    content = re.sub(
        r'(\.btn[^}]*background(?:-image)?:\s*)linear-gradient\([^)]+\)',
        r'\1linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
        content,
        flags=re.DOTALL
    )
    
    # 4. Button hover states
    content = re.sub(
        r'(\.btn:hover[^}]*background(?:-image)?:\s*)linear-gradient\([^)]+\)',
        r'\1linear-gradient(135deg, #0369a1 0%, #0ea5e9 100%)',
        content,
        flags=re.DOTALL
    )

    # 5. Catch any lingering Emerald colors
    content = content.replace('linear-gradient(135deg, #047857 0%, #064e3b 100%)', 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)')
    content = content.replace('linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%)', 'linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%)')

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Updated {filepath}")

# Process templates
for filepath in glob.glob('templates/*.html'):
    process_file(filepath)

# Process static html files
for filepath in glob.glob('static/*.html'):
    process_file(filepath)
