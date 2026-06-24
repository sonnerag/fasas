import os
import re
import glob

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # We need to change the body background and header/button backgrounds.
    # It might be tricky to distinguish them with a simple regex since they currently have the SAME value.
    # Let's find "body {" and ".header {" etc.

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
    # Assuming primary buttons use the same old gradient or specific classes like .btn-primary
    content = re.sub(
        r'(\.btn[^}]*background(?:-image)?:\s*)linear-gradient\([^)]+\)',
        r'\1linear-gradient(135deg, #0284c7 0%, #0369a1 100%)',
        content,
        flags=re.DOTALL
    )
    
    # 4. Button hover states
    # Often buttons have hover states. Let's see if there are hover gradients.
    content = re.sub(
        r'(\.btn:hover[^}]*background(?:-image)?:\s*)linear-gradient\([^)]+\)',
        r'\1linear-gradient(135deg, #0369a1 0%, #0ea5e9 100%)', # A slightly lighter blue for hover
        content,
        flags=re.DOTALL
    )

    # If there are any remaining old gradients, let's just make them the Royal Navy
    # This might catch other headers or elements.
    content = content.replace('linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'linear-gradient(135deg, #0284c7 0%, #0369a1 100%)')

    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Updated {filepath}")

for filepath in glob.glob('templates/*.html'):
    process_file(filepath)
