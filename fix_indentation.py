#!/usr/bin/env python3
"""
Simple indentation fix script for DanzarVLM.py
"""

import re

def fix_indentation():
    """Fix indentation issues in DanzarVLM.py"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the specific problematic line around line 763
    # Replace the malformed function call
    content = re.sub(
        r'response = None\s+audio_path=temp_file_path\s+\)\s+\)',
        'response = None',
        content
    )
    
    # Fix any other malformed function calls
    content = re.sub(
        r'response = None\s+[^)]*\)\s*\)',
        'response = None',
        content
    )
    
    # Write the fixed content back
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed indentation issues in DanzarVLM.py")

if __name__ == "__main__":
    fix_indentation() 