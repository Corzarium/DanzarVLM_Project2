#!/usr/bin/env python3
"""
Fix syntax error in DanzarVLM.py at line 1950
"""

def fix_syntax_error():
    """Fix the syntax error at line 1950"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix the problematic lines (1949-1952)
    # Replace the invalid else/except structure with proper code
    if len(lines) > 1948:
        # Line 1949: Change "else:" to proper structure
        lines[1948] = '                    self.logger.error("❌ Model client not available for LangChain tools")\n'
        lines[1949] = '                    self.app_context.langchain_tools = None\n'
        lines[1950] = '            except Exception as e:\n'
        lines[1951] = '                self.logger.error(f"❌ LangChain Tools Service error: {e}")\n'
        lines[1952] = '                self.app_context.langchain_tools = None\n'
    
    # Write the fixed file
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("✅ Fixed syntax error in DanzarVLM.py")

if __name__ == "__main__":
    fix_syntax_error() 