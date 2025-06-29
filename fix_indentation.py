#!/usr/bin/env python3
"""
Fix indentation errors in DanzarVLM.py
"""

def fix_indentation():
    """Fix indentation errors in DanzarVLM.py"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix line 1673 (remove extra spaces before comment)
    if len(lines) > 1672:
        lines[1672] = '            # Initialize LangChain Tools Service for agentic behavior\n'
    
    # Fix line 1694 (fix except statement indentation)
    if len(lines) > 1693:
        lines[1693] = '            except Exception as e:\n'
    
    # Fix line 1697 (fix outer except statement indentation)
    if len(lines) > 1696:
        lines[1696] = '        except Exception as e:\n'
    
    # Write the fixed file
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("✅ Fixed indentation errors in DanzarVLM.py")

if __name__ == "__main__":
    fix_indentation() 