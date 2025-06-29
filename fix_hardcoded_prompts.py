#!/usr/bin/env python3
"""
Script to fix all hardcoded prompts in the codebase.
This will replace hardcoded "You are Danzar" prompts with profile-based prompts.
"""

import os
import re
import shutil
from pathlib import Path

def fix_hardcoded_prompts():
    """Fix all hardcoded prompts in the codebase."""
    
    # Files to fix
    files_to_fix = [
        "services/conversational_ai_service.py",
        "services/vision_integration_service.py", 
        "services/llm_service.py",
        "services/tool_aware_llm_service.py",
        "services/real_time_streaming_llm.py",
        "services/langchain_tools_service.py",
        "services/memory_manager.py",
        "services/central_brain_service.py",
        "services/agentic_rag_service.py"
    ]
    
    # Hardcoded prompts to replace
    hardcoded_prompts = [
        r'You are DanzarAI, an intelligent gaming assistant with advanced vision capabilities\.',
        r'You are DanzarAI, a gaming commentary assistant with advanced vision capabilities\.',
        r'You are DanzarAI, a gaming commentary assistant\.',
        r'You are Danzar, an upbeat and witty gaming assistant',
        r'You are Danzar, an AI gaming assistant with sharp wit and sarcastic humor',
        r'You are Danzar, an AI assistant with the ability to see and understand',
        r'You are DanzarVLM, an expert gaming assistant',
        r'You are "Danzar," an AI whose sarcasm is sharper than a rusty blade'
    ]
    
    # Replacement pattern
    replacement_pattern = r'self\.app_context\.active_profile\.system_prompt_commentary'
    
    print("🔧 Fixing hardcoded prompts in the codebase...")
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue
            
        print(f"📝 Processing: {file_path}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace hardcoded prompts
        for pattern in hardcoded_prompts:
            # Find lines with hardcoded prompts
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    print(f"   Found hardcoded prompt in line {i+1}: {line.strip()[:50]}...")
                    
                    # Check if this is a system_prompt assignment
                    if 'system_prompt' in line and '=' in line:
                        # Replace the entire assignment
                        new_line = re.sub(r'system_prompt\s*=\s*["\']?.*["\']?', 
                                        'system_prompt = self.app_context.active_profile.system_prompt_commentary', 
                                        line)
                        lines[i] = new_line
                        print(f"   Fixed system_prompt assignment")
                    
                    # Check if this is a prompt variable assignment
                    elif 'prompt' in line and '=' in line and 'f"""' in line:
                        # This is more complex - we need to replace the f-string
                        # For now, just add a comment
                        lines[i] = f"# TODO: Replace hardcoded prompt with profile-based prompt\n{line}"
                        print(f"   Added TODO comment for prompt variable")
        
        content = '\n'.join(lines)
        
        # Write back if changed
        if content != original_content:
            # Create backup
            backup_path = f"{file_path}.backup"
            shutil.copy2(file_path, backup_path)
            print(f"   Created backup: {backup_path}")
            
            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"   ✅ Fixed hardcoded prompts")
        else:
            print(f"   No hardcoded prompts found")
    
    print("\n🎉 Hardcoded prompt fix completed!")
    print("📋 Next steps:")
    print("   1. Review the changes in each file")
    print("   2. Test the application to ensure prompts are now profile-based")
    print("   3. Remove .backup files once you're satisfied with the changes")

if __name__ == "__main__":
    fix_hardcoded_prompts() 