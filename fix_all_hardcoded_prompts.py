#!/usr/bin/env python3
"""
Automated script to fix all hardcoded prompts in the DanzarVLM codebase.
This script will replace all hardcoded "You are Danzar" prompts with profile-based prompts.
"""

import os
import re
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict

class HardcodedPromptFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backups" / f"prompt_fix_{int(time.time())}"
        self.files_processed = 0
        self.prompts_fixed = 0
        self.errors = []
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Hardcoded prompt patterns to find and replace
        self.hardcoded_patterns = [
            # System prompt assignments
            (r'system_prompt\s*=\s*["\']?You are Danzar[^"\']*["\']?', 
             'system_prompt = self.app_context.active_profile.system_prompt_commentary'),
            
            # Prompt variable assignments with f-strings
            (r'prompt\s*=\s*f["\']?You are Danzar[^"\']*["\']?', 
             'prompt = f"""{self.app_context.active_profile.system_prompt_commentary}'),
            
            # Direct string assignments
            (r'["\']?You are DanzarAI, an intelligent gaming assistant[^"\']*["\']?', 
             'self.app_context.active_profile.system_prompt_commentary'),
            
            (r'["\']?You are DanzarAI, a gaming commentary assistant[^"\']*["\']?', 
             'self.app_context.active_profile.system_prompt_commentary'),
            
            (r'["\']?You are Danzar, an upbeat and witty gaming assistant[^"\']*["\']?', 
             'self.app_context.active_profile.system_prompt_commentary'),
            
            (r'["\']?You are Danzar, an AI gaming assistant[^"\']*["\']?', 
             'self.app_context.active_profile.system_prompt_commentary'),
            
            (r'["\']?You are Danzar, an AI assistant[^"\']*["\']?', 
             'self.app_context.active_profile.system_prompt_commentary'),
            
            (r'["\']?You are DanzarVLM, an expert gaming assistant[^"\']*["\']?', 
             'self.app_context.active_profile.system_prompt_commentary'),
            
            (r'["\']?You are "Danzar," an AI whose sarcasm[^"\']*["\']?', 
             'self.app_context.active_profile.system_prompt_commentary'),
        ]
        
        # Files to process (Python files in services directory)
        self.target_files = [
            "services/conversational_ai_service.py",
            "services/vision_integration_service.py",
            "services/tool_aware_llm_service.py",
            "services/real_time_streaming_llm.py",
            "services/langchain_tools_service.py",
            "services/central_brain_service.py",
            "services/agentic_rag_service.py",
            "services/memory_manager.py",
            "services/vision_aware_conversation_service.py",
            "services/llm_service.py",
            "services/multi_llm_coordinator.py",
            "services/agentic_memory.py",
            "services/rag_service.py",
            "services/fact_check_service.py",
            "services/vision_tools.py",
            "services/vision_conversation_coordinator.py",
        ]
        
        # Files to exclude
        self.exclude_patterns = [
            r'\.pyc$',
            r'__pycache__',
            r'\.git',
            r'backups',
            r'logs',
            r'node_modules',
            r'venv',
            r'\.venv',
        ]

    def should_exclude_file(self, file_path: str) -> bool:
        """Check if file should be excluded from processing."""
        for pattern in self.exclude_patterns:
            if re.search(pattern, file_path):
                return True
        return False

    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file."""
        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        return backup_path

    def find_hardcoded_prompts(self, content: str) -> List[Tuple[str, int, str]]:
        """Find all hardcoded prompts in the content."""
        found_prompts = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, replacement in self.hardcoded_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    found_prompts.append((line.strip(), line_num, pattern))
                    break
        
        return found_prompts

    def fix_hardcoded_prompts(self, content: str) -> Tuple[str, int]:
        """Fix hardcoded prompts in the content."""
        original_content = content
        fixed_count = 0
        
        for pattern, replacement in self.hardcoded_patterns:
            # Handle different types of replacements
            if 'system_prompt =' in pattern:
                # System prompt assignment
                new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                if new_content != content:
                    fixed_count += 1
                    content = new_content
                    
            elif 'prompt = f' in pattern:
                # Prompt variable with f-string
                # This is more complex - we need to handle the f-string properly
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if re.search(pattern, line, re.IGNORECASE):
                        # Replace the line with profile-based prompt
                        lines[i] = f'        # Use profile-based system prompt instead of hardcoded one\n        prompt = f"""{replacement}\n\n'
                        fixed_count += 1
                content = '\n'.join(lines)
                
            else:
                # Direct string replacement
                new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                if new_content != content:
                    fixed_count += 1
                    content = new_content
        
        return content, fixed_count

    def process_file(self, file_path: Path) -> Dict[str, any]:
        """Process a single file to fix hardcoded prompts."""
        result = {
            'file': str(file_path),
            'processed': False,
            'prompts_found': 0,
            'prompts_fixed': 0,
            'backup_created': False,
            'error': None
        }
        
        try:
            if not file_path.exists():
                result['error'] = f"File not found: {file_path}"
                return result
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find hardcoded prompts
            found_prompts = self.find_hardcoded_prompts(content)
            result['prompts_found'] = len(found_prompts)
            
            if found_prompts:
                print(f"🔍 Found {len(found_prompts)} hardcoded prompts in {file_path.name}")
                for prompt, line_num, pattern in found_prompts:
                    print(f"   Line {line_num}: {prompt[:80]}...")
                
                # Create backup
                backup_path = self.backup_file(file_path)
                result['backup_created'] = True
                print(f"   💾 Backup created: {backup_path.name}")
                
                # Fix prompts
                fixed_content, fixed_count = self.fix_hardcoded_prompts(content)
                result['prompts_fixed'] = fixed_count
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                result['processed'] = True
                print(f"   ✅ Fixed {fixed_count} prompts")
                
            else:
                result['processed'] = True
                print(f"✅ No hardcoded prompts found in {file_path.name}")
                
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ Error processing {file_path.name}: {e}")
        
        return result

    def run(self) -> Dict[str, any]:
        """Run the hardcoded prompt fixer."""
        print("🚀 Starting Hardcoded Prompt Fixer")
        print("=" * 50)
        
        results = {
            'files_processed': 0,
            'files_with_prompts': 0,
            'total_prompts_found': 0,
            'total_prompts_fixed': 0,
            'backups_created': 0,
            'errors': [],
            'file_results': []
        }
        
        # Process target files
        for file_path_str in self.target_files:
            file_path = self.project_root / file_path_str
            
            if self.should_exclude_file(str(file_path)):
                continue
            
            result = self.process_file(file_path)
            results['file_results'].append(result)
            
            if result['processed']:
                results['files_processed'] += 1
                
                if result['prompts_found'] > 0:
                    results['files_with_prompts'] += 1
                    results['total_prompts_found'] += result['prompts_found']
                    results['total_prompts_fixed'] += result['prompts_fixed']
                    
                    if result['backup_created']:
                        results['backups_created'] += 1
            
            if result['error']:
                results['errors'].append(result['error'])
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 FIX SUMMARY")
        print("=" * 50)
        print(f"Files processed: {results['files_processed']}")
        print(f"Files with hardcoded prompts: {results['files_with_prompts']}")
        print(f"Total prompts found: {results['total_prompts_found']}")
        print(f"Total prompts fixed: {results['total_prompts_fixed']}")
        print(f"Backups created: {results['backups_created']}")
        
        if results['errors']:
            print(f"\n❌ Errors encountered: {len(results['errors'])}")
            for error in results['errors']:
                print(f"   - {error}")
        
        print(f"\n💾 Backups saved to: {self.backup_dir}")
        print("\n🎉 Hardcoded prompt fix completed!")
        
        return results

def main():
    """Main function to run the hardcoded prompt fixer."""
    fixer = HardcodedPromptFixer()
    results = fixer.run()
    
    if results['total_prompts_fixed'] > 0:
        print("\n✅ SUCCESS: All hardcoded prompts have been replaced with profile-based prompts!")
        print("🔄 Please restart your Danzar application to see the changes.")
    else:
        print("\nℹ️  No hardcoded prompts were found or fixed.")
    
    return results

if __name__ == "__main__":
    main() 