#!/usr/bin/env python3
"""
Find all hardcoded prompts in the codebase.
"""

import os
import re

def find_hardcoded_prompts():
    """Find all hardcoded prompts in the codebase."""
    
    # Files to search
    files_to_search = [
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
    
    # Hardcoded prompt patterns
    patterns = [
        r'You are DanzarAI, an intelligent gaming assistant',
        r'You are DanzarAI, a gaming commentary assistant',
        r'You are Danzar, an upbeat and witty gaming assistant',
        r'You are Danzar, an AI gaming assistant',
        r'You are Danzar, an AI assistant',
        r'You are DanzarVLM, an expert gaming assistant',
        r'You are "Danzar," an AI whose sarcasm'
    ]
    
    print("🔍 Searching for hardcoded prompts...")
    
    for file_path in files_to_search:
        if not os.path.exists(file_path):
            continue
            
        print(f"\n📁 {file_path}:")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        found_any = False
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                if re.search(pattern, line):
                    print(f"   Line {i+1}: {line.strip()[:80]}...")
                    found_any = True
        
        if not found_any:
            print("   ✅ No hardcoded prompts found")

if __name__ == "__main__":
    find_hardcoded_prompts() 