#!/usr/bin/env python3
"""
Fix LangChain Integration
=========================

This script manually adds LangChain tools initialization to DanzarVLM.py
since the edit tool is having trouble with the large file.
"""

import re

def add_langchain_initialization():
    """Add LangChain tools initialization to DanzarVLM.py."""
    
    # Read the current file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the location after vision integration service
    pattern = r'(\s+except Exception as e:\s*\n\s+self\.logger\.error\(f"❌ Vision Integration Service error: \{e\}"\)\s*\n\s+self\.app_context\.vision_integration_service = None\s*\n\s+)(\s+except Exception as e:\s*\n\s+self\.logger\.error\(f"❌ Service initialization failed: \{e\}"\)\s*\n\s+)'
    
    # LangChain initialization code to insert
    langchain_code = '''            # Initialize LangChain Tools Service for agentic behavior
            try:
                from services.langchain_tools_service import DanzarLangChainTools
                
                self.logger.info("🔧 Initializing LangChain Tools Service...")
                self.app_context.langchain_tools = DanzarLangChainTools(self.app_context)
                
                # Initialize the agent with the model client
                if hasattr(self.app_context, 'model_client') and self.app_context.model_client:
                    success = await self.app_context.langchain_tools.initialize_agent(self.app_context.model_client)
                    if success:
                        self.logger.info("✅ LangChain Tools Service initialized successfully")
                        self.logger.info("🤖 Agentic behavior enabled with tool awareness")
                        self.logger.info("🎯 LLM can now use vision, memory, and system tools")
                    else:
                        self.logger.error("❌ LangChain agent initialization failed")
                        self.app_context.langchain_tools = None
                else:
                    self.logger.error("❌ Model client not available for LangChain tools")
                    self.app_context.langchain_tools = None
            except Exception as e:
                self.logger.error(f"❌ LangChain Tools Service error: {e}")
                self.app_context.langchain_tools = None
            
'''
    
    # Replace the pattern
    new_content = re.sub(pattern, r'\1' + langchain_code + r'\2', content)
    
    # Write the updated content
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully added LangChain tools initialization to DanzarVLM.py")
    print("🔧 The LLM should now know about its tools!")

if __name__ == "__main__":
    add_langchain_initialization() 