#!/usr/bin/env python3
"""
Complete fix for DanzarVLM.py indentation errors
"""

def fix_danzar_complete():
    """Fix the entire problematic region in DanzarVLM.py"""
    
    # Read the file
    with open('DanzarVLM.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the problematic region to replace
    problematic_start = """                    self.app_context.vision_integration_service = None
            except Exception as e:
                self.logger.error(f"❌ Vision Integration Service error: {e}")
            # Initialize LangChain Tools Service for agentic behavior
            
                   # Initialize LangChain Tools Service for agentic behavior
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
            except Exception as e:
        except Exception as e:
                self.logger.error(f"❌ LangChain Tools Service error: {e}")
        except Exception as e:
            
 except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {e}")"""

    # Define the correct replacement
    correct_replacement = """                    self.app_context.vision_integration_service = None
            except Exception as e:
                self.logger.error(f"❌ Vision Integration Service error: {e}")
                self.app_context.vision_integration_service = None

            # Initialize LangChain Tools Service for agentic behavior
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
        except Exception as e:
            self.logger.error(f"❌ Service initialization failed: {e}")"""

    # Replace the problematic region
    if problematic_start in content:
        content = content.replace(problematic_start, correct_replacement)
        print("✅ Found and replaced problematic region")
    else:
        print("⚠️ Problematic region not found, trying alternative approach...")
        
        # Alternative: Find and replace just the corrupted except blocks
        content = content.replace("            except Exception as e:\n        except Exception as e:\n                self.logger.error(f\"❌ LangChain Tools Service error: {e}\")\n        except Exception as e:\n            \n except Exception as e:\n            self.logger.error(f\"❌ Service initialization failed: {e}\")", 
                                 "            except Exception as e:\n                self.logger.error(f\"❌ LangChain Tools Service error: {e}\")\n                self.app_context.langchain_tools = None\n        except Exception as e:\n            self.logger.error(f\"❌ Service initialization failed: {e}\")")
        print("✅ Applied alternative fix")

    # Write the fixed content back
    with open('DanzarVLM.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed DanzarVLM.py indentation errors")

if __name__ == "__main__":
    fix_danzar_complete() 