# services/langchain_tools_service.py
"""
LangChain Tools Integration Service
===================================

This service provides LangChain tool integration for the DanzarAI system,
enabling agentic behavior and natural tool usage by the LLM.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

# LangChain imports
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor
from langgraph.checkpoint.memory import MemorySaver

@dataclass
class ToolResult:
    """Result from a LangChain tool execution."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class DanzarLangChainTools:
    """LangChain tools wrapper for DanzarAI capabilities."""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.vision_tools = getattr(app_context, 'vision_tools', None)
        self.memory_service = getattr(app_context, 'memory_service', None)
        self.rag_service = getattr(app_context, 'rag_service', None)
        self.llm_service = getattr(app_context, 'llm_service', None)
        
        # Initialize LangChain tools
        self.tools = self._create_langchain_tools()
        self.agent = None
        self.agent_executor = None
        
        self.logger.info("[LangChainTools] 🛠️ LangChain tools service initialized")
    
    def _create_langchain_tools(self) -> List[BaseTool]:
        """Create LangChain tool wrappers for DanzarAI capabilities."""
        tools = []
        
        # Vision Tools
        if self.vision_tools:
            tools.extend([
                self._create_vision_tool("capture_screenshot", "Capture a screenshot of the current game screen from OBS NDI stream. Use this whenever you need to see what's happening in the game or when someone asks about the current screen."),
                self._create_vision_tool("analyze_screenshot", "Analyze a screenshot to understand what's happening in the game. Use this after capturing a screenshot to get detailed information about objects, text, UI elements, and game state."),
                self._create_vision_tool("get_vision_summary", "Get a summary of recent visual activity and detections. Use this to understand what has been happening recently."),
                self._create_vision_tool("check_vision_capabilities", "Check what vision capabilities are currently available and working.")
            ])
        
        # Memory Tools
        if self.memory_service:
            tools.extend([
                self._create_memory_tool("search_memory", "Search for relevant memories and past interactions"),
                self._create_memory_tool("store_memory", "Store new information in memory for future reference")
            ])
        
        # Game Context Tools
        tools.extend([
            self._create_game_context_tool("get_game_context", "Get information about the current game context"),
            self._create_game_context_tool("set_game_context", "Set or update the current game context")
        ])
        
        # System Tools
        tools.extend([
            self._create_system_tool("get_system_status", "Get current system status and capabilities"),
            self._create_system_tool("get_conversation_history", "Get recent conversation history")
        ])
        
        self.logger.info(f"[LangChainTools] Created {len(tools)} LangChain tools")
        return tools
    
    def _create_vision_tool(self, tool_name: str, description: str) -> BaseTool:
        """Create a LangChain tool wrapper for vision capabilities."""
        
        @tool
        async def vision_tool_func(**kwargs) -> str:
            """Execute vision-related tools for screen capture and analysis."""
            try:
                if hasattr(self.vision_tools, tool_name):
                    method = getattr(self.vision_tools, tool_name)
                    if asyncio.iscoroutinefunction(method):
                        result = await method(**kwargs)
                    else:
                        result = method(**kwargs)
                    if hasattr(result, 'success') and result.success:
                        return f"✅ {tool_name} completed successfully: {result.data}"
                    else:
                        err = getattr(result, 'error', 'Unknown error')
                        return f"❌ {tool_name} failed: {err}"
                else:
                    return f"❌ Tool '{tool_name}' not found."
            except Exception as e:
                return f"❌ Exception in {tool_name}: {e}"

        # Set the tool name and description after function definition
        vision_tool_func.name = tool_name
        vision_tool_func.description = description
        return vision_tool_func
    
    def _create_memory_tool(self, tool_name: str, description: str) -> BaseTool:
        """Create a LangChain tool wrapper for memory capabilities."""
        
        @tool
        async def memory_tool_func(**kwargs) -> str:
            """Execute memory-related tools for searching and storing information."""
            try:
                if tool_name == "search_memory":
                    query = kwargs.get("query", "")
                    if self.memory_service:
                        results = await self.memory_service.search_memories(query, limit=5)
                        if results:
                            return f"🔍 Memory search results for '{query}':\n" + "\n".join([f"- {r}" for r in results])
                        else:
                            return f"🔍 No memories found for '{query}'"
                    else:
                        return "❌ Memory service not available"
                
                elif tool_name == "store_memory":
                    content = kwargs.get("content", "")
                    importance = kwargs.get("importance", "medium")
                    if self.memory_service:
                        await self.memory_service.store_memory(content, importance=importance)
                        return f"💾 Memory stored successfully: {content[:100]}..."
                    else:
                        return "❌ Memory service not available"
                
                return f"❌ Unknown memory tool: {tool_name}"
            except Exception as e:
                self.logger.error(f"[LangChainTools] Error in {tool_name}: {e}")
                return f"❌ Error executing {tool_name}: {str(e)}"
        
        # Set the tool name and description
        memory_tool_func.name = tool_name
        memory_tool_func.description = description
        return memory_tool_func
    
    def _create_game_context_tool(self, tool_name: str, description: str) -> BaseTool:
        """Create a LangChain tool wrapper for game context capabilities."""
        
        @tool
        async def game_context_tool_func(**kwargs) -> str:
            """Execute game context tools for managing game profiles and settings."""
            try:
                if tool_name == "get_game_context":
                    game_profile = getattr(self.app_context, 'active_profile', None)
                    if game_profile:
                        return f"🎮 Current game context: {game_profile.name} - {game_profile.description}"
                    else:
                        return "🎮 No game context currently set"
                
                elif tool_name == "set_game_context":
                    game_type = kwargs.get("game_type", "")
                    if game_type:
                        # Update active profile
                        from core.game_profile import GameProfile
                        try:
                            profile = GameProfile.load_profile(game_type, self.app_context)
                            if profile:
                                self.app_context.active_profile = profile
                                return f"🎮 Game context set to: {game_type}"
                            else:
                                return f"❌ Unknown game type: {game_type}"
                        except Exception as e:
                            return f"❌ Error loading game profile: {str(e)}"
                    else:
                        return "❌ No game type specified"
                
                return f"❌ Unknown game context tool: {tool_name}"
            except Exception as e:
                self.logger.error(f"[LangChainTools] Error in {tool_name}: {e}")
                return f"❌ Error executing {tool_name}: {str(e)}"
        
        # Set the tool name and description
        game_context_tool_func.name = tool_name
        game_context_tool_func.description = description
        return game_context_tool_func
    
    def _create_system_tool(self, tool_name: str, description: str) -> BaseTool:
        """Create a LangChain tool wrapper for system capabilities."""
        
        @tool
        async def system_tool_func(**kwargs) -> str:
            """Execute system-related tools for status checking and conversation history."""
            try:
                if tool_name == "get_system_status":
                    status = {
                        "vision_available": bool(self.vision_tools),
                        "memory_available": bool(self.memory_service),
                        "rag_available": bool(self.rag_service),
                        "llm_available": bool(self.llm_service),
                        "active_profile": getattr(self.app_context, 'active_profile', None),
                        "tools_count": len(self.tools)
                    }
                    return f"🔧 System Status:\n" + "\n".join([f"- {k}: {v}" for k, v in status.items()])
                
                elif tool_name == "get_conversation_history":
                    conv_service = getattr(self.app_context, 'conversational_ai_service', None)
                    if conv_service and hasattr(conv_service, 'conversation_history'):
                        recent = conv_service.conversation_history[-5:]  # Last 5 turns
                        if recent:
                            return f"💬 Recent conversation:\n" + "\n".join([f"- {turn.user_name}: {turn.message}" for turn in recent])
                        else:
                            return "💬 Recent conversation history"
                    else:
                        return "💬 Conversation service not available"
                
                return f"❌ Unknown system tool: {tool_name}"
            except Exception as e:
                self.logger.error(f"[LangChainTools] Error in {tool_name}: {e}")
                return f"❌ Error executing {tool_name}: {str(e)}"
        
        # Set the tool name and description
        system_tool_func.name = tool_name
        system_tool_func.description = description
        return system_tool_func
    
    async def initialize_agent(self, model_client) -> bool:
        """Initialize the LangChain agent with the provided model."""
        try:
            self.logger.info("[LangChainTools] 🤖 Initializing LangChain agent")
            
            # Use profile-based system prompt instead of hardcoded one
            system_prompt = self.app_context.active_profile.system_prompt_commentary

            # Create the agent
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent with memory
            memory = MemorySaver()
            self.agent = create_react_agent(
                model_client, 
                self.tools, 
                prompt=prompt,
            )
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
            
            self.logger.info("[LangChainTools] ✅ LangChain agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[LangChainTools] ❌ Failed to initialize agent: {e}")
            return False
    
    async def process_message_with_agent(self, message: str, user_id: str = "user", thread_id: str = "default") -> str:
        """Process a message using the LangChain agent."""
        try:
            if not self.agent_executor:
                return "🤖 My agentic capabilities aren't ready yet. Let me know if you need anything else!"
            
            self.logger.info(f"[LangChainTools] 🤖 Processing message with agent: {message[:50]}...")
            
            # Invoke agent with memory
            result = await self.agent_executor.ainvoke(
                {"input": message},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            response = result.get("output", "I'm not sure how to respond to that.")
            self.logger.info(f"[LangChainTools] ✅ Agent response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            self.logger.error(f"[LangChainTools] ❌ Agent processing error: {e}")
            return f"🤖 I ran into an issue processing that with my agentic brain: {str(e)}"
    
    def get_tools_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        return {
            "total_tools": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "agent_ready": bool(self.agent_executor),
            "vision_tools_available": bool(self.vision_tools),
            "memory_tools_available": bool(self.memory_service),
            "rag_tools_available": bool(self.rag_service)
        }
    
    async def test_tools(self) -> Dict[str, Any]:
        """Test all available tools and return results."""
        results = {}
        
        for tool in self.tools:
            try:
                # Test with minimal parameters
                if tool.name == "check_vision_capabilities":
                    result = await tool.ainvoke({})
                elif tool.name == "get_system_status":
                    result = await tool.ainvoke({})
                elif tool.name == "get_game_context":
                    result = await tool.ainvoke({})
                else:
                    result = f"Tool {tool.name} available (not tested)"
                
                results[tool.name] = {"status": "success", "result": result}
                
            except Exception as e:
                results[tool.name] = {"status": "error", "error": str(e)}
        
        return results 
    
    async def process_with_vision(self, user_input: str, username: str = "User") -> str:
        """Process user input with vision-aware capabilities using LangChain tools."""
        try:
            self.logger.info(f"[LangChainTools] Processing vision-aware input: '{user_input}'")
            
            # Check if we have an agent
            if not self.agent:
                self.logger.warning("[LangChainTools] No agent available, falling back to direct LLM")
                return await self._fallback_direct_llm(user_input, username)
            
            # Create a simple prompt for the agent
            prompt = f"User ({username}): {user_input}\n\nPlease respond naturally and helpfully. If the user asks about seeing their screen or what's visible, use the available vision tools to capture and analyze the screen content."
            
            # Run the agent
            try:
                result = await self.agent.ainvoke({"input": prompt})
                response = result.get("output", "")
                
                if response:
                    self.logger.info(f"[LangChainTools] Generated vision-aware response: {len(response)} chars")
                    return response
                else:
                    self.logger.warning("[LangChainTools] Agent returned empty response")
                    return await self._fallback_direct_llm(user_input, username)
                    
            except Exception as e:
                self.logger.error(f"[LangChainTools] Agent execution error: {e}")
                return await self._fallback_direct_llm(user_input, username)
                
        except Exception as e:
            self.logger.error(f"[LangChainTools] Vision processing error: {e}")
            return "I'm having trouble processing your request right now."
    
    async def _fallback_direct_llm(self, user_input: str, username: str) -> str:
        """Fallback to direct LLM processing when agent is not available."""
        try:
            # Get model client from app context
            model_client = getattr(self.app_context, 'model_client', None)
            if not model_client:
                return "I'm not connected to my language model right now."
            
            # Create a simple prompt
            prompt = f"User ({username}): {user_input}\n\nPlease respond naturally and helpfully as DANZAR, a gaming assistant."
            
            # Get response from model client
            response = await model_client.generate_response(prompt)
            return response if response else "I'm having trouble generating a response right now."
            
        except Exception as e:
            self.logger.error(f"[LangChainTools] Fallback LLM error: {e}")
            return "I'm having trouble connecting to my language model right now." 