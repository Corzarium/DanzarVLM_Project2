#!/usr/bin/env python3
"""
Tool-Aware LLM Service - Lets the LLM decide when to use RAG and search tools
"""

import json
import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from services.model_client import ModelClient
from services.memory_service import MemoryService, MemoryEntry


@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    success: bool
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ToolAwareLLMService:
    """LLM Service that provides tools to the model and lets it decide when to use them"""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.model_client: ModelClient = app_context.model_client
        self.memory_service: MemoryService = app_context.memory_service
        self.rag_service = app_context.rag_service_instance
        
        # Validate that model_client is properly initialized
        if self.model_client is None:
            self.logger.error("[ToolAwareLLM] ModelClient is None - this will cause errors")
            raise ValueError("ModelClient not found in app_context")
        
        # Available tools
        self.available_tools = {
            "search_knowledge": {
                "description": "Search the EverQuest knowledge base for specific game information",
                "parameters": {
                    "query": "The search query for game-specific information",
                    "collection": "Optional collection name (defaults to 'Everquest')"
                }
            },
            "web_search": {
                "description": "Search the web for current information not in the knowledge base",
                "parameters": {
                    "query": "The web search query"
                }
            },
            "get_conversation_context": {
                "description": "Retrieve recent conversation history with the user",
                "parameters": {
                    "user_name": "The user's name",
                    "limit": "Number of recent messages to retrieve (default: 5)"
                }
            }
        }
        
        self.logger.info("✅ Tool-Aware LLM Service initialized")
    
    async def handle_user_query(self, user_text: str, user_name: str = "User") -> str:
        """
        Handle user query with tool-aware LLM approach
        The LLM decides if and when to use tools
        """
        try:
            self.logger.info(f"[ToolAwareLLM] Processing query from {user_name}: '{user_text}'")
            
            # First, let the LLM analyze the query and decide what tools (if any) to use
            tool_plan = await self._get_tool_plan(user_text, user_name)
            
            if not tool_plan.get('needs_tools', False):
                # Simple conversational response - no tools needed
                self.logger.info("[ToolAwareLLM] No tools needed - generating direct response")
                return await self._generate_direct_response(user_text, user_name)
            
            # Execute the planned tools
            tool_results = []
            knowledge_search_failed = False
            
            for tool_call in tool_plan.get('tool_calls', []):
                result = await self._execute_tool(tool_call)
                tool_results.append(result)
                self.logger.info(f"[ToolAwareLLM] Executed tool '{result.tool_name}': {'✅' if result.success else '❌'}")
                
                # Track if knowledge search failed or returned no useful results
                if (result.tool_name == "search_knowledge" and 
                    (not result.success or 
                     "No results found" in result.content or 
                     "No readable content found" in result.content)):
                    knowledge_search_failed = True
                    self.logger.info("[ToolAwareLLM] Knowledge search failed or returned no results")
            
            # If knowledge search failed and we haven't tried web search yet, try it as fallback
            if (knowledge_search_failed and 
                not any(r.tool_name == "web_search" for r in tool_results)):
                
                self.logger.info("[ToolAwareLLM] Triggering web search fallback due to failed knowledge search")
                web_search_result = await self._web_search_tool({"query": user_text})
                tool_results.append(web_search_result)
                self.logger.info(f"[ToolAwareLLM] Web search fallback: {'✅' if web_search_result.success else '❌'}")
            
            # Generate final response using tool results
            return await self._generate_response_with_tools(user_text, user_name, tool_results)
            
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Error processing query: {e}")
            return "I encountered an error processing your request. Please try again."
    
    async def _get_tool_plan(self, user_text: str, user_name: str) -> Dict[str, Any]:
        """Determine if tools are needed and which ones to use"""
        
        # Quick check for simple greetings - no tools needed
        simple_greetings = [
            r'\b(hi|hello|hey|greetings?)\b',
            r'\b(good\s+(morning|afternoon|evening|night))\b',
            r'\b(how\s+are\s+you)\b',
            r'\b(what\'?s\s+up)\b'
        ]
        
        user_lower = user_text.lower().strip()
        if any(re.search(pattern, user_lower) for pattern in simple_greetings) and len(user_text.split()) <= 5:
            return {
                "needs_tools": False,
                "reasoning": "Simple greeting detected - no tools needed",
                "tool_calls": []
            }
        
        # Use LLM to determine tool usage
        system_prompt = self.app_context.active_profile.system_prompt_commentary
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {user_text}"}
        ]
        
        try:
            self.logger.info(f"[ToolAwareLLM] Getting tool plan for: '{user_text}'")
            
            # Add timeout protection for LLM calls
            import asyncio
            try:
                # Use asyncio.wait_for to add timeout protection
                llm_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model_client.generate(
                        messages=messages,
                        temperature=0.1,  # Lower temperature for more consistent JSON
                        max_tokens=8192,  # Increased for DeepSeek-R1 reasoning model tool planning
                        stream=False
                    )
                )
                response = await asyncio.wait_for(llm_task, timeout=45.0)  # 45 second timeout
                
            except asyncio.TimeoutError:
                self.logger.error(f"[ToolAwareLLM] Tool planning timeout after 45 seconds for query: '{user_text}'")
                return self._fallback_tool_plan(user_text)
            except Exception as llm_error:
                self.logger.error(f"[ToolAwareLLM] Tool planning LLM error: {llm_error}")
                return self._fallback_tool_plan(user_text)
            
            if not response or not response.strip():
                self.logger.warning(f"[ToolAwareLLM] Empty response from tool planning LLM")
                return self._fallback_tool_plan(user_text)
            
            # Clean up the response before parsing
            response = response.strip()
            
            # Try to extract JSON from the response
            import json
            try:
                # Look for JSON object in the response
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    tool_plan = json.loads(json_str)
                    
                    # Validate the structure
                    if "needs_tools" in tool_plan and "tool_calls" in tool_plan:
                        self.logger.info(f"[ToolAwareLLM] Tool plan: needs_tools={tool_plan['needs_tools']}, tools={len(tool_plan.get('tool_calls', []))}")
                        return tool_plan
                    else:
                        self.logger.warning(f"[ToolAwareLLM] Invalid tool plan structure: {tool_plan}")
                        return self._fallback_tool_plan(user_text)
                else:
                    self.logger.warning(f"[ToolAwareLLM] No JSON found in response: {response[:200]}...")
                    return self._fallback_tool_plan(user_text)
                    
            except json.JSONDecodeError as e:
                self.logger.warning(f"[ToolAwareLLM] Failed to parse tool plan JSON: {e}")
                self.logger.debug(f"[ToolAwareLLM] Raw response: {response[:500]}...")
                return self._fallback_tool_plan(user_text)
                
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Error getting tool plan: {e}", exc_info=True)
            return self._fallback_tool_plan(user_text)
    
    async def _execute_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a specific tool call"""
        tool_name = tool_call.get('tool', 'unknown')
        parameters = tool_call.get('parameters', {})
        
        try:
            if tool_name == "search_knowledge":
                return await self._search_knowledge_tool(parameters)
            elif tool_name == "web_search":
                return await self._web_search_tool(parameters)
            elif tool_name == "get_conversation_context":
                return await self._get_conversation_context_tool(parameters)
            else:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    content=f"Unknown tool: {tool_name}"
                )
                
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Error executing tool {tool_name}: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content=f"Tool execution failed: {str(e)}"
            )
    
    async def _search_knowledge_tool(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute knowledge base search with robust error handling"""
        query = parameters.get('query', '')
        collection = parameters.get('collection', 'Everquest')
        
        if not query:
            return ToolResult("search_knowledge", False, "No search query provided")
        
        if not self.rag_service:
            return ToolResult("search_knowledge", False, "RAG service not available")
        
        try:
            self.logger.info(f"[ToolAwareLLM] Searching knowledge base: '{query}' in collection '{collection}'")
            
            # Add timeout protection for the search operation
            import asyncio
            try:
                # Use asyncio.wait_for to add timeout protection
                search_task = asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.rag_service.query_knowledge(query, collection_name=collection, limit=3)
                )
                results = await asyncio.wait_for(search_task, timeout=30.0)  # 30 second timeout
                
            except asyncio.TimeoutError:
                self.logger.error(f"[ToolAwareLLM] Search timeout after 30 seconds for query: '{query}'")
                return ToolResult(
                    tool_name="search_knowledge",
                    success=False,
                    content=f"Search timed out for '{query}' in {collection}. Please try a simpler query."
                )
            except Exception as search_error:
                self.logger.error(f"[ToolAwareLLM] Search execution error: {search_error}")
                return ToolResult(
                    tool_name="search_knowledge",
                    success=False,
                    content=f"Search failed due to technical error: {str(search_error)}"
                )
            
            if results:
                # Extract text content from the results
                content_parts = []
                for i, result in enumerate(results):
                    try:
                        # Handle different possible formats
                        if isinstance(result, dict):
                            text = result.get('text', '') or result.get('content', '') or str(result)
                        else:
                            text = str(result)
                        
                        if text.strip():
                            content_parts.append(f"Result {i+1}: {text.strip()}")
                    except Exception as e:
                        self.logger.warning(f"[ToolAwareLLM] Error processing search result {i}: {e}")
                        continue
                
                if content_parts:
                    content = "\n\n".join(content_parts)
                    self.logger.info(f"[ToolAwareLLM] Search successful: {len(content_parts)} results found")
                    return ToolResult(
                        tool_name="search_knowledge",
                        success=True,
                        content=content,
                        metadata={"query": query, "collection": collection, "result_count": len(results)}
                    )
                else:
                    return ToolResult(
                        tool_name="search_knowledge",
                        success=False,
                        content=f"No readable content found for '{query}' in {collection}"
                    )
            else:
                return ToolResult(
                    tool_name="search_knowledge",
                    success=False,
                    content=f"No results found for '{query}' in {collection}"
                )
                
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Search knowledge tool error: {e}", exc_info=True)
            return ToolResult(
                tool_name="search_knowledge",
                success=False,
                content=f"Search failed: {str(e)}"
            )
    
    async def _web_search_tool(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute web search using the web_search tool"""
        query = parameters.get('query', '')
        
        if not query:
            return ToolResult(
                tool_name="web_search",
                success=False,
                content="No search query provided"
            )
        
        try:
            self.logger.info(f"[ToolAwareLLM] Executing web search for: '{query}'")
            
            # Import and use the web_search function
            try:
                from utils.web_search import web_search
                
                # Execute web search with timeout
                import asyncio
                search_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: web_search(query)
                )
                search_results = await asyncio.wait_for(search_task, timeout=30.0)
                
                if search_results and len(search_results.strip()) > 0:
                    self.logger.info(f"[ToolAwareLLM] Web search successful, got {len(search_results)} characters")
                    return ToolResult(
                        tool_name="web_search",
                        success=True,
                        content=search_results,
                        metadata={"query": query, "result_length": len(search_results)}
                    )
                else:
                    self.logger.warning(f"[ToolAwareLLM] Web search returned empty results for: '{query}'")
                    return ToolResult(
                        tool_name="web_search",
                        success=False,
                        content="Web search returned no results"
                    )
                    
            except ImportError:
                self.logger.error("[ToolAwareLLM] Web search module not available")
                return ToolResult(
                    tool_name="web_search",
                    success=False,
                    content="Web search functionality not available"
                )
            except asyncio.TimeoutError:
                self.logger.error(f"[ToolAwareLLM] Web search timeout for query: '{query}'")
                return ToolResult(
                    tool_name="web_search",
                    success=False,
                    content="Web search timed out"
                )
                
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Web search error: {e}")
            return ToolResult(
                tool_name="web_search",
                success=False,
                content=f"Web search failed: {str(e)}"
            )
    
    async def _get_conversation_context_tool(self, parameters: Dict[str, Any]) -> ToolResult:
        """Get recent conversation history"""
        user_name = parameters.get('user_name', 'User')
        limit = parameters.get('limit', 5)
        
        try:
            if not self.memory_service:
                return ToolResult("get_conversation_context", False, "Memory service not available")
            
            # Get relevant memories for this user using the available method
            try:
                # Use get_relevant_memories with a generic query to get recent memories
                query = f"conversation with {user_name}"
                memories = self.memory_service.get_relevant_memories(query, top_k=limit*2)
            except Exception as e:
                # Fallback if method fails
                self.logger.warning(f"[ToolAwareLLM] Memory service query failed: {e}")
                return ToolResult("get_conversation_context", False, f"Memory service query failed: {str(e)}")
            
            # Filter for this user's conversation
            user_memories = []
            for memory in memories:
                if memory.metadata and memory.metadata.get('user') == user_name:
                    user_memories.append(memory)
                if len(user_memories) >= limit:
                    break
            
            if user_memories:
                context = "\n".join([f"{memory.content}" for memory in reversed(user_memories)])
                return ToolResult(
                    tool_name="get_conversation_context",
                    success=True,
                    content=context,
                    metadata={"user_name": user_name, "message_count": len(user_memories)}
                )
            else:
                return ToolResult(
                    tool_name="get_conversation_context",
                    success=False,
                    content=f"No recent conversation history found for {user_name}"
                )
                
        except Exception as e:
            return ToolResult(
                tool_name="get_conversation_context",
                success=False,
                content=f"Failed to retrieve conversation context: {str(e)}"
            )
    
    def _fallback_tool_plan(self, user_text: str) -> Dict[str, Any]:
        """Fallback tool planning when LLM fails"""
        user_lower = user_text.lower()
        
        # Check for game-related keywords
        game_keywords = ['everquest', 'eq', 'class', 'quest', 'spell', 'item', 'zone', 'raid', 'guild']
        if any(keyword in user_lower for keyword in game_keywords):
            return {
                "needs_tools": True,
                "reasoning": "Detected game-related query - using knowledge search with web search fallback",
                "tool_calls": [
                    {
                        "tool": "search_knowledge",
                        "parameters": {
                            "query": user_text,
                            "collection": "Everquest"
                        }
                    }
                ]
            }
        
        # Check for current events, weather, news, or other web-searchable content
        web_keywords = ['weather', 'news', 'current', 'latest', 'today', 'recent', 'new', '2024', '2025', 'price', 'cost']
        if any(keyword in user_lower for keyword in web_keywords):
            return {
                "needs_tools": True,
                "reasoning": "Detected query requiring current information - using web search",
                "tool_calls": [
                    {
                        "tool": "web_search",
                        "parameters": {
                            "query": user_text
                        }
                    }
                ]
            }
        
        # For other questions that might need information, try knowledge search first
        question_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'tell me', 'explain', 'describe']
        if any(indicator in user_lower for indicator in question_indicators):
            return {
                "needs_tools": True,
                "reasoning": "Detected information request - trying knowledge search with web fallback",
                "tool_calls": [
                    {
                        "tool": "search_knowledge",
                        "parameters": {
                            "query": user_text,
                            "collection": "Everquest"
                        }
                    }
                ]
            }
        
        # Default to no tools for simple queries
        return {
            "needs_tools": False,
            "reasoning": "Fallback - treating as simple query",
            "tool_calls": []
        }

    def _clean_reasoning_response(self, response: str) -> str:
        """Clean up response that may contain reasoning content"""
        if not response:
            return response
            
        # Enhanced patterns for reasoning content that should be removed
        reasoning_patterns = [
            "Okay, let's tackle this",
            "First, I need to figure out",
            "Let me think about this",
            "I need to analyze",
            "Looking at this question",
            "Based on the information provided",
            "From the search results",
            "The user is asking about",
            "The user asked about",
            "Looking at the search results",
            "Hmm,",
            "Let me see",
            "I should",
            "Now I need to",
            "Let me check",
            "Let me search",
            "I'll search",
            "I need to search",
            "Looking at the user's question",
            "The user wants to know",
            "This is asking about",
            "This question is about",
            "I can see that",
            "It looks like",
            "It seems like",
            "From what I can tell",
            "Based on the query",
            "Given the question",
            "To answer this",
            "For this question",
            "Regarding this question",
            "About this topic",
            "On this subject",
            "Let me provide",
            "I'll provide",
            "I can provide",
            "I should provide",
            "I need to provide",
            "I'll help with",
            "I can help with",
            "Let me help with",
            "I should help with",
            "I need to help with",
            "Looking at what you're asking",
            "From your question",
            "Based on your question",
            "Considering your question",
            "To address your question",
            "In response to your question",
            "For your question about",
            "Regarding your question about",
            "About your question on",
            "Concerning your question",
            "With respect to your question",
            "As for your question",
            "In answer to your question",
            "To respond to your question"
        ]
        
        lines = response.split('\n')
        cleaned_lines = []
        skip_reasoning = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Skip lines that start with common reasoning patterns
            if any(line_lower.startswith(pattern.lower()) for pattern in reasoning_patterns):
                skip_reasoning = True
                continue
                
            # Skip lines that contain reasoning indicators
            if any(pattern.lower() in line_lower for pattern in reasoning_patterns):
                continue
                
            # Skip incomplete sentences that end with commas (like "Hmm,")
            if line.strip().endswith(',') and len(line.strip()) < 20:
                continue
                
            # If we find a proper sentence, stop skipping
            if line.strip() and not skip_reasoning:
                cleaned_lines.append(line)
            elif line.strip() and skip_reasoning:
                # Check if this looks like the start of actual response content
                if (line.strip().endswith('.') or line.strip().endswith('!') or line.strip().endswith('?')) and len(line.strip()) > 10:
                    skip_reasoning = False
                    cleaned_lines.append(line)
        
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        # If we cleaned everything away, try to extract the last meaningful sentence
        if not cleaned_response and response:
            sentences = response.split('.')
            for sentence in reversed(sentences):
                sentence = sentence.strip()
                if (sentence and len(sentence) > 10 and 
                    not any(pattern.lower() in sentence.lower() for pattern in reasoning_patterns)):
                    cleaned_response = sentence + '.'
                    break
        
        # Final fallback - if still empty, provide a generic response
        if not cleaned_response:
            cleaned_response = "I found some information, but I need to search more specifically for that topic."
            
        return cleaned_response

    def _strip_comprehensive_reasoning(self, text: str) -> str:
        """Comprehensive removal of reasoning, thinking, and analysis content"""
        import re
        
        if not text:
            return text
        
        # Remove <think>...</think> tags and content (case insensitive, multiline)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove reasoning sections that start with common patterns
        reasoning_starters = [
            r'let me think.*?\.',
            r'hmm,.*?\.',
            r'okay,.*?\.',
            r'first,.*?\.',
            r'looking at.*?\.',
            r'based on.*?\.',
            r'from.*?results.*?\.',
            r'the user.*?asking.*?\.',
            r'this.*?question.*?\.',
            r'i need to.*?\.',
            r'i should.*?\.',
            r'let me.*?\.',
            r'i\'ll.*?\.',
            r'i can.*?\.',
            r'to answer.*?\.',
            r'for this.*?\.',
            r'regarding.*?\.',
            r'about.*?topic.*?\.',
            r'concerning.*?\.',
            r'with respect to.*?\.',
            r'as for.*?\.',
            r'in.*?to.*?question.*?\.'
        ]
        
        for pattern in reasoning_starters:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove standalone reasoning words/phrases at the start of lines
        reasoning_lines = [
            r'^hmm[,.]?\s*',
            r'^okay[,.]?\s*',
            r'^well[,.]?\s*',
            r'^so[,.]?\s*',
            r'^now[,.]?\s*',
            r'^first[,.]?\s*',
            r'^let me see[,.]?\s*',
            r'^let me think[,.]?\s*',
            r'^i need to[,.]?\s*',
            r'^i should[,.]?\s*'
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            original_line = line
            for pattern in reasoning_lines:
                line = re.sub(pattern, '', line, flags=re.IGNORECASE)
            
            # Only keep lines that have substantial content after cleaning
            if line.strip() and len(line.strip()) > 3:
                cleaned_lines.append(line)
            elif original_line.strip() and not any(re.match(pattern, original_line.strip(), re.IGNORECASE) for pattern in reasoning_lines):
                # Keep original line if it wasn't a reasoning line
                cleaned_lines.append(original_line)
        
        # Join lines and clean up extra whitespace
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n\s*\n', '\n\n', result)  # Remove extra blank lines
        result = result.strip()
        
        # If we cleaned everything away, try to extract the last meaningful sentence
        if not result and text:
            sentences = re.split(r'[.!?]+', text)
            for sentence in reversed(sentences):
                sentence = sentence.strip()
                if (sentence and len(sentence) > 15 and 
                    not any(re.search(pattern, sentence, re.IGNORECASE) for pattern in reasoning_starters)):
                    result = sentence + '.'
                    break
        
        return result

    async def _generate_direct_response(self, user_text: str, user_name: str) -> str:
        """Generate a direct response without using any tools"""
        system_prompt = self.app_context.active_profile.system_prompt_commentary
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        try:
            self.logger.info(f"[ToolAwareLLM] Generating direct response for: '{user_text}'")
            
            # Add timeout protection for LLM calls
            import asyncio
            try:
                # Use asyncio.wait_for to add timeout protection
                llm_task = asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model_client.generate(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=16384,  # Maximum for DeepSeek-R1 reasoning model completion
                        stream=False
                    )
                )
                response = await asyncio.wait_for(llm_task, timeout=300.0)  # 5 minute timeout for DeepSeek-R1 reasoning
                
            except asyncio.TimeoutError:
                self.logger.error(f"[ToolAwareLLM] Direct response timeout after 300 seconds for query: '{user_text}'")
                # Raise exception to trigger fallback in LLM service
                raise Exception("Tool-Aware LLM connection failed: Direct response timeout")
            except Exception as llm_error:
                self.logger.error(f"[ToolAwareLLM] Direct response LLM error: {llm_error}")
                # Raise exception to trigger fallback in LLM service
                raise Exception(f"Tool-Aware LLM connection failed: {str(llm_error)}")
            
            if not response or not response.strip():
                self.logger.error("[ToolAwareLLM] Model returned empty response")
                # Raise exception to trigger fallback in LLM service
                raise Exception("Tool-Aware LLM connection failed: Model returned empty response")
            
            # Clean up any reasoning content before returning
            cleaned_response = self._strip_comprehensive_reasoning(response.strip())
            
            if not cleaned_response or not cleaned_response.strip():
                self.logger.error("[ToolAwareLLM] Response became empty after cleaning reasoning content")
                # Raise exception to trigger fallback in LLM service
                raise Exception("Tool-Aware LLM connection failed: Response empty after cleaning")
            
            self.logger.info(f"[ToolAwareLLM] Direct response generated successfully")
            return cleaned_response.strip()
            
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Error generating direct response: {e}")
            # Re-raise to trigger fallback in LLM service
            raise
    
    async def _generate_response_with_tools(self, user_text: str, user_name: str, tool_results: List[ToolResult]) -> str:
        """Generate final response using tool results"""
        
        # Prepare tool results for the prompt
        tool_context = ""
        for result in tool_results:
            if result.success:
                tool_context += f"\n\n=== {result.tool_name.upper()} RESULTS ===\n{result.content}"
            else:
                tool_context += f"\n\n=== {result.tool_name.upper()} FAILED ===\n{result.content}"
        
        system_prompt = self.app_context.active_profile.system_prompt_commentary
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        
        try:
            response = self.model_client.generate(
                messages=messages,
                temperature=float(self.app_context.active_profile.conversational_temperature),
                max_tokens=int(self.app_context.active_profile.conversational_max_tokens),
                model=self.app_context.active_profile.conversational_llm_model
            )
            
            if response:
                # Clean up any reasoning content before storing and returning
                cleaned_response = self._strip_comprehensive_reasoning(response.strip())
                # Store in memory
                await self._store_interaction(user_text, cleaned_response, user_name)
                return cleaned_response
            else:
                # Raise exception instead of returning error message to trigger fallback
                raise Exception("Model returned empty response")
            
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Error generating response with tools: {e}")
            # Raise the exception instead of returning error message to trigger LLM service fallback
            raise Exception(f"Tool-Aware LLM connection failed: {e}")
    
    async def _store_interaction(self, user_text: str, response: str, user_name: str):
        """Store the interaction in memory"""
        if not self.memory_service:
            return
        
        try:
            # Store user message
            user_memory = MemoryEntry(
                content=f"User ({user_name}): {user_text}",
                source="user_query",
                timestamp=time.time(),
                metadata={
                    "user": user_name, 
                    "game": self.app_context.active_profile.game_name, 
                    "type": "user_input"
                }
            )
            self.memory_service.store_memory(user_memory)
            
            # Store bot response
            bot_memory = MemoryEntry(
                content=f"AI (Danzar): {response}",
                source="bot_response",
                timestamp=time.time(),
                metadata={
                    "user_query": user_text,
                    "user": user_name,
                    "game": self.app_context.active_profile.game_name,
                    "type": "bot_response",
                    "llm_model": self.app_context.active_profile.conversational_llm_model
                }
            )
            self.memory_service.store_memory(bot_memory)
            
        except Exception as e:
            self.logger.error(f"[ToolAwareLLM] Error storing interaction: {e}")
