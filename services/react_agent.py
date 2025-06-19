#!/usr/bin/env python3
"""
ReAct Agent Service for DanzarVLM
Implements ReAct (Reasoning + Acting) pattern for agentic decision making
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .agentic_memory import (
    MemoryQuery, MemoryType, MemoryPriority, AgenticAction, 
    AgenticMemoryService, MemoryNode
)

class AgentState(Enum):
    REASONING = "reasoning"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"

class ToolType(Enum):
    MEMORY_SEARCH = "memory_search"
    WEB_SEARCH = "web_search"
    FACT_CHECK = "fact_check"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    MEMORY_STORE = "memory_store"
    CONVERSATION_CONTEXT = "conversation_context"

@dataclass
class Observation:
    """Result of an action/tool execution"""
    tool_type: ToolType
    success: bool
    data: Any
    confidence: float
    reasoning: str
    timestamp: float

@dataclass
class ReActStep:
    """A single step in the ReAct reasoning process"""
    step_number: int
    thought: str
    action: str
    tool_type: ToolType
    tool_params: Dict[str, Any]
    observation: Optional[Observation] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class AgentSession:
    """Represents a complete agent reasoning session"""
    session_id: str
    user_name: str
    original_query: str
    steps: List[ReActStep]
    final_response: str
    confidence_score: float
    total_time: float
    success: bool
    context: Dict[str, Any]

class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent for intelligent query processing
    
    Implements the ReAct pattern:
    1. Thought: Reason about what to do next
    2. Action: Execute a tool/action
    3. Observation: Observe the result
    4. Repeat until satisfied or max steps reached
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = logging.getLogger("DanzarVLM.ReActAgent")
        
        # Initialize agent state
        self.max_steps = app_context.global_settings.get("REACT_AGENT", {}).get("max_steps", 5)
        self.confidence_threshold = app_context.global_settings.get("REACT_AGENT", {}).get("confidence_threshold", 0.7)
        
        # Tool availability (based on initialized services)
        self.available_tools = self._detect_available_tools()
        
        # Memory integration
        self.agentic_memory: Optional[AgenticMemoryService] = None
        
        # Session tracking
        self.active_sessions: Dict[str, AgentSession] = {}
        
        self.logger.info(f"[ReActAgent] Initialized with tools: {[tool.value for tool in self.available_tools]}")
    
    def set_agentic_memory(self, agentic_memory: AgenticMemoryService):
        """Set the agentic memory service"""
        self.agentic_memory = agentic_memory
        self.logger.info("[ReActAgent] Agentic memory service connected")
    
    def _detect_available_tools(self) -> List[ToolType]:
        """Detect which tools are available based on initialized services"""
        tools = []
        
        # Always available
        tools.append(ToolType.MEMORY_SEARCH)
        tools.append(ToolType.KNOWLEDGE_SYNTHESIS)
        tools.append(ToolType.MEMORY_STORE)
        tools.append(ToolType.CONVERSATION_CONTEXT)
        
        # Conditional on service availability
        if hasattr(self.app_context, 'smart_rag_service') and self.app_context.smart_rag_service:
            tools.append(ToolType.WEB_SEARCH)
            tools.append(ToolType.FACT_CHECK)
        
        return tools
    
    async def process_query(self, user_name: str, query: str, context: Dict[str, Any] = None) -> AgentSession:
        """
        Process a user query using ReAct reasoning
        
        Args:
            user_name: Name of the user asking the query
            query: The user's question
            context: Additional context (conversation history, etc.)
            
        Returns:
            AgentSession with complete reasoning process and final response
        """
        try:
            session_id = f"{user_name}_{int(time.time())}"
            start_time = time.time()
            
            self.logger.info(f"[ReActAgent] Starting session {session_id} for query: {query[:100]}...")
            
            # Initialize session
            session = AgentSession(
                session_id=session_id,
                user_name=user_name,
                original_query=query,
                steps=[],
                final_response="",
                confidence_score=0.0,
                total_time=0.0,
                success=False,
                context=context or {}
            )
            
            self.active_sessions[session_id] = session
            
            # ReAct Loop
            state = AgentState.REASONING
            step_number = 1
            
            while state != AgentState.COMPLETE and step_number <= self.max_steps:
                if state == AgentState.REASONING:
                    thought, action, tool_type, tool_params = await self._reasoning_step(session, step_number)
                    
                    step = ReActStep(
                        step_number=step_number,
                        thought=thought,
                        action=action,
                        tool_type=tool_type,
                        tool_params=tool_params
                    )
                    
                    session.steps.append(step)
                    state = AgentState.ACTING
                
                elif state == AgentState.ACTING:
                    current_step = session.steps[-1]
                    observation = await self._execute_tool(current_step.tool_type, current_step.tool_params, session)
                    current_step.observation = observation
                    
                    state = AgentState.OBSERVING
                
                elif state == AgentState.OBSERVING:
                    should_continue = await self._evaluate_observation(session)
                    if should_continue:
                        state = AgentState.REASONING
                        step_number += 1
                    else:
                        state = AgentState.COMPLETE
            
            # Generate final response
            session.final_response, session.confidence_score = await self._synthesize_final_response(session)
            session.total_time = time.time() - start_time
            session.success = session.confidence_score >= self.confidence_threshold
            
            self.logger.info(f"[ReActAgent] Session {session_id} completed in {session.total_time:.2f}s with confidence {session.confidence_score:.2f}")
            
            # Store session in memory if successful
            if session.success and self.agentic_memory:
                await self._store_session_memory(session)
            
            return session
            
        except Exception as e:
            self.logger.error(f"[ReActAgent] Query processing failed: {e}", exc_info=True)
            
            # Return failed session
            session.final_response = "I encountered an error while processing your request. Please try again."
            session.confidence_score = 0.0
            session.success = False
            session.total_time = time.time() - start_time
            
            return session
    
    async def _reasoning_step(self, session: AgentSession, step_number: int) -> Tuple[str, str, ToolType, Dict[str, Any]]:
        """
        Perform reasoning to determine next action
        
        Returns: (thought, action_description, tool_type, tool_parameters)
        """
        try:
            query = session.original_query
            context = session.context
            previous_steps = session.steps
            
            self.logger.debug(f"[ReActAgent] Reasoning step {step_number} for: {query[:50]}...")
            
            # Analyze query intent and current state
            if step_number == 1:
                # First step: Analyze query type and plan approach
                thought, action, tool_type, params = await self._plan_initial_approach(query, context)
            else:
                # Subsequent steps: Based on previous observations
                thought, action, tool_type, params = await self._plan_next_action(session, step_number)
            
            self.logger.debug(f"[ReActAgent] Step {step_number} - Thought: {thought[:100]}...")
            self.logger.debug(f"[ReActAgent] Step {step_number} - Action: {action}")
            
            return thought, action, tool_type, params
            
        except Exception as e:
            self.logger.error(f"[ReActAgent] Reasoning step failed: {e}")
            return (
                "I need to search for relevant information.",
                "Search memory for related information",
                ToolType.MEMORY_SEARCH,
                {"query": query, "user_name": session.user_name}
            )
    
    async def _plan_initial_approach(self, query: str, context: Dict[str, Any]) -> Tuple[str, str, ToolType, Dict[str, Any]]:
        """Plan the initial approach based on query analysis"""
        query_lower = query.lower()
        
        # Detect query patterns
        factual_patterns = ['what is', 'what are', 'who is', 'when did', 'where is', 'how many']
        procedural_patterns = ['how do i', 'how to', 'steps to', 'process of', 'way to']
        conversational_patterns = ['what about', 'also', 'and', 'too', 'it', 'that']
        game_specific_patterns = ['class', 'quest', 'skill', 'item', 'spell', 'level']
        
        # Check for follow-up conversation
        if any(pattern in query_lower for pattern in conversational_patterns):
            if context and context.get('has_recent_context'):
                thought = "This seems like a follow-up question. I should get the recent conversation context first."
                action = "Retrieve recent conversation context to understand the ongoing discussion"
                return thought, action, ToolType.CONVERSATION_CONTEXT, {"user_name": context.get("user_name"), "max_turns": 3}
        
        # Check for procedural queries
        if any(pattern in query_lower for pattern in procedural_patterns):
            thought = "This is a procedural question asking for steps or instructions. I should search for procedural knowledge first."
            action = "Search procedural memory for step-by-step instructions"
            return thought, action, ToolType.MEMORY_SEARCH, {
                "query": query,
                "memory_types": [MemoryType.PROCEDURAL],
                "user_name": context.get("user_name", "unknown")
            }
        
        # Check for game-specific factual queries
        if any(pattern in query_lower for pattern in game_specific_patterns):
            thought = "This is a game-specific question. I should check my game knowledge first, then search web if needed."
            action = "Search semantic memory for game-related facts"
            return thought, action, ToolType.MEMORY_SEARCH, {
                "query": query,
                "memory_types": [MemoryType.SEMANTIC, MemoryType.EPISODIC],
                "user_name": context.get("user_name", "unknown")
            }
        
        # Check for general factual queries  
        if any(pattern in query_lower for pattern in factual_patterns):
            thought = "This is a factual question. I should search my memory first, then consider web search if I don't have enough information."
            action = "Search memory for relevant factual information"
            return thought, action, ToolType.MEMORY_SEARCH, {
                "query": query,
                "memory_types": [MemoryType.SEMANTIC, MemoryType.EPISODIC],
                "user_name": context.get("user_name", "unknown")
            }
        
        # Default approach
        thought = "I'll start by searching my memory for any relevant information about this topic."
        action = "Search all memory types for relevant information"
        return thought, action, ToolType.MEMORY_SEARCH, {
            "query": query,
            "memory_types": [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
            "user_name": context.get("user_name", "unknown")
        }
    
    async def _plan_next_action(self, session: AgentSession, step_number: int) -> Tuple[str, str, ToolType, Dict[str, Any]]:
        """Plan next action based on previous observations"""
        last_step = session.steps[-1]
        last_observation = last_step.observation
        
        if not last_observation or not last_observation.success:
            # Previous action failed, try alternative approach
            thought = "The previous action didn't provide useful results. Let me try a different approach."
            
            if last_step.tool_type == ToolType.MEMORY_SEARCH and ToolType.WEB_SEARCH in self.available_tools:
                action = "Search the web for more current information"
                return thought, action, ToolType.WEB_SEARCH, {"query": session.original_query}
            else:
                action = "Synthesize whatever information I have available"
                return thought, action, ToolType.KNOWLEDGE_SYNTHESIS, {"session": session}
        
        # Analyze the quality and completeness of information
        if last_observation.confidence < 0.6:
            thought = "The information I found has low confidence. I should search for additional sources."
            
            if last_step.tool_type == ToolType.MEMORY_SEARCH and ToolType.WEB_SEARCH in self.available_tools:
                action = "Search web to verify and supplement the information"
                return thought, action, ToolType.WEB_SEARCH, {"query": session.original_query}
            elif ToolType.FACT_CHECK in self.available_tools:
                action = "Fact-check the information I found"
                return thought, action, ToolType.FACT_CHECK, {"information": last_observation.data}
        
        # If we have good information, synthesize it
        if last_observation.confidence >= 0.6:
            thought = "I have found relevant information with good confidence. Let me synthesize a comprehensive response."
            action = "Synthesize the gathered information into a complete answer"
            return thought, action, ToolType.KNOWLEDGE_SYNTHESIS, {"session": session}
        
        # Default: try to gather more context
        thought = "I need more context to provide a better answer."
        action = "Get additional conversation context"
        return thought, action, ToolType.CONVERSATION_CONTEXT, {
            "user_name": session.user_name,
            "max_turns": 5
        }
    
    async def _execute_tool(self, tool_type: ToolType, params: Dict[str, Any], session: AgentSession) -> Observation:
        """Execute a specific tool and return observation"""
        try:
            start_time = time.time()
            self.logger.debug(f"[ReActAgent] Executing tool: {tool_type.value}")
            
            if tool_type == ToolType.MEMORY_SEARCH:
                return await self._execute_memory_search(params)
            elif tool_type == ToolType.WEB_SEARCH:
                return await self._execute_web_search(params)
            elif tool_type == ToolType.FACT_CHECK:
                return await self._execute_fact_check(params)
            elif tool_type == ToolType.KNOWLEDGE_SYNTHESIS:
                return await self._execute_knowledge_synthesis(params)
            elif tool_type == ToolType.MEMORY_STORE:
                return await self._execute_memory_store(params)
            elif tool_type == ToolType.CONVERSATION_CONTEXT:
                return await self._execute_conversation_context(params)
            else:
                return Observation(
                    tool_type=tool_type,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning=f"Tool {tool_type.value} not implemented",
                    timestamp=time.time()
                )
                
        except Exception as e:
            self.logger.error(f"[ReActAgent] Tool execution failed: {e}")
            return Observation(
                tool_type=tool_type,
                success=False,
                data=None,
                confidence=0.0,
                reasoning=f"Tool execution error: {str(e)}",
                timestamp=time.time()
            )
    
    async def _execute_memory_search(self, params: Dict[str, Any]) -> Observation:
        """Execute memory search tool"""
        try:
            if not self.agentic_memory:
                return Observation(
                    tool_type=ToolType.MEMORY_SEARCH,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning="Agentic memory service not available",
                    timestamp=time.time()
                )
            
            # Create memory query
            query = MemoryQuery(
                query_text=params["query"],
                memory_types=params.get("memory_types", [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]),
                user_name=params.get("user_name", "unknown"),
                context=params,
                max_results=params.get("max_results", 10)
            )
            
            # Execute agentic query
            memories, actions = await self.agentic_memory.agentic_query(query)
            
            # Calculate confidence based on results
            if memories:
                avg_relevance = sum(
                    self.agentic_memory._calculate_relevance_score(memory, query) 
                    for memory in memories
                ) / len(memories)
                confidence = min(0.95, avg_relevance * 1.2)  # Boost confidence slightly
            else:
                confidence = 0.0
            
            return Observation(
                tool_type=ToolType.MEMORY_SEARCH,
                success=len(memories) > 0,
                data={
                    "memories": memories,
                    "suggested_actions": actions,
                    "memory_count": len(memories)
                },
                confidence=confidence,
                reasoning=f"Found {len(memories)} relevant memories with average relevance {avg_relevance:.2f}" if memories else "No relevant memories found",
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"[ReActAgent] Memory search failed: {e}")
            return Observation(
                tool_type=ToolType.MEMORY_SEARCH,
                success=False,
                data=None,
                confidence=0.0,
                reasoning=f"Memory search error: {str(e)}",
                timestamp=time.time()
            )
    
    async def _execute_web_search(self, params: Dict[str, Any]) -> Observation:
        """Execute web search tool"""
        try:
            smart_rag_service = getattr(self.app_context, 'smart_rag_service', None)
            
            if not smart_rag_service:
                return Observation(
                    tool_type=ToolType.WEB_SEARCH,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning="Smart RAG service not available for web search",
                    timestamp=time.time()
                )
            
            # Execute web search through Smart RAG
            query = params["query"]
            
            # Use Smart RAG's web search capability
            if hasattr(smart_rag_service, '_search_web'):
                search_results = smart_rag_service._search_web(query, max_results=5)
                
                if search_results:
                    # Store search results in memory if agentic memory is available
                    if self.agentic_memory:
                        for result in search_results:
                            self.agentic_memory.store_memory(
                                content=f"Web search result: {result}",
                                memory_type=MemoryType.SEMANTIC,
                                user_name=params.get("user_name", "system"),
                                priority=MemoryPriority.MEDIUM,
                                metadata={
                                    "source": "web_search",
                                    "query": query,
                                    "timestamp": time.time()
                                }
                            )
                    
                    return Observation(
                        tool_type=ToolType.WEB_SEARCH,
                        success=True,
                        data={
                            "results": search_results,
                            "result_count": len(search_results)
                        },
                        confidence=0.8,  # Web search generally reliable
                        reasoning=f"Found {len(search_results)} web search results",
                        timestamp=time.time()
                    )
                else:
                    return Observation(
                        tool_type=ToolType.WEB_SEARCH,
                        success=False,
                        data=None,
                        confidence=0.0,
                        reasoning="Web search returned no results",
                        timestamp=time.time()
                    )
            else:
                return Observation(
                    tool_type=ToolType.WEB_SEARCH,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning="Web search functionality not available in Smart RAG",
                    timestamp=time.time()
                )
                
        except Exception as e:
            self.logger.error(f"[ReActAgent] Web search failed: {e}")
            return Observation(
                tool_type=ToolType.WEB_SEARCH,
                success=False,
                data=None,
                confidence=0.0,
                reasoning=f"Web search error: {str(e)}",
                timestamp=time.time()
            )
    
    async def _execute_fact_check(self, params: Dict[str, Any]) -> Observation:
        """Execute fact checking tool"""
        try:
            # This would integrate with fact-checking services
            # For now, return a placeholder
            information = params.get("information", "")
            
            return Observation(
                tool_type=ToolType.FACT_CHECK,
                success=True,
                data={
                    "verified": True,
                    "confidence": 0.7,
                    "information": information
                },
                confidence=0.7,
                reasoning="Fact checking completed (placeholder implementation)",
                timestamp=time.time()
            )
            
        except Exception as e:
            return Observation(
                tool_type=ToolType.FACT_CHECK,
                success=False,
                data=None,
                confidence=0.0,
                reasoning=f"Fact check error: {str(e)}",
                timestamp=time.time()
            )
    
    async def _execute_knowledge_synthesis(self, params: Dict[str, Any]) -> Observation:
        """Execute knowledge synthesis tool"""
        try:
            session = params["session"]
            
            # Gather all information from previous steps
            all_information = []
            sources = []
            
            for step in session.steps:
                if step.observation and step.observation.success:
                    if step.tool_type == ToolType.MEMORY_SEARCH:
                        memories = step.observation.data.get("memories", [])
                        for memory in memories:
                            all_information.append(memory.content)
                            sources.append(f"memory_{memory.memory_type.value}")
                    elif step.tool_type == ToolType.WEB_SEARCH:
                        results = step.observation.data.get("results", [])
                        all_information.extend(results)
                        sources.extend(["web_search"] * len(results))
                    elif step.tool_type == ToolType.CONVERSATION_CONTEXT:
                        context = step.observation.data.get("context", "")
                        if context:
                            all_information.append(context)
                            sources.append("conversation_context")
            
            if not all_information:
                return Observation(
                    tool_type=ToolType.KNOWLEDGE_SYNTHESIS,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning="No information available to synthesize",
                    timestamp=time.time()
                )
            
            # Calculate synthesis confidence based on source diversity and quality
            unique_sources = set(sources)
            source_diversity = len(unique_sources) / max(1, len(sources))
            confidence = min(0.9, 0.5 + (source_diversity * 0.4))
            
            return Observation(
                tool_type=ToolType.KNOWLEDGE_SYNTHESIS,
                success=True,
                data={
                    "synthesized_information": all_information,
                    "sources": sources,
                    "source_count": len(unique_sources)
                },
                confidence=confidence,
                reasoning=f"Synthesized information from {len(unique_sources)} different sources",
                timestamp=time.time()
            )
            
        except Exception as e:
            return Observation(
                tool_type=ToolType.KNOWLEDGE_SYNTHESIS,
                success=False,
                data=None,
                confidence=0.0,
                reasoning=f"Knowledge synthesis error: {str(e)}",
                timestamp=time.time()
            )
    
    async def _execute_memory_store(self, params: Dict[str, Any]) -> Observation:
        """Execute memory storage tool"""
        try:
            if not self.agentic_memory:
                return Observation(
                    tool_type=ToolType.MEMORY_STORE,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning="Agentic memory service not available",
                    timestamp=time.time()
                )
            
            content = params["content"]
            memory_type = params.get("memory_type", MemoryType.EPISODIC)
            user_name = params.get("user_name", "system")
            priority = params.get("priority", MemoryPriority.MEDIUM)
            metadata = params.get("metadata", {})
            
            memory_id = self.agentic_memory.store_memory(
                content=content,
                memory_type=memory_type,
                user_name=user_name,
                priority=priority,
                metadata=metadata
            )
            
            return Observation(
                tool_type=ToolType.MEMORY_STORE,
                success=bool(memory_id),
                data={"memory_id": memory_id},
                confidence=0.9 if memory_id else 0.0,
                reasoning=f"Stored memory with ID: {memory_id}" if memory_id else "Failed to store memory",
                timestamp=time.time()
            )
            
        except Exception as e:
            return Observation(
                tool_type=ToolType.MEMORY_STORE,
                success=False,
                data=None,
                confidence=0.0,
                reasoning=f"Memory storage error: {str(e)}",
                timestamp=time.time()
            )
    
    async def _execute_conversation_context(self, params: Dict[str, Any]) -> Observation:
        """Execute conversation context retrieval"""
        try:
            conversation_memory = getattr(self.app_context, 'conversation_memory_service', None)
            
            if not conversation_memory:
                return Observation(
                    tool_type=ToolType.CONVERSATION_CONTEXT,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning="Conversation memory service not available",
                    timestamp=time.time()
                )
            
            user_name = params["user_name"]
            max_turns = params.get("max_turns", 3)
            
            # Get conversation context
            context_turns = conversation_memory.get_conversation_context(user_name, max_turns)
            
            if context_turns:
                # Format context for use
                context_text = "\n".join([
                    f"User: {turn.user_message}\nAI: {turn.bot_response}"
                    for turn in context_turns
                ])
                
                return Observation(
                    tool_type=ToolType.CONVERSATION_CONTEXT,
                    success=True,
                    data={
                        "context": context_text,
                        "turn_count": len(context_turns),
                        "turns": context_turns
                    },
                    confidence=0.8,
                    reasoning=f"Retrieved {len(context_turns)} conversation turns",
                    timestamp=time.time()
                )
            else:
                return Observation(
                    tool_type=ToolType.CONVERSATION_CONTEXT,
                    success=False,
                    data=None,
                    confidence=0.0,
                    reasoning="No recent conversation context found",
                    timestamp=time.time()
                )
                
        except Exception as e:
            return Observation(
                tool_type=ToolType.CONVERSATION_CONTEXT,
                success=False,
                data=None,
                confidence=0.0,
                reasoning=f"Context retrieval error: {str(e)}",
                timestamp=time.time()
            )
    
    async def _evaluate_observation(self, session: AgentSession) -> bool:
        """Evaluate whether to continue reasoning or finish"""
        last_step = session.steps[-1]
        last_observation = last_step.observation
        
        # Continue if we don't have enough confidence
        if not last_observation or last_observation.confidence < self.confidence_threshold:
            return len(session.steps) < self.max_steps  # Continue if we haven't reached max steps
        
        # Continue if we haven't attempted knowledge synthesis yet
        synthesis_attempted = any(
            step.tool_type == ToolType.KNOWLEDGE_SYNTHESIS 
            for step in session.steps
        )
        
        if not synthesis_attempted and len(session.steps) < self.max_steps:
            return True
        
        # Stop if we have good confidence or reached max steps
        return False
    
    async def _synthesize_final_response(self, session: AgentSession) -> Tuple[str, float]:
        """Synthesize final response from all gathered information"""
        try:
            # Look for synthesis results first
            synthesis_data = None
            for step in reversed(session.steps):
                if (step.tool_type == ToolType.KNOWLEDGE_SYNTHESIS and 
                    step.observation and step.observation.success):
                    synthesis_data = step.observation.data
                    break
            
            if synthesis_data:
                # Use synthesized information
                information = synthesis_data["synthesized_information"]
                sources = synthesis_data["sources"]
                
                # Create comprehensive response
                response_parts = []
                
                # Add main information
                for info in information[:3]:  # Limit to top 3 pieces of info
                    if len(info) > 200:
                        info = info[:200] + "..."
                    response_parts.append(info)
                
                final_response = "\n\n".join(response_parts)
                
                # Calculate confidence based on source diversity
                unique_sources = set(sources)
                confidence = min(0.9, 0.6 + (len(unique_sources) * 0.1))
                
                return final_response, confidence
            
            # Fallback: gather information from individual steps
            response_parts = []
            total_confidence = 0.0
            confidence_count = 0
            
            for step in session.steps:
                if step.observation and step.observation.success:
                    total_confidence += step.observation.confidence
                    confidence_count += 1
                    
                    if step.tool_type == ToolType.MEMORY_SEARCH:
                        memories = step.observation.data.get("memories", [])
                        for memory in memories[:2]:  # Top 2 memories
                            content = memory.content
                            if len(content) > 150:
                                content = content[:150] + "..."
                            response_parts.append(content)
                    
                    elif step.tool_type == ToolType.WEB_SEARCH:
                        results = step.observation.data.get("results", [])
                        for result in results[:2]:  # Top 2 results
                            if len(result) > 150:
                                result = result[:150] + "..."
                            response_parts.append(result)
            
            if response_parts:
                final_response = "\n\n".join(response_parts)
                avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
                return final_response, avg_confidence
            else:
                return "I couldn't find sufficient information to answer your question. Please try rephrasing or asking about something else.", 0.1
                
        except Exception as e:
            self.logger.error(f"[ReActAgent] Response synthesis failed: {e}")
            return "I encountered an error while processing your request. Please try again.", 0.1
    
    async def _store_session_memory(self, session: AgentSession):
        """Store successful session as memory for future reference"""
        try:
            if not self.agentic_memory:
                return
            
            # Store the reasoning process as procedural memory
            reasoning_content = f"Query: {session.original_query}\n"
            reasoning_content += f"Approach: {' -> '.join([step.action for step in session.steps])}\n"
            reasoning_content += f"Result: {session.final_response[:200]}..."
            
            self.agentic_memory.store_memory(
                content=reasoning_content,
                memory_type=MemoryType.PROCEDURAL,
                user_name=session.user_name,
                priority=MemoryPriority.HIGH if session.confidence_score > 0.8 else MemoryPriority.MEDIUM,
                metadata={
                    "type": "agent_session",
                    "session_id": session.session_id,
                    "confidence": session.confidence_score,
                    "steps": len(session.steps),
                    "tools_used": [step.tool_type.value for step in session.steps]
                }
            )
            
            # Store the final response as episodic memory
            response_content = f"Q: {session.original_query}\nA: {session.final_response}"
            
            self.agentic_memory.store_memory(
                content=response_content,
                memory_type=MemoryType.EPISODIC,
                user_name=session.user_name,
                priority=MemoryPriority.HIGH if session.confidence_score > 0.8 else MemoryPriority.MEDIUM,
                metadata={
                    "type": "qa_pair",
                    "session_id": session.session_id,
                    "confidence": session.confidence_score,
                    "timestamp": time.time()
                }
            )
            
            self.logger.debug(f"[ReActAgent] Stored session memory for {session.session_id}")
            
        except Exception as e:
            self.logger.error(f"[ReActAgent] Failed to store session memory: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about agent sessions"""
        return {
            "active_sessions": len(self.active_sessions),
            "available_tools": [tool.value for tool in self.available_tools],
            "max_steps": self.max_steps,
            "confidence_threshold": self.confidence_threshold
        }