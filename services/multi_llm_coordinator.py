# services/multi_llm_coordinator.py

import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .model_client import ModelClient

class LLMSpecialization(Enum):
    """Different LLM specializations for multi-LLM architecture"""
    COORDINATOR = "coordinator"  # Main planning and intent classification
    RETRIEVER = "retriever"     # Embedding and search operations
    GENERATOR = "generator"     # Response generation
    MEMORY = "memory"          # Memory reasoning and context
    FACTCHECK = "factcheck"    # Fact verification

@dataclass
class LLMNode:
    """Represents a specialized LLM node"""
    name: str
    specialization: LLMSpecialization
    model_name: str
    endpoint: str
    client: ModelClient
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30

@dataclass
class QueryPlan:
    """Execution plan for multi-LLM processing"""
    intent: str
    confidence: float
    required_nodes: List[str]
    execution_order: List[str]
    parallel_tasks: List[List[str]]
    estimated_time: float

class MultiLLMCoordinator:
    """
    Multi-LLM Architecture Coordinator
    
    Implements the multi-LLM pattern you described:
    - Coordinator/Planner: Main LLM for intent classification and planning
    - Retriever+Embedder: Specialized embedding models (nomic-embed-text)
    - Generator: Specialized response generation  
    - Memory: Memory reasoning
    - Advantages: Concurrency, efficiency, modularity, better debugging
    """
    
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = logging.getLogger("DanzarVLM.MultiLLMCoordinator")
        
        # Initialize LLM nodes
        self.nodes: Dict[str, LLMNode] = {}
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'node_usage': {},
            'parallel_efficiency': 0.0
        }
        
        self._initialize_llm_nodes()
        self.logger.info(f"[MultiLLMCoordinator] Initialized with {len(self.nodes)} specialized LLM nodes")

    def _initialize_llm_nodes(self):
        """Initialize specialized LLM nodes with different models and configurations"""
        base_url = self.ctx.global_settings.get("LLAMA_API_BASE_URL", "http://localhost:8080")
        
        # Coordinator LLM - Fast, reliable for planning and intent classification
        coordinator_model = self.ctx.global_settings.get("COORDINATOR_MODEL", "qwen2.5:3b")
        self.nodes["coordinator"] = LLMNode(
            name="coordinator",
            specialization=LLMSpecialization.COORDINATOR,
            model_name=coordinator_model,
            endpoint=f"{base_url}/api/chat",
            client=ModelClient(
                api_base_url=f"{base_url}/api/chat",
                app_context=self.ctx
            ),
            max_tokens=512,  # Short, focused responses
            temperature=0.3  # More deterministic for planning
        )
        
        # Retriever LLM - Specialized for embeddings and search
        retriever_model = self.ctx.global_settings.get("RETRIEVER_MODEL", "nomic-embed-text:latest")
        self.nodes["retriever"] = LLMNode(
            name="retriever",
            specialization=LLMSpecialization.RETRIEVER,
            model_name=retriever_model,
            endpoint=f"{base_url}/api/embeddings",
            client=ModelClient(
                api_base_url=f"{base_url}/api/embeddings",
                app_context=self.ctx
            ),
            max_tokens=1024,
            temperature=0.1  # Very deterministic for search
        )
        
        # Generator LLM - Specialized for high-quality response generation
        generator_model = self.ctx.global_settings.get("GENERATOR_MODEL", "qwen2.5:7b")
        self.nodes["generator"] = LLMNode(
            name="generator",
            specialization=LLMSpecialization.GENERATOR,
            model_name=generator_model,
            endpoint=f"{base_url}/api/chat",
            client=ModelClient(
                api_base_url=f"{base_url}/api/chat",
                app_context=self.ctx
            ),
            max_tokens=2048,
            temperature=0.7  # Creative but controlled
        )
        
        # Memory LLM - Specialized for context and memory reasoning
        memory_model = self.ctx.global_settings.get("MEMORY_MODEL", "qwen2.5:3b")
        self.nodes["memory"] = LLMNode(
            name="memory",
            specialization=LLMSpecialization.MEMORY,
            model_name=memory_model,
            endpoint=f"{base_url}/api/chat", 
            client=ModelClient(
                api_base_url=f"{base_url}/api/chat",
                app_context=self.ctx
            ),
            max_tokens=1024,
            temperature=0.4  # Balanced for memory tasks
        )
        
        # Fact-check LLM - Specialized for verification and cross-referencing
        factcheck_model = self.ctx.global_settings.get("FACTCHECK_MODEL", "qwen2.5:3b")
        self.nodes["factcheck"] = LLMNode(
            name="factcheck",
            specialization=LLMSpecialization.FACTCHECK,
            model_name=factcheck_model,
            endpoint=f"{base_url}/api/chat",
            client=ModelClient(
                api_base_url=f"{base_url}/api/chat",
                app_context=self.ctx
            ),
            max_tokens=512,
            temperature=0.2  # Very careful and precise
        )

    async def process_query(self, query: str, user_name: str = "User") -> Tuple[str, Dict[str, Any]]:
        """
        Process query through multi-LLM architecture
        
        Returns:
            Tuple of (response_text, metadata with performance stats)
        """
        start_time = time.time()
        self.performance_stats['total_queries'] += 1
        
        try:
            # Step 1: Coordinator plans the execution
            plan = await self._plan_execution(query, user_name)
            self.logger.info(f"[MultiLLMCoordinator] Query plan: {plan.intent} (confidence: {plan.confidence:.2f})")
            
            # Step 2: Execute plan with parallel processing where possible
            context_data = await self._execute_plan(plan, query, user_name)
            
            # Step 3: Generate final response
            response = await self._generate_final_response(query, plan, context_data, user_name)
            
            # Update performance stats
            total_time = time.time() - start_time
            self._update_performance_stats(total_time, plan)
            
            metadata = {
                'execution_time': total_time,
                'plan': plan.__dict__,  # Convert to dict for JSON serialization
                'nodes_used': plan.required_nodes,
                'parallel_efficiency': self.performance_stats['parallel_efficiency']
            }
            
            return response, metadata
            
        except Exception as e:
            self.logger.error(f"[MultiLLMCoordinator] Error processing query: {e}", exc_info=True)
            # Fallback to simple single-LLM response
            fallback_response = await self._fallback_response(query)
            return fallback_response, {'error': str(e), 'fallback': True, 'execution_time': time.time() - start_time}

    async def _plan_execution(self, query: str, user_name: str) -> QueryPlan:
        """Use coordinator LLM to plan multi-LLM execution"""
        
        planning_prompt = f"""
        Analyze this user query and create an execution plan for multi-LLM processing.
        
        User: {user_name}
        Query: "{query}"
        
        Available LLM nodes:
        - coordinator: Intent classification, planning
        - retriever: Embedding search, RAG retrieval  
        - generator: High-quality response generation
        - memory: Context and memory reasoning
        - factcheck: Fact verification
        
        Classify the intent and determine which nodes are needed:
        
        Intent options:
        - conversational: Simple chat, no data needed
        - game_specific: EverQuest game information needed
        - factual: General facts, may need verification
        - complex: Multiple sources and reasoning needed
        
        Respond in JSON format:
        {{
            "intent": "intent_type",
            "confidence": 0.0-1.0,
            "required_nodes": ["node1", "node2"],
            "execution_order": ["step1", "step2", "step3"],
            "parallel_tasks": [["task1", "task2"], ["task3"]],
            "estimated_time": 5.0
        }}
        """
        
        try:
            # Use the node's client directly instead of the broken _call_node_async
            coordinator = self.nodes["coordinator"]
            messages = [{"role": "user", "content": planning_prompt}]
            
            response = coordinator.client.generate(
                messages,
                model=coordinator.model_name,
                temperature=coordinator.temperature,
                max_tokens=512
            )
            
            if not response:
                raise Exception("Empty response from coordinator")
            
            # Parse JSON response
            import json
            plan_data = json.loads(response.strip())
            
            return QueryPlan(
                intent=plan_data.get("intent", "conversational"),
                confidence=plan_data.get("confidence", 0.5),
                required_nodes=plan_data.get("required_nodes", ["generator"]),
                execution_order=plan_data.get("execution_order", ["generator"]),
                parallel_tasks=plan_data.get("parallel_tasks", []),
                estimated_time=plan_data.get("estimated_time", 5.0)
            )
            
        except Exception as e:
            self.logger.warning(f"[MultiLLMCoordinator] Planning failed, using default plan: {e}")
            # Default simple plan
            return QueryPlan(
                intent="conversational",
                confidence=0.3,
                required_nodes=["generator"],
                execution_order=["generator"],
                parallel_tasks=[],
                estimated_time=3.0
            )

    async def _execute_plan(self, plan: QueryPlan, query: str, user_name: str) -> Dict[str, Any]:
        """Execute the planned multi-LLM workflow"""
        context_data = {
            'query': query,
            'user_name': user_name,
            'intent': plan.intent,
            'retrieved_context': [],
            'memory_context': [],
            'verified_facts': []
        }
        
        # Execute parallel tasks first
        for parallel_group in plan.parallel_tasks:
            tasks = []
            for task_node in parallel_group:
                if task_node in plan.required_nodes:
                    task = self._execute_node_task(task_node, query, context_data)
                    tasks.append(task)
            
            # Wait for parallel tasks to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if not isinstance(result, Exception):
                        node_name = parallel_group[i]
                        context_data[f'{node_name}_result'] = result
        
        # Execute sequential tasks
        for step in plan.execution_order:
            if step in plan.required_nodes and step not in [task for group in plan.parallel_tasks for task in group]:
                result = await self._execute_node_task(step, query, context_data)
                context_data[f'{step}_result'] = result
        
        return context_data

    async def _execute_node_task(self, node_name: str, query: str, context_data: Dict[str, Any]) -> Any:
        """Execute a specific node task"""
        try:
            if node_name == "retriever":
                return await self._retriever_task(query, context_data)
            elif node_name == "memory":
                return await self._memory_task(query, context_data)
            elif node_name == "factcheck":
                return await self._factcheck_task(query, context_data)
            elif node_name == "generator":
                return await self._generator_task(query, context_data)
            else:
                self.logger.warning(f"[MultiLLMCoordinator] Unknown node task: {node_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"[MultiLLMCoordinator] Error executing {node_name} task: {e}")
            return None

    async def _retriever_task(self, query: str, context_data: Dict[str, Any]) -> List[str]:
        """Execute retrieval task using OllamaRAGService"""
        try:
            # Import and initialize OllamaRAGService
            from .ollama_rag_service import OllamaRAGService
            
            rag_service = OllamaRAGService(self.ctx)
            
            # Determine collection based on intent
            intent = context_data.get('intent', 'conversational')
            if intent == 'game_specific':
                collection = "Everquest"  # Use the collection with 11,185 documents
            else:
                collection = "danzar_knowledge"  # Default collection
            
            # Check if collection exists
            if not await asyncio.to_thread(rag_service.collection_exists, collection):
                self.logger.warning(f"[MultiLLMCoordinator] Collection '{collection}' not found, skipping retrieval")
                return []
            
            # Perform retrieval
            results = await asyncio.to_thread(rag_service.query, collection, query, 3)
            
            context_pieces = []
            if results:
                for doc in results:
                    if isinstance(doc, str):
                        context_pieces.append(doc)
                    elif isinstance(doc, dict) and 'content' in doc:
                        context_pieces.append(doc['content'])
            
            context_data['retrieved_context'] = context_pieces
            self.logger.info(f"[MultiLLMCoordinator] Retrieved {len(context_pieces)} context pieces from {collection}")
            return context_pieces
            
        except Exception as e:
            self.logger.error(f"[MultiLLMCoordinator] Retrieval task failed: {e}")
            return []

    async def _memory_task(self, query: str, context_data: Dict[str, Any]) -> List[str]:
        """Execute memory reasoning task"""
        memory_prompt = f"""
        Analyze this query for relevant conversation context and memories:
        
        Query: "{query}"
        Intent: {context_data['intent']}
        
        Look for:
        1. Recent conversation references
        2. User preferences or history
        3. Context that would improve the response
        
        Return relevant memory context as a JSON list of strings:
        ["memory1", "memory2", ...]
        """
        
        try:
            response = await self._call_node_async("memory", memory_prompt, 512)
            import json
            memory_context = json.loads(response)
            context_data['memory_context'] = memory_context
            return memory_context
        except:
            return []

    async def _factcheck_task(self, query: str, context_data: Dict[str, Any]) -> List[str]:
        """Execute fact-checking task"""
        if not context_data.get('retrieved_context'):
            return []
            
        factcheck_prompt = f"""
        Verify the accuracy of this retrieved information for the query:
        
        Query: "{query}"
        Retrieved Information:
        {chr(10).join(context_data['retrieved_context'])}
        
        Rate accuracy and provide verified facts as JSON:
        {{
            "accuracy_score": 0.0-1.0,
            "verified_facts": ["fact1", "fact2"],
            "concerns": ["concern1"]
        }}
        """
        
        try:
            response = await self._call_node_async("factcheck", factcheck_prompt, 512)
            import json
            factcheck_result = json.loads(response)
            verified_facts = factcheck_result.get('verified_facts', [])
            context_data['verified_facts'] = verified_facts
            return verified_facts
        except:
            return []

    async def _generator_task(self, query: str, context_data: Dict[str, Any]) -> str:
        """Execute response generation task"""
        # This will be called in _generate_final_response
        return ""

    async def _generate_final_response(self, query: str, plan: QueryPlan, context_data: Dict[str, Any], user_name: str) -> str:
        """Generate final response using the generator LLM"""
        
        # Build context for generator
        context_parts = []
        
        if context_data.get('retrieved_context'):
            context_parts.append("Retrieved Information:")
            context_parts.extend(context_data['retrieved_context'])
            context_parts.append("")
            
        if context_data.get('memory_context'):
            context_parts.append("Conversation Context:")
            context_parts.extend(context_data['memory_context'])
            context_parts.append("")
            
        if context_data.get('verified_facts'):
            context_parts.append("Verified Facts:")
            context_parts.extend(context_data['verified_facts'])
            context_parts.append("")
        
        context_text = "\n".join(context_parts) if context_parts else ""
        
        generation_prompt = f"""
        You are DanzarAI, a helpful gaming assistant specializing in EverQuest.
        
        User: {user_name}
        Query: "{query}"
        Intent: {plan.intent}
        
        {context_text}
        
        Provide a helpful, accurate response. Be conversational and engaging.
        If you have specific information from the context, use it. If not, provide general helpful guidance.
        """
        
        try:
            response = await self._call_node_async("generator", generation_prompt, 2048)
            return response.strip()
        except Exception as e:
            self.logger.error(f"[MultiLLMCoordinator] Generation failed: {e}")
            return "I'm having trouble processing that request right now. Please try again in a moment."

    async def _call_node_async(self, node_name: str, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Call a specific LLM node asynchronously"""
        if node_name not in self.nodes:
            raise ValueError(f"Unknown node: {node_name}")
            
        node = self.nodes[node_name]
        
        # Track node usage
        if node_name not in self.performance_stats['node_usage']:
            self.performance_stats['node_usage'][node_name] = 0
        self.performance_stats['node_usage'][node_name] += 1
        
        # Use the node's client to make the call
        messages = [{"role": "user", "content": prompt}]
        
        def make_request():
            return node.client.generate(
                messages,
                model=node.model_name,
                temperature=node.temperature,
                max_tokens=max_tokens or node.max_tokens
            )
        
        result = await asyncio.to_thread(make_request)
        if not result:
            raise Exception(f"Empty response from {node_name} node")
        return result

    async def _fallback_response(self, query: str) -> str:
        """Fallback to simple single-LLM response"""
        try:
            generator = self.nodes.get("generator") or self.nodes.get("coordinator")
            if generator:
                simple_prompt = f"Please provide a helpful response to: {query}"
                return await self._call_node_async(generator.name, simple_prompt, 1024)
        except:
            pass
        return "I'm experiencing technical difficulties. Please try again later."

    def _update_performance_stats(self, execution_time: float, plan: QueryPlan):
        """Update performance statistics"""
        # Update average response time
        total_queries = self.performance_stats['total_queries']
        current_avg = self.performance_stats['avg_response_time']
        self.performance_stats['avg_response_time'] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )
        
        # Calculate parallel efficiency
        if plan.parallel_tasks:
            estimated_sequential = plan.estimated_time
            actual_time = execution_time
            efficiency = min(1.0, estimated_sequential / actual_time) if actual_time > 0 else 0.0
            self.performance_stats['parallel_efficiency'] = (
                (self.performance_stats['parallel_efficiency'] * 0.9) + (efficiency * 0.1)
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()

    def cleanup(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.logger.info("[MultiLLMCoordinator] Cleaned up resources") 