# services/llm_service_multi.py

import asyncio
import time
import logging
from typing import Optional, Dict, Any, Tuple

from .llm_service import LLMService
from .multi_llm_coordinator import MultiLLMCoordinator

class EnhancedLLMService(LLMService):
    """
    Enhanced LLM Service with Multi-LLM Architecture
    
    Extends the existing LLMService to use the MultiLLMCoordinator
    for improved performance, reliability, and modularity.
    
    Benefits over single-LLM approach:
    - Faster response times through specialized models
    - Better reliability (no single point of failure)
    - Parallel processing capabilities
    - Specialized models for different tasks
    - Better debugging and monitoring
    """
    
    def __init__(self, app_context, audio_service, rag_service=None, model_client=None, default_collection: str = "multimodal_rag_default"):
        # Initialize the base LLMService
        super().__init__(app_context, audio_service, rag_service, model_client, default_collection)
        
        # Check if multi-LLM is enabled
        multi_llm_config = self.ctx.global_settings.get("MULTI_LLM", {})
        self.multi_llm_enabled = multi_llm_config.get("enabled", False)
        
        if self.multi_llm_enabled:
            try:
                # Initialize the multi-LLM coordinator
                self.multi_llm_coordinator: Optional[MultiLLMCoordinator] = MultiLLMCoordinator(app_context)
                self.logger.info("[EnhancedLLMService] Multi-LLM coordinator initialized successfully")
                
                # Track performance comparison
                self.single_llm_stats = {"calls": 0, "total_time": 0.0, "avg_time": 0.0}
                self.multi_llm_stats = {"calls": 0, "total_time": 0.0, "avg_time": 0.0}
                
            except Exception as e:
                self.logger.error(f"[EnhancedLLMService] Failed to initialize multi-LLM coordinator: {e}")
                self.multi_llm_coordinator = None
                self.multi_llm_enabled = False
                self.logger.warning("[EnhancedLLMService] Falling back to single-LLM mode")
        else:
            self.multi_llm_coordinator = None
            self.logger.info("[EnhancedLLMService] Multi-LLM disabled, using single-LLM mode")

    def handle_user_text_query(self, user_text: str, user_name: str = "User") -> str:
        """
        Enhanced query handling with multi-LLM support
        
        Uses multi-LLM coordinator if available, otherwise falls back to single-LLM
        """
        start_time = time.time()
        
        if self.multi_llm_enabled and self.multi_llm_coordinator is not None:
            return self._handle_query_multi_llm(user_text, user_name, start_time)
        else:
            return self._handle_query_single_llm(user_text, user_name, start_time)

    def _handle_query_multi_llm(self, user_text: str, user_name: str, start_time: float) -> str:
        """Handle query using multi-LLM coordinator"""
        try:
            self.logger.info(f"[EnhancedLLMService] Processing query with multi-LLM: '{user_text[:50]}...'")
            
            # Verify coordinator is available
            if self.multi_llm_coordinator is None:
                raise Exception("Multi-LLM coordinator not initialized")
            
            # Use async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response, metadata = loop.run_until_complete(
                    self.multi_llm_coordinator.process_query(user_text, user_name)
                )
                
                # Update performance stats
                execution_time = time.time() - start_time
                self._update_multi_llm_stats(execution_time)
                
                # Log performance metrics
                self.logger.info(
                    f"[EnhancedLLMService] Multi-LLM response generated in {execution_time:.2f}s "
                    f"(plan: {metadata.get('plan', {}).get('intent', 'unknown')}, "
                    f"nodes: {len(metadata.get('nodes_used', []))}, "
                    f"parallel_efficiency: {metadata.get('parallel_efficiency', 0):.2f})"
                )
                
                # Store response in memory and send to TTS queue
                self._store_and_send_response(response, user_text, user_name)
                
                return response
                
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Multi-LLM processing failed: {e}")
            # Fallback to single-LLM
            self.logger.info("[EnhancedLLMService] Falling back to single-LLM processing")
            return self._handle_query_single_llm(user_text, user_name, start_time)

    def _handle_query_single_llm(self, user_text: str, user_name: str, start_time: float) -> str:
        """Handle query using original single-LLM approach"""
        try:
            self.logger.info(f"[EnhancedLLMService] Processing query with single-LLM: '{user_text[:50]}...'")
            
            # Use the parent class method
            response = super().handle_user_text_query(user_text, user_name)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_single_llm_stats(execution_time)
            
            self.logger.info(f"[EnhancedLLMService] Single-LLM response generated in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Single-LLM processing failed: {e}")
            return "I'm having trouble processing that request right now. Please try again in a moment."

    def _update_multi_llm_stats(self, execution_time: float):
        """Update multi-LLM performance statistics"""
        self.multi_llm_stats["calls"] += 1
        self.multi_llm_stats["total_time"] += execution_time
        self.multi_llm_stats["avg_time"] = self.multi_llm_stats["total_time"] / self.multi_llm_stats["calls"]

    def _update_single_llm_stats(self, execution_time: float):
        """Update single-LLM performance statistics"""
        self.single_llm_stats["calls"] += 1
        self.single_llm_stats["total_time"] += execution_time
        self.single_llm_stats["avg_time"] = self.single_llm_stats["total_time"] / self.single_llm_stats["calls"]

    def get_performance_comparison(self) -> Dict[str, Any]:
        """Get performance comparison between single-LLM and multi-LLM"""
        comparison = {
            "single_llm": self.single_llm_stats.copy(),
            "multi_llm": self.multi_llm_stats.copy(),
            "multi_llm_enabled": self.multi_llm_enabled
        }
        
        # Calculate improvement metrics
        if self.single_llm_stats["calls"] > 0 and self.multi_llm_stats["calls"] > 0:
            single_avg = self.single_llm_stats["avg_time"]
            multi_avg = self.multi_llm_stats["avg_time"]
            
            if single_avg > 0:
                improvement = ((single_avg - multi_avg) / single_avg) * 100
                comparison["speed_improvement_percent"] = improvement
                comparison["faster_system"] = "multi-LLM" if improvement > 0 else "single-LLM"
        
        # Add coordinator stats if available
        if self.multi_llm_coordinator:
            comparison["coordinator_stats"] = self.multi_llm_coordinator.get_performance_stats()
        
        return comparison

    def log_performance_summary(self):
        """Log a summary of performance statistics"""
        comparison = self.get_performance_comparison()
        
        self.logger.info("=" * 60)
        self.logger.info("[EnhancedLLMService] Performance Summary")
        self.logger.info("=" * 60)
        
        if comparison["single_llm"]["calls"] > 0:
            stats = comparison["single_llm"]
            self.logger.info(f"Single-LLM: {stats['calls']} calls, avg {stats['avg_time']:.2f}s")
        
        if comparison["multi_llm"]["calls"] > 0:
            stats = comparison["multi_llm"]
            self.logger.info(f"Multi-LLM:  {stats['calls']} calls, avg {stats['avg_time']:.2f}s")
            
            if "speed_improvement_percent" in comparison:
                improvement = comparison["speed_improvement_percent"]
                faster = comparison["faster_system"]
                self.logger.info(f"Speed improvement: {improvement:.1f}% faster with {faster}")
        
        if comparison.get("coordinator_stats"):
            coord_stats = comparison["coordinator_stats"]
            self.logger.info(f"Coordinator: {coord_stats['total_queries']} queries, {coord_stats['parallel_efficiency']:.2f} efficiency")
            self.logger.info(f"Node usage: {coord_stats['node_usage']}")
        
        self.logger.info("=" * 60)

    def switch_to_multi_llm(self) -> bool:
        """
        Switch to multi-LLM mode if not already enabled
        
        Returns:
            True if successfully switched or already enabled, False otherwise
        """
        if self.multi_llm_enabled:
            self.logger.info("[EnhancedLLMService] Multi-LLM already enabled")
            return True
        
        try:
            self.multi_llm_coordinator = MultiLLMCoordinator(self.ctx)
            self.multi_llm_enabled = True
            self.logger.info("[EnhancedLLMService] Successfully switched to multi-LLM mode")
            return True
        except Exception as e:
            self.logger.error(f"[EnhancedLLMService] Failed to switch to multi-LLM: {e}")
            return False

    def switch_to_single_llm(self):
        """Switch to single-LLM mode"""
        if not self.multi_llm_enabled:
            self.logger.info("[EnhancedLLMService] Single-LLM already active")
            return
        
        self.multi_llm_enabled = False
        if self.multi_llm_coordinator:
            self.multi_llm_coordinator.cleanup()
            self.multi_llm_coordinator = None
        
        self.logger.info("[EnhancedLLMService] Switched to single-LLM mode")

    def cleanup(self):
        """Cleanup resources"""
        if self.multi_llm_coordinator:
            self.multi_llm_coordinator.cleanup()
        
        # Log final performance summary
        if self.single_llm_stats["calls"] > 0 or self.multi_llm_stats["calls"] > 0:
            self.log_performance_summary()
        
        self.logger.info("[EnhancedLLMService] Cleanup completed") 