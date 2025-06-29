# services/vision_tools.py
"""
Vision Tools for Agentic Behavior
================================

This module provides tool-based vision capabilities that can be used by the LLM
to naturally request visual information when needed. The LLM can "decide" to
call these tools when it needs to see what's happening on screen.
"""

import asyncio
import time
import base64
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from PIL import Image
import io
import cv2
import numpy as np

@dataclass
class VisionToolResult:
    """Result from a vision tool operation."""
    success: bool
    data: Any
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class VisionTools:
    """Collection of vision-related tools for agentic behavior."""
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.vision_integration_service = None
        self.last_screenshot_time = 0
        self.screenshot_cooldown = 2.0  # Minimum seconds between screenshots
        
        # Tool descriptions for the LLM
        self.tool_descriptions = {
            "capture_screenshot": {
                "name": "capture_screenshot",
                "description": "Capture a screenshot of the current game screen from OBS NDI stream. Use this when you need to see what's currently happening in the game.",
                "parameters": {
                    "reason": "string - Why you need to see the screen (e.g., 'checking inventory', 'analyzing combat', 'reading UI text')"
                }
            },
            "analyze_screenshot": {
                "name": "analyze_screenshot", 
                "description": "Analyze a screenshot to detect objects, read text, and understand what's happening. Use this to get detailed information about the current game state.",
                "parameters": {
                    "focus_areas": "list of strings - Specific areas to focus on (e.g., ['inventory', 'health_bar', 'chat_window'])",
                    "analysis_type": "string - Type of analysis ('general', 'text_heavy', 'object_detection', 'ui_analysis')"
                }
            },
            "get_vision_summary": {
                "name": "get_vision_summary",
                "description": "Get a summary of recent visual activity and detections. Use this to understand what has been happening recently.",
                "parameters": {
                    "time_window": "string - Time window for summary ('recent', 'last_minute', 'last_5_minutes')"
                }
            },
            "check_vision_capabilities": {
                "name": "check_vision_capabilities",
                "description": "Check what vision capabilities are currently available and working.",
                "parameters": {}
            }
        }
    
    def get_tool_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """Get descriptions of all available vision tools."""
        return self.tool_descriptions
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tools formatted for LLM consumption."""
        tools = []
        for tool_name, tool_info in self.tool_descriptions.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool_info["parameters"],
                        "required": list(tool_info["parameters"].keys())
                    }
                }
            })
        return tools
    
    async def capture_screenshot(self, reason: str = "general_analysis") -> VisionToolResult:
        """
        Capture a screenshot from the OBS NDI stream.
        
        Args:
            reason: Why the screenshot is being captured
            
        Returns:
            VisionToolResult with base64 encoded image or error
        """
        try:
            # Check cooldown to prevent spam
            current_time = time.time()
            if current_time - self.last_screenshot_time < self.screenshot_cooldown:
                return VisionToolResult(
                    success=False,
                    data=None,
                    metadata={"reason": reason, "cooldown_active": True},
                    error_message=f"Screenshot cooldown active. Wait {self.screenshot_cooldown - (current_time - self.last_screenshot_time):.1f}s"
                )
            
            self.logger.info(f"[VisionTools] 📸 Capturing screenshot for reason: {reason}")
            
            # Try to get screenshot from vision integration service
            if hasattr(self.app_context, 'vision_integration_service') and self.app_context.vision_integration_service:
                vision_service = self.app_context.vision_integration_service
                if hasattr(vision_service, '_capture_current_screenshot'):
                    screenshot_b64 = vision_service._capture_current_screenshot()
                    if screenshot_b64:
                        self.last_screenshot_time = current_time
                        self.logger.info(f"[VisionTools] ✅ Screenshot captured successfully: {len(screenshot_b64)} chars")
                        return VisionToolResult(
                            success=True,
                            data=screenshot_b64,
                            metadata={
                                "reason": reason,
                                "source": "vision_integration_service",
                                "size_chars": len(screenshot_b64),
                                "timestamp": current_time
                            }
                        )
            
            # Fallback to direct NDI service access
            if hasattr(self.app_context, 'vision_pipeline') and self.app_context.vision_pipeline:
                vision_pipeline = self.app_context.vision_pipeline
                if hasattr(vision_pipeline, 'ndi_service') and vision_pipeline.ndi_service:
                    ndi_service = vision_pipeline.ndi_service
                    if hasattr(ndi_service, 'last_captured_frame') and ndi_service.last_captured_frame is not None:
                        # Convert frame to base64
                        frame = ndi_service.last_captured_frame
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if success:
                            screenshot_b64 = base64.b64encode(buffer).decode('utf-8')
                            self.last_screenshot_time = current_time
                            self.logger.info(f"[VisionTools] ✅ Screenshot captured from NDI service: {len(screenshot_b64)} chars")
                            return VisionToolResult(
                                success=True,
                                data=screenshot_b64,
                                metadata={
                                    "reason": reason,
                                    "source": "ndi_service",
                                    "size_chars": len(screenshot_b64),
                                    "timestamp": current_time
                                }
                            )
            
            # Final fallback to PIL screen capture
            try:
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                buffer = io.BytesIO()
                screenshot.save(buffer, format='JPEG', quality=85)
                screenshot_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                self.last_screenshot_time = current_time
                self.logger.info(f"[VisionTools] ✅ Screenshot captured via PIL fallback: {len(screenshot_b64)} chars")
                return VisionToolResult(
                    success=True,
                    data=screenshot_b64,
                    metadata={
                        "reason": reason,
                        "source": "pil_fallback",
                        "size_chars": len(screenshot_b64),
                        "timestamp": current_time
                    }
                )
            except Exception as pil_error:
                self.logger.error(f"[VisionTools] PIL fallback failed: {pil_error}")
            
            return VisionToolResult(
                success=False,
                data=None,
                metadata={"reason": reason},
                error_message="No screenshot capture method available"
            )
            
        except Exception as e:
            self.logger.error(f"[VisionTools] Screenshot capture error: {e}", exc_info=True)
            return VisionToolResult(
                success=False,
                data=None,
                metadata={"reason": reason},
                error_message=f"Screenshot capture failed: {str(e)}"
            )
    
    async def analyze_screenshot(self, focus_areas: List[str] = None, analysis_type: str = "general") -> VisionToolResult:
        """
        Analyze a screenshot using the VLM.
        
        Args:
            focus_areas: Specific areas to focus on
            analysis_type: Type of analysis to perform
            
        Returns:
            VisionToolResult with analysis results
        """
        try:
            self.logger.info(f"[VisionTools] 🔍 Analyzing screenshot - focus: {focus_areas}, type: {analysis_type}")
            
            # First capture a screenshot
            screenshot_result = await self.capture_screenshot(reason=f"analysis_{analysis_type}")
            if not screenshot_result.success:
                return screenshot_result
            
            # Create analysis prompt
            focus_text = ""
            if focus_areas:
                focus_text = f" Focus specifically on: {', '.join(focus_areas)}."
            
            analysis_prompts = {
                "general": f"Analyze this game screenshot and describe what you see happening.{focus_text} Include any important UI elements, player actions, or game state information.",
                "text_heavy": f"Carefully read and analyze all text in this screenshot.{focus_text} Pay special attention to chat messages, UI labels, notifications, and any readable text.",
                "object_detection": f"Identify and describe all objects, characters, and items visible in this screenshot.{focus_text} Focus on what objects are present and their locations.",
                "ui_analysis": f"Analyze the user interface elements in this screenshot.{focus_text} Describe menus, health bars, inventory, chat windows, and other UI components."
            }
            
            prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
            
            # Create VLM message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot_result.data}"}}
                    ]
                }
            ]
            
            # Get VLM response
            if hasattr(self.app_context, 'model_client') and self.app_context.model_client:
                response = await self.app_context.model_client.chat_completion(
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
                if response and hasattr(response, 'choices') and response.choices:
                    analysis = response.choices[0].message.content
                    self.logger.info(f"[VisionTools] ✅ Screenshot analysis completed: {len(analysis)} chars")
                    return VisionToolResult(
                        success=True,
                        data=analysis,
                        metadata={
                            "analysis_type": analysis_type,
                            "focus_areas": focus_areas,
                            "screenshot_source": screenshot_result.metadata.get("source"),
                            "analysis_length": len(analysis)
                        }
                    )
                else:
                    return VisionToolResult(
                        success=False,
                        data=None,
                        metadata={"analysis_type": analysis_type},
                        error_message="No response from VLM model"
                    )
            else:
                return VisionToolResult(
                    success=False,
                    data=None,
                    metadata={"analysis_type": analysis_type},
                    error_message="VLM model client not available"
                )
                
        except Exception as e:
            self.logger.error(f"[VisionTools] Screenshot analysis error: {e}", exc_info=True)
            return VisionToolResult(
                success=False,
                data=None,
                metadata={"analysis_type": analysis_type},
                error_message=f"Screenshot analysis failed: {str(e)}"
            )
    
    async def get_vision_summary(self, time_window: str = "recent") -> VisionToolResult:
        """
        Get a summary of recent visual activity.
        
        Args:
            time_window: Time window for summary
            
        Returns:
            VisionToolResult with summary data
        """
        try:
            self.logger.info(f"[VisionTools] 📊 Getting vision summary for window: {time_window}")
            
            if hasattr(self.app_context, 'vision_integration_service') and self.app_context.vision_integration_service:
                vision_service = self.app_context.vision_integration_service
                
                if hasattr(vision_service, 'get_recent_vision_summary'):
                    summary = vision_service.get_recent_vision_summary()
                    return VisionToolResult(
                        success=True,
                        data=summary,
                        metadata={
                            "time_window": time_window,
                            "summary_length": len(summary)
                        }
                    )
                elif hasattr(vision_service, 'get_vision_summary'):
                    summary = vision_service.get_vision_summary()
                    return VisionToolResult(
                        success=True,
                        data=summary,
                        metadata={
                            "time_window": time_window,
                            "summary_length": len(summary)
                        }
                    )
            
            return VisionToolResult(
                success=False,
                data=None,
                metadata={"time_window": time_window},
                error_message="Vision integration service not available"
            )
            
        except Exception as e:
            self.logger.error(f"[VisionTools] Vision summary error: {e}", exc_info=True)
            return VisionToolResult(
                success=False,
                data=None,
                metadata={"time_window": time_window},
                error_message=f"Vision summary failed: {str(e)}"
            )
    
    async def check_vision_capabilities(self) -> VisionToolResult:
        """
        Check what vision capabilities are currently available.
        
        Returns:
            VisionToolResult with capability status
        """
        try:
            self.logger.info("[VisionTools] 🔍 Checking vision capabilities")
            
            capabilities = {
                "ndi_service": False,
                "vision_pipeline": False,
                "vision_integration": False,
                "model_client": False,
                "screenshot_capture": False
            }
            
            # Check NDI service
            if hasattr(self.app_context, 'vision_pipeline') and self.app_context.vision_pipeline:
                vision_pipeline = self.app_context.vision_pipeline
                capabilities["vision_pipeline"] = True
                
                if hasattr(vision_pipeline, 'ndi_service') and vision_pipeline.ndi_service:
                    ndi_service = vision_pipeline.ndi_service
                    capabilities["ndi_service"] = getattr(ndi_service, 'is_initialized', False)
            
            # Check vision integration service
            if hasattr(self.app_context, 'vision_integration_service') and self.app_context.vision_integration_service:
                capabilities["vision_integration"] = True
            
            # Check model client
            if hasattr(self.app_context, 'model_client') and self.app_context.model_client:
                capabilities["model_client"] = True
            
            # Test screenshot capture
            test_result = await self.capture_screenshot(reason="capability_test")
            capabilities["screenshot_capture"] = test_result.success
            
            # Create summary
            working_capabilities = [k for k, v in capabilities.items() if v]
            summary = f"Vision capabilities: {'✅' if capabilities['screenshot_capture'] else '❌'} Screenshot capture, {'✅' if capabilities['model_client'] else '❌'} VLM analysis, {'✅' if capabilities['ndi_service'] else '❌'} NDI stream"
            
            return VisionToolResult(
                success=True,
                data=summary,
                metadata={
                    "capabilities": capabilities,
                    "working_count": len(working_capabilities),
                    "total_count": len(capabilities)
                }
            )
            
        except Exception as e:
            self.logger.error(f"[VisionTools] Capability check error: {e}", exc_info=True)
            return VisionToolResult(
                success=False,
                data=None,
                metadata={},
                error_message=f"Capability check failed: {str(e)}"
            )
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> VisionToolResult:
        """
        Execute a vision tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for the tool
            
        Returns:
            VisionToolResult with tool execution results
        """
        try:
            self.logger.info(f"[VisionTools] 🛠️ Executing tool: {tool_name} with parameters: {parameters}")
            
            if tool_name == "capture_screenshot":
                reason = parameters.get("reason", "general_analysis")
                return await self.capture_screenshot(reason=reason)
            
            elif tool_name == "analyze_screenshot":
                focus_areas = parameters.get("focus_areas", [])
                analysis_type = parameters.get("analysis_type", "general")
                return await self.analyze_screenshot(focus_areas=focus_areas, analysis_type=analysis_type)
            
            elif tool_name == "get_vision_summary":
                time_window = parameters.get("time_window", "recent")
                return await self.get_vision_summary(time_window=time_window)
            
            elif tool_name == "check_vision_capabilities":
                return await self.check_vision_capabilities()
            
            else:
                return VisionToolResult(
                    success=False,
                    data=None,
                    metadata={"tool_name": tool_name},
                    error_message=f"Unknown tool: {tool_name}"
                )
                
        except Exception as e:
            self.logger.error(f"[VisionTools] Tool execution error: {e}", exc_info=True)
            return VisionToolResult(
                success=False,
                data=None,
                metadata={"tool_name": tool_name},
                error_message=f"Tool execution failed: {str(e)}"
            ) 