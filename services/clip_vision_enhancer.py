# services/clip_vision_enhancer.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import time

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    try:
        import clip
        CLIP_AVAILABLE = True
    except ImportError:
        CLIP_AVAILABLE = False
        clip = None
        open_clip = None

class CLIPVisionEnhancer:
    """
    CLIP-based vision enhancer for the vision-aware conversation system.
    Provides semantic understanding of visual elements using CLIP's language-vision alignment.
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Get optimal device from GPU memory manager
        self.device = self._get_optimal_device()
        
        # CLIP model and preprocessing
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
        # Game-specific visual concepts
        self.game_concepts = {
            'everquest': [
                "health bar", "mana bar", "inventory window", "spell book", "chat window",
                "character portrait", "experience bar", "compass", "map", "group window",
                "combat log", "target window", "hotbar", "spell effects", "monster",
                "player character", "npc", "quest window", "trade window", "guild window"
            ],
            'generic_game': [
                "health bar", "inventory", "menu", "button", "text", "character",
                "enemy", "weapon", "armor", "potion", "gold", "experience", "level",
                "skill tree", "map", "minimap", "chat", "notification", "loading screen"
            ],
            'rimworld': [
                "colonist", "room", "bed", "table", "chair", "door", "wall", "floor",
                "crop", "animal", "tool", "weapon", "medicine", "food", "storage",
                "workbench", "power generator", "battery", "wire", "research bench"
            ]
        }
        
        # Initialize CLIP
        self._initialize_clip()
        
    def _get_optimal_device(self) -> str:
        """Get the optimal device for CLIP processing using GPU memory manager."""
        try:
            # Check if GPU memory manager is available
            if hasattr(self.app_context, 'gpu_memory_manager') and self.app_context.gpu_memory_manager:
                device, reason = self.app_context.gpu_memory_manager.get_best_vision_device()
                self.logger.info(f"[CLIPVisionEnhancer] GPU Memory Manager selected device: {device} - {reason}")
                return device
            else:
                # Fallback to configuration
                vision_config = self.config.get('vision_config', {})
                clip_config = vision_config.get('clip', {})
                device = clip_config.get('device', 'cuda:1')  # Default to cuda:1 to avoid main LLM
                self.logger.info(f"[CLIPVisionEnhancer] Using configured device: {device}")
                return device
        except Exception as e:
            self.logger.warning(f"[CLIPVisionEnhancer] Error getting optimal device: {e}")
            # Safe fallback to avoid main LLM
            return "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
    
    def _initialize_clip(self):
        """Initialize CLIP model and preprocessing."""
        if not CLIP_AVAILABLE:
            self.logger.warning("[CLIPVisionEnhancer] CLIP not available. Install with: pip install open_clip_torch")
            return False
            
        try:
            self.logger.info("[CLIPVisionEnhancer] Loading CLIP model...")
            
            # Try open_clip first (newer, better maintained)
            if open_clip:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="openai", device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
                self.logger.info(f"[CLIPVisionEnhancer] OpenCLIP loaded successfully on {self.device}")
            else:
                # Fallback to original CLIP
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.logger.info(f"[CLIPVisionEnhancer] Original CLIP loaded successfully on {self.device}")
            
            return True
        except Exception as e:
            self.logger.error(f"[CLIPVisionEnhancer] Failed to load CLIP: {e}")
            return False
    
    def enhance_visual_context(self, frame: np.ndarray, detected_objects: List[Dict], 
                             ocr_results: List[str], game_type: str = "generic_game") -> Dict[str, Any]:
        """
        Enhance visual context using CLIP's semantic understanding.
        
        Args:
            frame: Current video frame
            detected_objects: YOLO detections
            ocr_results: OCR text results
            game_type: Type of game for concept filtering
            
        Returns:
            Enhanced visual context with CLIP insights
        """
        if not self.model or not self.preprocess:
            return self._fallback_context(detected_objects, ocr_results)
        
        try:
            # Convert frame to PIL Image for CLIP
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get game-specific concepts
            concepts = self.game_concepts.get(game_type, self.game_concepts['generic_game'])
            
            # Add OCR text as concepts
            if ocr_results:
                concepts.extend([f"text saying {text}" for text in ocr_results[:3]])
            
            # Add detected objects as concepts
            if detected_objects:
                for obj in detected_objects:
                    label = obj.get('label', '')
                    if label and label not in concepts:
                        concepts.append(label)
            
            # Prepare CLIP inputs
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            text_inputs = self.tokenizer(concepts).to(self.device)
            
            # Get CLIP embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get top matches
            top_k = min(10, len(concepts))
            top_probs, top_indices = similarities[0].topk(top_k)
            
            # Build enhanced context
            enhanced_context = {
                'clip_insights': [],
                'semantic_understanding': {},
                'confidence_scores': {},
                'visual_descriptions': []
            }
            
            # Check if CLIP logging is enabled
            enable_logging = self.config.get('clip_enable_logging', False)
            log_insights = self.config.get('clip_log_insights', False)
            
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                concept = concepts[idx]
                confidence = prob.item()
                
                if confidence > 0.1:  # Only include confident matches
                    enhanced_context['clip_insights'].append({
                        'concept': concept,
                        'confidence': confidence,
                        'description': self._generate_concept_description(concept, confidence)
                    })
                    
                    enhanced_context['confidence_scores'][concept] = confidence
                    
                    # Generate natural language description
                    if confidence > 0.3:
                        enhanced_context['visual_descriptions'].append(
                            self._generate_concept_description(concept, confidence)
                        )
            
            # Log CLIP insights if enabled
            if enable_logging and enhanced_context['clip_insights']:
                if log_insights:
                    # Log all insights
                    insight_text = ", ".join([f"{insight['concept']} ({insight['confidence']:.2f})" for insight in enhanced_context['clip_insights']])
                    self.logger.info(f"[CLIPVisionEnhancer] Detected: {insight_text}")
                else:
                    # Log top insight only
                    top_insight = enhanced_context['clip_insights'][0]
                    self.logger.info(f"[CLIPVisionEnhancer] Top detection: {top_insight['concept']} (conf: {top_insight['confidence']:.2f})")
            
            # Add semantic understanding
            enhanced_context['semantic_understanding'] = self._analyze_semantic_context(
                enhanced_context['clip_insights'], detected_objects, ocr_results
            )
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"[CLIPVisionEnhancer] Error enhancing visual context: {e}")
            return self._fallback_context(detected_objects, ocr_results)
    
    def _generate_concept_description(self, concept: str, confidence: float) -> str:
        """Generate natural language description of a visual concept."""
        confidence_level = "very clearly" if confidence > 0.7 else "clearly" if confidence > 0.5 else "somewhat"
        
        if "health" in concept.lower():
            return f"I can {confidence_level} see a health indicator"
        elif "inventory" in concept.lower():
            return f"I can {confidence_level} see an inventory or item management interface"
        elif "chat" in concept.lower():
            return f"I can {confidence_level} see chat or communication elements"
        elif "text" in concept.lower():
            return f"I can {confidence_level} read some text on screen"
        elif "bar" in concept.lower():
            return f"I can {confidence_level} see a progress or status bar"
        else:
            return f"I can {confidence_level} see {concept}"
    
    def _analyze_semantic_context(self, clip_insights: List[Dict], 
                                detected_objects: List[Dict], 
                                ocr_results: List[str]) -> Dict[str, Any]:
        """Analyze semantic context from CLIP insights and other vision data."""
        analysis = {
            'game_state': 'unknown',
            'player_status': 'unknown',
            'interface_elements': [],
            'notable_objects': [],
            'text_content': []
        }
        
        # Analyze game state
        health_indicators = [insight for insight in clip_insights if 'health' in insight['concept'].lower()]
        if health_indicators:
            analysis['game_state'] = 'in_game'
            analysis['player_status'] = 'active'
        
        # Collect interface elements
        for insight in clip_insights:
            if insight['confidence'] > 0.3:
                if any(keyword in insight['concept'].lower() for keyword in ['bar', 'window', 'menu', 'button']):
                    analysis['interface_elements'].append(insight['concept'])
                elif any(keyword in insight['concept'].lower() for keyword in ['character', 'enemy', 'monster', 'npc']):
                    analysis['notable_objects'].append(insight['concept'])
        
        # Add OCR text
        if ocr_results:
            analysis['text_content'] = ocr_results[:5]  # Limit to 5 most recent
        
        return analysis
    
    def _fallback_context(self, detected_objects: List[Dict], ocr_results: List[str]) -> Dict[str, Any]:
        """Fallback context when CLIP is not available."""
        return {
            'clip_insights': [],
            'semantic_understanding': {
                'game_state': 'unknown',
                'player_status': 'unknown',
                'interface_elements': [obj['label'] for obj in detected_objects if obj.get('label')],
                'notable_objects': [],
                'text_content': ocr_results[:5]
            },
            'confidence_scores': {},
            'visual_descriptions': []
        }
    
    def query_visual_concept(self, frame: np.ndarray, query: str) -> Dict[str, Any]:
        """
        Query CLIP about a specific visual concept in the frame.
        
        Args:
            frame: Current video frame
            query: Natural language query about what to look for
            
        Returns:
            CLIP's response about the query
        """
        if not self.model or not self.preprocess:
            return {'confidence': 0.0, 'description': 'CLIP not available'}
        
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Prepare inputs
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            text_input = self.tokenizer([query]).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # Normalize and calculate similarity
                image_features = F.normalize(image_features, p=2, dim=1)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                confidence = similarity[0][0].item()
            
            return {
                'confidence': confidence,
                'description': f"CLIP is {confidence:.1%} confident that '{query}' is present in the image",
                'query': query
            }
            
        except Exception as e:
            self.logger.error(f"[CLIPVisionEnhancer] Error querying concept: {e}")
            return {'confidence': 0.0, 'description': f'Error querying: {str(e)}'}
    
    def get_status(self) -> Dict[str, Any]:
        """Get CLIP vision enhancer status."""
        return {
            'clip_available': CLIP_AVAILABLE and self.model is not None,
            'device': self.device,
            'model_loaded': self.model is not None,
            'preprocess_loaded': self.preprocess is not None
        }
    
    def cleanup(self):
        """Cleanup CLIP resources."""
        if self.model:
            del self.model
        if self.preprocess:
            del self.preprocess
        torch.cuda.empty_cache() if torch.cuda.is_available() else None 

    async def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a frame with CLIP to understand visual content.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dict containing CLIP insights and confidence scores
        """
        if not self.model or not self.preprocess:
            self.logger.warning("[CLIPVisionEnhancer] CLIP model not initialized")
            return {}
        
        try:
            # Check if we should process this frame based on frequency
            current_time = time.time()
            if current_time - self.last_processing_time < self.processing_interval:
                return {}
            
            self.last_processing_time = current_time
            
            # Preprocess frame for CLIP
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Resize if needed
            if frame_rgb.shape[0] > 224 or frame_rgb.shape[1] > 224:
                frame_rgb = cv2.resize(frame_rgb, (224, 224))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Preprocess for CLIP
            inputs = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Get game concepts based on current profile
            game_profile = self.config.get('current_game_profile', 'generic_game')
            concepts = self.game_concepts.get(game_profile, self.game_concepts['generic_game'])
            
            # Encode text concepts
            text_inputs = self.tokenizer(concepts).to(self.device)
            
            # Get CLIP predictions
            with torch.no_grad():
                image_features = self.model.encode_image(inputs)
                text_features = self.model.encode_text(text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity scores
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get top insights
            confidence_threshold = self.config.get('clip_confidence_threshold', 0.3)
            max_insights = self.config.get('clip_max_insights', 5)
            
            # Convert to numpy for easier processing
            scores = similarity.cpu().numpy()[0]
            
            # Get top matches
            top_indices = np.argsort(scores)[::-1][:max_insights]
            insights = []
            
            for idx in top_indices:
                confidence = float(scores[idx])
                if confidence >= confidence_threshold:
                    concept = concepts[idx]
                    insights.append({
                        'concept': concept,
                        'confidence': confidence
                    })
            
            # Log CLIP insights if enabled
            enable_logging = self.config.get('clip_enable_logging', False)
            log_insights = self.config.get('clip_log_insights', False)
            
            if enable_logging and insights:
                if log_insights:
                    # Log all insights
                    insight_text = ", ".join([f"{insight['concept']} ({insight['confidence']:.2f})" for insight in insights])
                    self.logger.info(f"[CLIPVisionEnhancer] Detected: {insight_text}")
                else:
                    # Log summary
                    top_insight = insights[0]
                    self.logger.info(f"[CLIPVisionEnhancer] Top detection: {top_insight['concept']} (conf: {top_insight['confidence']:.2f})")
            
            # Update processing frequency
            self.processing_count += 1
            if self.processing_count % 10 == 0:
                self.logger.info(f"[CLIPVisionEnhancer] Processed {self.processing_count} frames, last insights: {len(insights)} concepts detected")
            
            return {
                'insights': insights,
                'frame_shape': frame.shape,
                'processing_time': time.time() - current_time,
                'game_profile': game_profile
            }
            
        except Exception as e:
            self.logger.error(f"[CLIPVisionEnhancer] Error processing frame: {e}")
            return {} 