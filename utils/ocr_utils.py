# utils/ocr_utils.py
import cv2
import numpy as np
import easyocr
import json
import os
from typing import Optional, List, Dict, Tuple

class OCRProcessor:
    def __init__(self, app_context):
        self.ctx = app_context
        self.logger = self.ctx.logger
        self.reader = None
        self.ocr_settings = {} # Will be loaded from game profile
        self.ocr_layout_rois: Dict[str, Dict] = {} # Loaded from ocr_layout_path

        self._initialize_ocr_engine()
        self._load_ocr_layout() # Load ROIs based on active profile

        self.ctx.active_profile_change_subscribers.append(self.on_profile_change) # If you add a subscriber list to AppContext
        self.logger.info("[OCRProcessor] Initialized.")

    def on_profile_change(self, new_profile):
        """Called when the game profile changes in AppContext."""
        self.logger.info("[OCRProcessor] Active profile changed. Re-initializing OCR settings and layout.")
        self._initialize_ocr_engine() # Re-init reader if languages/GPU change
        self._load_ocr_layout()       # Reload ROIs

    def _initialize_ocr_engine(self):
        profile_ocr_settings = self.ctx.active_profile.ocr_settings
        if not profile_ocr_settings or profile_ocr_settings.get("engine", "easyocr").lower() != "easyocr":
            self.logger.warning("[OCRProcessor] EasyOCR engine not configured in profile or disabled. OCR will not function.")
            self.reader = None
            self.ocr_settings = {}
            return

        self.ocr_settings = profile_ocr_settings
        languages = self.ocr_settings.get("easyocr_languages", ['en'])
        use_gpu = self.ocr_settings.get("easyocr_gpu", True)

        # Check if settings have changed to avoid re-initializing unnecessarily
        # This basic check might need to be more robust if reader instance is complex
        if self.reader and self.reader.lang_list == languages and self.reader.gpu == use_gpu:
             self.logger.info(f"[OCRProcessor] EasyOCR settings unchanged ({languages}, GPU: {use_gpu}). Reader not re-initialized.")
             return

        self.logger.info(f"[OCRProcessor] Initializing EasyOCR Reader. Languages: {languages}, GPU: {use_gpu}")
        try:
            if use_gpu:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self.logger.warning("[OCRProcessor] GPU requested, but CUDA not available. EasyOCR will fall back to CPU.")
                except ImportError:
                    self.logger.warning("[OCRProcessor] PyTorch not found. GPU support requires PyTorch. Falling back to CPU.")
            
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
            self.logger.info("[OCRProcessor] EasyOCR Reader initialized successfully.")
        except Exception as e:
            self.logger.error(f"[OCRProcessor] Error initializing EasyOCR Reader: {e}", exc_info=True)
            self.reader = None

    def _load_ocr_layout(self):
        self.ocr_layout_rois = {} 
        # Get path from profile, ensuring profile itself is loaded
        if not hasattr(self.ctx, 'active_profile') or self.ctx.active_profile is None:
            self.logger.error("[OCRProcessor] Cannot load OCR layout: AppContext.active_profile is not set!")
            return

        ocr_layout_path_from_profile = getattr(self.ctx.active_profile, "ocr_layout_path", None)
        self.logger.info(f"[OCRProcessor] Attempting to load OCR layout. Path from profile: '{ocr_layout_path_from_profile}'")

        if ocr_layout_path_from_profile:
            # Assume path is relative to project root if not absolute
            # Project root can be tricky to get consistently if script is run from different dirs
            # For now, assume it's relative to where DanzarVLM.py is
            # A better way is to pass project_root in AppContext or use absolute paths in config
            
            # Let's try resolving from current working directory as a common case
            resolved_path = os.path.abspath(ocr_layout_path_from_profile)
            self.logger.info(f"[OCRProcessor] Resolved OCR layout path to: '{resolved_path}'")

            if os.path.exists(resolved_path):
                try:
                    with open(resolved_path, "r") as f:
                        self.ocr_layout_rois = json.load(f)
                    self.logger.info(f"[OCRProcessor] Loaded OCR layout with {len(self.ocr_layout_rois)} ROIs from: {resolved_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"[OCRProcessor] Failed to decode JSON from OCR layout file {resolved_path}: {e}", exc_info=True)
                except Exception as e:
                    self.logger.error(f"[OCRProcessor] Failed to load OCR layout from {resolved_path}: {e}", exc_info=True)
            else:
                self.logger.warning(f"[OCRProcessor] OCR layout file NOT FOUND at resolved path: {resolved_path}")
                self.logger.warning("[OCRProcessor] Ensure ocr_layout_path in profile is correct and relative to project execution dir, or an absolute path.")
        else:
            self.logger.info("[OCRProcessor] No ocr_layout_path defined in active profile. Full image OCR will be attempted.")
        
        if not self.ocr_layout_rois:
             self.logger.warning("[OCRProcessor] CONFIRMED: ocr_layout_rois is EMPTY. This will result in slow full-image OCR for VLM commentary.")

    def _preprocess_roi(self, roi_image_np: np.ndarray, roi_config: Dict) -> np.ndarray:
        """Applies preprocessing to a single ROI image."""
        processed_img = roi_image_np.copy()

        # Get ROI-specific or default preprocessing params
        upscale = roi_config.get('upscale_factor', self.ocr_settings.get('default_upscale_factor', 1.0))
        grayscale = roi_config.get('grayscale', self.ocr_settings.get('default_grayscale', False))
        binarize = roi_config.get('binarize', self.ocr_settings.get('default_binarize', False))
        binarize_method = roi_config.get('binarize_method', self.ocr_settings.get('default_binarize_method', 'otsu'))

        if upscale > 1.0:
            width = int(processed_img.shape[1] * upscale)
            height = int(processed_img.shape[0] * upscale)
            processed_img = cv2.resize(processed_img, (width, height), interpolation=cv2.INTER_LANCZOS4)

        if grayscale or binarize:
            if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        
        if binarize:
            if binarize_method == 'otsu':
                _, processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif binarize_method == 'adaptive':
                processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, 11, 2)
            else: # Simple binary
                _, processed_img = cv2.threshold(processed_img, 127, 255, cv2.THRESH_BINARY)
        
        return processed_img

    def extract_text_from_image(self, full_image_np: np.ndarray) -> List[str]:
        """
        Extracts text from the image.
        If ROIs are defined in ocr_layout_rois, processes them.
        Otherwise, processes the full image using default settings.
        Returns a list of strings, like ["ROI_Name: text found", "Another_ROI: other text"].
        """
        if not self.reader:
            self.logger.warning("[OCRProcessor] EasyOCR reader not initialized. Cannot extract text.")
            return []

        extracted_texts: List[str] = []

        if self.ocr_layout_rois:
            self.logger.debug(f"[OCRProcessor] Processing {len(self.ocr_layout_rois)} ROIs.")
            for roi_name, roi_config in self.ocr_layout_rois.items():
                coords = roi_config.get('coords')
                if not coords or len(coords) != 4:
                    self.logger.warning(f"[OCRProcessor] Invalid or missing coords for ROI '{roi_name}'. Skipping.")
                    continue
                
                x, y, w, h = coords
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > full_image_np.shape[1] or y + h > full_image_np.shape[0]:
                    self.logger.warning(f"[OCRProcessor] ROI '{roi_name}' coords {coords} are out of bounds for image shape {full_image_np.shape[:2]}. Skipping.")
                    continue

                roi_img_np = full_image_np[y:y+h, x:x+w]
                if roi_img_np.size == 0:
                    self.logger.warning(f"[OCRProcessor] ROI '{roi_name}' crop resulted in an empty image. Skipping.")
                    continue
                
                preprocessed_roi_img = self._preprocess_roi(roi_img_np, roi_config)
                
                try:
                    results = self.reader.readtext(preprocessed_roi_img, detail=1, paragraph=False)
                    
                    roi_conf_thresh = roi_config.get('confidence_threshold', 
                                                     self.ocr_settings.get('default_confidence_threshold', 0.1))
                    
                    current_roi_texts = []
                    for (bbox, text, conf) in results:
                        if conf >= roi_conf_thresh:
                            current_roi_texts.append(text.strip())
                    
                    if current_roi_texts:
                        combined_roi_text = " ".join(current_roi_texts)
                        extracted_texts.append(f"{roi_name}: {combined_roi_text}")
                        self.logger.debug(f"[OCRProcessor] ROI '{roi_name}' Text (Conf>={roi_conf_thresh:.2f}): {combined_roi_text}")

                except Exception as e:
                    self.logger.error(f"[OCRProcessor] Error processing ROI '{roi_name}': {e}", exc_info=True)
        else:
            # Fallback: Process full image with default settings
            self.logger.debug("[OCRProcessor] No ROIs defined, processing full image with default settings.")
            full_image_config = {} # Use defaults from self.ocr_settings
            preprocessed_full_img = self._preprocess_roi(full_image_np, full_image_config)
            try:
                results = self.reader.readtext(preprocessed_full_img, detail=1, paragraph=False)
                default_conf_thresh = self.ocr_settings.get('default_confidence_threshold', 0.1)
                
                full_image_texts = []
                for (bbox, text, conf) in results:
                    if conf >= default_conf_thresh:
                        full_image_texts.append(text.strip())
                if full_image_texts:
                    extracted_texts.append("FullFrameOCR: " + " ".join(full_image_texts))
            except Exception as e:
                self.logger.error(f"[OCRProcessor] Error processing full image: {e}", exc_info=True)

        return extracted_texts