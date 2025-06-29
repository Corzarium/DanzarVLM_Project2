"""
OCR Filter Service for DanzarAI
Improves OCR text quality by filtering out garbage text
"""

import re
import logging
from typing import List, Optional, Tuple

class OCRFilterService:
    """Service for filtering and improving OCR text quality"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.garbage_patterns = [
            re.compile(r'^[^a-zA-Z0-9]*$'),  # No alphanumeric characters
            re.compile(r'^[0-9]{1,2}$'),     # Just 1-2 digits
            re.compile(r'^[a-zA-Z]{1,2}$'),  # Just 1-2 letters
            re.compile(r'^[^a-zA-Z]*[a-zA-Z][^a-zA-Z]*$'),  # Single letter surrounded by non-letters
            re.compile(r'^[^0-9]*[0-9][^0-9]*$'),  # Single digit surrounded by non-digits
            re.compile(r'^[\W_]+$'),        # Only special characters
            re.compile(r'^[\s\W_]+$'),     # Only whitespace and special characters
        ]
        
        self.valid_patterns = [
            re.compile(r'[a-zA-Z]{3,}'),     # At least 3 consecutive letters
            re.compile(r'[0-9]{2,}'),        # At least 2 consecutive digits
            re.compile(r'[a-zA-Z][0-9]'),    # Letter followed by digit
            re.compile(r'[0-9][a-zA-Z]'),    # Digit followed by letter
        ]
    
    def filter_text(self, text: str, min_confidence: float = 0.7, 
                   min_length: int = 2, max_length: int = 200) -> Optional[str]:
        """
        Filter OCR text to remove garbage and improve quality
        
        Args:
            text: Raw OCR text
            min_confidence: Minimum confidence threshold
            min_length: Minimum text length
            max_length: Maximum text length
            
        Returns:
            Filtered text or None if filtered out
        """
        if not text or not text.strip():
            return None
        
        # Clean the text
        cleaned_text = text.strip()
        
        # Check length constraints
        if len(cleaned_text) < min_length or len(cleaned_text) > max_length:
            self.logger.debug(f"OCR text filtered by length: {cleaned_text}")
            return None
        
        # Check for garbage patterns
        for pattern in self.garbage_patterns:
            if pattern.match(cleaned_text):
                self.logger.debug(f"OCR text filtered by garbage pattern: {cleaned_text}")
                return None
        
        # Check for valid patterns (at least one must match)
        has_valid_pattern = False
        for pattern in self.valid_patterns:
            if pattern.search(cleaned_text):
                has_valid_pattern = True
                break
        
        if not has_valid_pattern:
            self.logger.debug(f"OCR text filtered - no valid patterns: {cleaned_text}")
            return None
        
        # Additional quality checks
        if self._is_likely_garbage(cleaned_text):
            self.logger.debug(f"OCR text filtered by quality check: {cleaned_text}")
            return None
        
        self.logger.debug(f"OCR text passed filtering: {cleaned_text}")
        return cleaned_text
    
    def _is_likely_garbage(self, text: str) -> bool:
        """Additional heuristic checks for garbage text"""
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text)
        if special_char_ratio > 0.5:
            return True
        
        # Check for repeated characters
        for char in text:
            if text.count(char) > len(text) * 0.7:  # More than 70% same character
                return True
        
        # Check for alternating patterns
        if len(text) > 4:
            alternating = all(text[i] != text[i+1] for i in range(len(text)-1))
            if alternating and len(set(text)) <= 2:
                return True
        
        return False
    
    def batch_filter(self, texts: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Filter a batch of OCR texts with their confidence scores
        
        Args:
            texts: List of (text, confidence) tuples
            
        Returns:
            Filtered list of (text, confidence) tuples
        """
        filtered_texts = []
        
        for text, confidence in texts:
            if confidence >= 0.7:  # Only process high-confidence text
                filtered_text = self.filter_text(text)
                if filtered_text:
                    filtered_texts.append((filtered_text, confidence))
        
        return filtered_texts
