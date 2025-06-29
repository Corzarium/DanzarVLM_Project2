# services/enhanced_conditional_fact_checker.py
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
import asyncio
from urllib.parse import quote_plus

class EnhancedConditionalFactChecker:
    """
    Enhanced conditional fact checker that aggressively detects when the system
    is making things up, especially about game content, classes, and features.
    
    Based on research from OpenAI Cookbook and Mondegreen paper for handling
    ASR errors and fact verification.
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Load game-specific knowledge
        self.game_knowledge = self._load_game_knowledge()
        
        # Fact checking patterns
        self.challenge_patterns = self._load_challenge_patterns()
        self.uncertainty_patterns = self._load_uncertainty_patterns()
        self.fabrication_patterns = self._load_fabrication_patterns()
        
        # Confidence thresholds
        self.challenge_threshold = self.config.get('FACT_CHECK_CHALLENGE_THRESHOLD', 0.7)
        self.uncertainty_threshold = self.config.get('FACT_CHECK_UNCERTAINTY_THRESHOLD', 0.6)
        self.fabrication_threshold = self.config.get('FACT_CHECK_FABRICATION_THRESHOLD', 0.8)
        
        self.logger.info("[EnhancedConditionalFactChecker] Initialized with enhanced detection patterns")
    
    def _load_game_knowledge(self) -> Dict[str, Dict]:
        """Load game-specific knowledge from profiles and research."""
        knowledge = {}
        
        try:
            # Load from game profiles
            profiles_dir = self.config.get('PROFILES_DIR', 'config/profiles')
            import os
            import yaml
            
            if os.path.exists(profiles_dir):
                for profile_file in os.listdir(profiles_dir):
                    if profile_file.endswith('.yaml'):
                        profile_path = os.path.join(profiles_dir, profile_file)
                        try:
                            with open(profile_path, 'r', encoding='utf-8') as f:
                                profile_data = yaml.safe_load(f)
                                
                            game_name = profile_data.get('game_name', profile_file.replace('.yaml', ''))
                            knowledge[game_name.lower()] = {
                                'classes': profile_data.get('classes', []),
                                'races': profile_data.get('races', []),
                                'items': profile_data.get('items', []),
                                'locations': profile_data.get('locations', []),
                                'npcs': profile_data.get('npcs', []),
                                'spells': profile_data.get('spells', []),
                                'features': profile_data.get('features', []),
                                'mechanics': profile_data.get('mechanics', [])
                            }
                            
                        except Exception as e:
                            self.logger.warning(f"[EnhancedConditionalFactChecker] Failed to load profile {profile_file}: {e}")
                            
        except Exception as e:
            self.logger.error(f"[EnhancedConditionalFactChecker] Error loading game knowledge: {e}")
        
        return knowledge
    
    def _load_challenge_patterns(self) -> List[re.Pattern]:
        """Load patterns that indicate user is challenging information."""
        patterns = [
            # Direct challenges
            re.compile(r'\b(that\'s|thats)\s+(wrong|incorrect|not right|false|not true)\b', re.IGNORECASE),
            re.compile(r'\b(you\'re|youre)\s+(wrong|incorrect|mistaken|lying|making this up)\b', re.IGNORECASE),
            re.compile(r'\b(that\'s|thats)\s+(not|isn\'t|isnt)\s+(right|correct|true|accurate)\b', re.IGNORECASE),
            re.compile(r'\b(no|nope|nah)\s+(that\'s|thats)\s+(not|isn\'t|isnt)\b', re.IGNORECASE),
            re.compile(r'\b(actually|in fact|as a matter of fact)\b', re.IGNORECASE),
            re.compile(r'\b(you\'ve|youve)\s+(got|have)\s+(it|that)\s+(wrong|incorrect)\b', re.IGNORECASE),
            re.compile(r'\b(that\'s|thats)\s+(not|isn\'t|isnt)\s+(how|what|when|where|why)\b', re.IGNORECASE),
            
            # Questioning patterns
            re.compile(r'\b(are you sure|are you certain|is that right|is that correct)\b', re.IGNORECASE),
            re.compile(r'\b(how do you know|where did you get that|what makes you think)\b', re.IGNORECASE),
            re.compile(r'\b(that doesn\'t|that doesnt)\s+(sound|seem)\s+(right|correct|accurate)\b', re.IGNORECASE),
            re.compile(r'\b(i don\'t|i dont)\s+(think|believe)\s+(that\'s|thats)\s+(right|correct)\b', re.IGNORECASE),
            
            # Correction attempts
            re.compile(r'\b(no|nope|nah)\s+(it\'s|its)\s+(actually|really)\b', re.IGNORECASE),
            re.compile(r'\b(but|however|though)\s+(that\'s|thats)\s+(not|isn\'t|isnt)\b', re.IGNORECASE),
            re.compile(r'\b(you said|you mentioned|you claimed)\s+(but|however)\b', re.IGNORECASE),
            
            # Specific game challenges
            re.compile(r'\b(that\'s|thats)\s+(not|isn\'t|isnt)\s+(a class|a race|an item|a spell)\b', re.IGNORECASE),
            re.compile(r'\b(there\'s|theres)\s+(no|not)\s+(such thing as|class called|race called)\b', re.IGNORECASE),
            re.compile(r'\b(that class|that race|that item|that spell)\s+(doesn\'t|doesnt)\s+(exist)\b', re.IGNORECASE),
        ]
        return patterns
    
    def _load_uncertainty_patterns(self) -> List[re.Pattern]:
        """Load patterns that indicate uncertainty or lack of confidence."""
        patterns = [
            # Uncertainty indicators
            re.compile(r'\b(i think|i believe|i guess|i suppose|maybe|perhaps|possibly)\b', re.IGNORECASE),
            re.compile(r'\b(as far as i know|to my knowledge|from what i remember)\b', re.IGNORECASE),
            re.compile(r'\b(i\'m not sure|im not sure|i don\'t know|i dont know)\b', re.IGNORECASE),
            re.compile(r'\b(i could be wrong|i might be wrong|i may be mistaken)\b', re.IGNORECASE),
            re.compile(r'\b(if i remember correctly|if memory serves|if i recall)\b', re.IGNORECASE),
            
            # Vague language
            re.compile(r'\b(some|several|many|various|different|various)\s+(classes|races|items|spells)\b', re.IGNORECASE),
            re.compile(r'\b(things like|stuff like|such as|for example)\b', re.IGNORECASE),
            re.compile(r'\b(and so on|etc|et cetera|and others|and more)\b', re.IGNORECASE),
            
            # Generic descriptions
            re.compile(r'\b(a class|a race|an item|a spell|a feature)\s+(that|which)\b', re.IGNORECASE),
            re.compile(r'\b(something like|similar to|kind of like)\b', re.IGNORECASE),
        ]
        return patterns
    
    def _load_fabrication_patterns(self) -> List[re.Pattern]:
        """Load patterns that indicate potential fabrication."""
        patterns = [
            # Specific game content claims
            re.compile(r'\b(class|race|item|spell|ability|skill)\s+(called|named)\s+["\']?([^"\']+)["\']?\b', re.IGNORECASE),
            re.compile(r'\b(there\'s|theres)\s+(a|an)\s+(class|race|item|spell)\s+(called|named)\b', re.IGNORECASE),
            re.compile(r'\b(you can|players can|you have)\s+(a|an)\s+(class|race|item|spell)\s+(called|named)\b', re.IGNORECASE),
            
            # Specific mechanics claims
            re.compile(r'\b(the|this)\s+(class|race|item|spell)\s+(can|has|does|provides|gives)\b', re.IGNORECASE),
            re.compile(r'\b(ability|skill|power|feature)\s+(that|which)\s+(allows|enables|lets)\b', re.IGNORECASE),
            
            # Specific numbers and stats
            re.compile(r'\b(level|damage|health|mana|stamina)\s+(\d+)\b', re.IGNORECASE),
            re.compile(r'\b(\d+)\s+(damage|health|mana|stamina|points)\b', re.IGNORECASE),
            
            # Specific locations and NPCs
            re.compile(r'\b(location|area|zone|dungeon|city|town)\s+(called|named)\s+["\']?([^"\']+)["\']?\b', re.IGNORECASE),
            re.compile(r'\b(npc|character|boss|enemy)\s+(called|named)\s+["\']?([^"\']+)["\']?\b', re.IGNORECASE),
        ]
        return patterns
    
    def should_fact_check(self, user_input: str, bot_response: str, game_context: Optional[str] = None) -> Tuple[bool, str, float]:
        """
        Determine if fact checking should be triggered.
        
        Args:
            user_input: User's input text
            bot_response: Bot's response text
            game_context: Optional game context
            
        Returns:
            Tuple of (should_check, reason, confidence)
        """
        # Check for user challenges
        challenge_score = self._detect_challenge(user_input)
        if challenge_score > self.challenge_threshold:
            return True, "User challenge detected", challenge_score
        
        # Check for uncertainty in bot response
        uncertainty_score = self._detect_uncertainty(bot_response)
        if uncertainty_score > self.uncertainty_threshold:
            return True, "Uncertainty detected in response", uncertainty_score
        
        # Check for potential fabrication
        fabrication_score = self._detect_fabrication(bot_response, game_context)
        if fabrication_score > self.fabrication_threshold:
            return True, "Potential fabrication detected", fabrication_score
        
        # Check for specific game content claims
        game_claim_score = self._detect_game_claims(bot_response, game_context)
        if game_claim_score > self.fabrication_threshold:
            return True, "Specific game content claim detected", game_claim_score
        
        return False, "No fact check needed", 0.0
    
    def _detect_challenge(self, text: str) -> float:
        """Detect if user is challenging information."""
        score = 0.0
        matches = 0
        
        for pattern in self.challenge_patterns:
            if pattern.search(text):
                matches += 1
                score += 0.3  # Each challenge pattern adds 0.3 to score
        
        # Normalize score
        if matches > 0:
            score = min(score, 1.0)
        
        return score
    
    def _detect_uncertainty(self, text: str) -> float:
        """Detect uncertainty indicators in bot response."""
        score = 0.0
        matches = 0
        
        for pattern in self.uncertainty_patterns:
            if pattern.search(text):
                matches += 1
                score += 0.2  # Each uncertainty pattern adds 0.2 to score
        
        # Normalize score
        if matches > 0:
            score = min(score, 1.0)
        
        return score
    
    def _detect_fabrication(self, text: str, game_context: Optional[str] = None) -> float:
        """Detect potential fabrication in bot response."""
        score = 0.0
        matches = 0
        
        # Check fabrication patterns
        for pattern in self.fabrication_patterns:
            if pattern.search(text):
                matches += 1
                score += 0.25  # Each fabrication pattern adds 0.25 to score
        
        # Check against known game knowledge
        if game_context and game_context.lower() in self.game_knowledge:
            game_knowledge = self.game_knowledge[game_context.lower()]
            score += self._check_against_game_knowledge(text, game_knowledge)
        
        # Normalize score
        if matches > 0:
            score = min(score, 1.0)
        
        return score
    
    def _detect_game_claims(self, text: str, game_context: Optional[str] = None) -> float:
        """Detect specific game content claims that should be verified."""
        score = 0.0
        
        # Extract potential game content mentions
        content_mentions = []
        
        # Look for class/race/item/spell mentions
        class_pattern = re.compile(r'\b(class|race|item|spell|ability|skill)\s+(called|named)\s+["\']?([^"\']+)["\']?\b', re.IGNORECASE)
        for match in class_pattern.finditer(text):
            content_type = match.group(1)
            content_name = match.group(3)
            content_mentions.append((content_type, content_name))
        
        # Check if mentioned content exists in game knowledge
        if game_context and game_context.lower() in self.game_knowledge:
            game_knowledge = self.game_knowledge[game_context.lower()]
            
            for content_type, content_name in content_mentions:
                if content_type.lower() in game_knowledge:
                    known_content = game_knowledge[content_type.lower()]
                    if content_name.lower() not in [item.lower() for item in known_content]:
                        score += 0.4  # Unknown content mentioned
        
        return min(score, 1.0)
    
    def _check_against_game_knowledge(self, text: str, game_knowledge: Dict) -> float:
        """Check text against known game knowledge."""
        score = 0.0
        
        # Check each category of game knowledge
        for category, known_items in game_knowledge.items():
            if not known_items:
                continue
            
            # Look for mentions of items in this category
            for item in known_items:
                item_pattern = re.compile(rf'\b{re.escape(item)}\b', re.IGNORECASE)
                if item_pattern.search(text):
                    # Known item mentioned - this is good, reduce fabrication score
                    score -= 0.1
                else:
                    # Check for similar but incorrect names
                    for known_item in known_items:
                        if known_item != item:
                            similarity = self._calculate_similarity(item, known_item)
                            if similarity > 0.8:  # Very similar names
                                score += 0.2  # Potential confusion
        
        return max(score, 0.0)  # Don't go negative
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def generate_search_query(self, user_input: str, bot_response: str, game_context: Optional[str] = None) -> str:
        """Generate a search query for fact checking."""
        # Extract key terms from bot response
        key_terms = []
        
        # Extract game content mentions
        content_pattern = re.compile(r'\b(class|race|item|spell|ability|skill)\s+(called|named)\s+["\']?([^"\']+)["\']?\b', re.IGNORECASE)
        for match in content_pattern.finditer(bot_response):
            content_type = match.group(1)
            content_name = match.group(3)
            key_terms.append(f"{content_name} {content_type}")
        
        # Extract game names
        game_pattern = re.compile(r'\b(everquest|emberquest|world of warcraft|wow|rimworld)\b', re.IGNORECASE)
        for match in game_pattern.finditer(bot_response):
            key_terms.append(match.group(1))
        
        # Extract specific mechanics or features
        mechanic_pattern = re.compile(r'\b(ability|skill|power|feature)\s+(that|which)\s+(allows|enables|lets)\s+([^.!?]+)', re.IGNORECASE)
        for match in mechanic_pattern.finditer(bot_response):
            mechanic_desc = match.group(4).strip()
            key_terms.append(mechanic_desc)
        
        # Combine with game context
        if game_context:
            search_query = f"{game_context} {' '.join(key_terms)}"
        else:
            search_query = ' '.join(key_terms)
        
        # Clean up the query
        search_query = re.sub(r'\s+', ' ', search_query.strip())
        
        if not search_query:
            # Fallback to general game search
            search_query = f"{game_context or 'gaming'} classes races items"
        
        return search_query
    
    async def fact_check_response(self, user_input: str, bot_response: str, game_context: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Perform fact checking on bot response.
        
        Args:
            user_input: User's input text
            bot_response: Bot's response text
            game_context: Optional game context
            
        Returns:
            Tuple of (corrected_response, fact_check_results)
        """
        should_check, reason, confidence = self.should_fact_check(user_input, bot_response, game_context)
        
        if not should_check:
            return bot_response, {
                'fact_checked': False,
                'reason': reason,
                'confidence': confidence
            }
        
        self.logger.info(f"[EnhancedConditionalFactChecker] Fact checking triggered: {reason} (confidence: {confidence:.2f})")
        
        # Generate search query
        search_query = self.generate_search_query(user_input, bot_response, game_context)
        
        # Perform web search
        try:
            from utils.web_search import search_web
            search_results = await search_web(search_query, max_results=5)
            
            # Analyze search results
            fact_check_results = self._analyze_search_results(bot_response, search_results, game_context)
            
            # Generate corrected response
            corrected_response = self._generate_corrected_response(bot_response, fact_check_results)
            
            return corrected_response, {
                'fact_checked': True,
                'reason': reason,
                'confidence': confidence,
                'search_query': search_query,
                'search_results': search_results,
                'fact_check_results': fact_check_results,
                'corrections_made': len(fact_check_results.get('corrections', []))
            }
            
        except Exception as e:
            self.logger.error(f"[EnhancedConditionalFactChecker] Error during fact checking: {e}")
            return bot_response, {
                'fact_checked': False,
                'reason': f"Error during fact checking: {e}",
                'confidence': confidence
            }
    
    def _analyze_search_results(self, bot_response: str, search_results: List[Dict], game_context: Optional[str] = None) -> Dict:
        """Analyze search results for fact checking."""
        corrections = []
        verified_claims = []
        disputed_claims = []
        
        # Extract claims from bot response
        claims = self._extract_claims(bot_response)
        
        for claim in claims:
            claim_verified = False
            claim_disputed = False
            
            # Check each search result
            for result in search_results:
                result_text = result.get('snippet', '') + ' ' + result.get('title', '')
                
                # Check if claim is supported
                if self._claim_supported(claim, result_text):
                    claim_verified = True
                    verified_claims.append({
                        'claim': claim,
                        'supporting_result': result
                    })
                
                # Check if claim is disputed
                if self._claim_disputed(claim, result_text):
                    claim_disputed = True
                    disputed_claims.append({
                        'claim': claim,
                        'disputing_result': result
                    })
            
            # Generate correction if claim is disputed or unverified
            if claim_disputed:
                corrections.append({
                    'type': 'disputed_claim',
                    'original': claim,
                    'correction': f"I apologize, but I cannot verify that {claim}. Let me check the current information.",
                    'confidence': 0.8
                })
            elif not claim_verified and self._is_specific_claim(claim):
                corrections.append({
                    'type': 'unverified_claim',
                    'original': claim,
                    'correction': f"I should verify this information about {claim} before making specific claims.",
                    'confidence': 0.6
                })
        
        return {
            'corrections': corrections,
            'verified_claims': verified_claims,
            'disputed_claims': disputed_claims
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract specific claims from text."""
        claims = []
        
        # Extract game content claims
        content_pattern = re.compile(r'\b(class|race|item|spell|ability|skill)\s+(called|named)\s+["\']?([^"\']+)["\']?\b', re.IGNORECASE)
        for match in content_pattern.finditer(text):
            content_type = match.group(1)
            content_name = match.group(3)
            claims.append(f"{content_name} is a {content_type}")
        
        # Extract mechanic claims
        mechanic_pattern = re.compile(r'\b(the|this)\s+(class|race|item|spell)\s+(can|has|does|provides|gives)\s+([^.!?]+)', re.IGNORECASE)
        for match in mechanic_pattern.finditer(text):
            subject = match.group(2)
            action = match.group(3)
            description = match.group(4).strip()
            claims.append(f"{subject} {action} {description}")
        
        return claims
    
    def _claim_supported(self, claim: str, result_text: str) -> bool:
        """Check if a claim is supported by search result."""
        # Simple keyword matching for now
        claim_keywords = claim.lower().split()
        result_keywords = result_text.lower().split()
        
        # Check if most claim keywords appear in result
        matches = sum(1 for keyword in claim_keywords if keyword in result_keywords)
        return matches >= len(claim_keywords) * 0.7
    
    def _claim_disputed(self, claim: str, result_text: str) -> bool:
        """Check if a claim is disputed by search result."""
        dispute_indicators = [
            'not exist', 'does not exist', 'doesn\'t exist', 'doesnt exist',
            'no such', 'not a', 'is not', 'are not', 'was not', 'were not',
            'never existed', 'never was', 'never were', 'fake', 'false',
            'incorrect', 'wrong', 'mistake', 'error', 'myth', 'rumor'
        ]
        
        result_lower = result_text.lower()
        return any(indicator in result_lower for indicator in dispute_indicators)
    
    def _is_specific_claim(self, claim: str) -> bool:
        """Check if a claim is specific enough to warrant verification."""
        specific_indicators = [
            'called', 'named', 'specifically', 'exactly', 'precisely',
            'level', 'damage', 'health', 'mana', 'stamina', 'points'
        ]
        
        claim_lower = claim.lower()
        return any(indicator in claim_lower for indicator in specific_indicators)
    
    def _generate_corrected_response(self, original_response: str, fact_check_results: Dict) -> str:
        """Generate a corrected response based on fact check results."""
        corrections = fact_check_results.get('corrections', [])
        
        if not corrections:
            return original_response
        
        # Add correction prefix
        correction_prefix = "I need to correct some information: "
        
        # Combine all corrections
        correction_text = " ".join([correction['correction'] for correction in corrections])
        
        # Create corrected response
        corrected_response = f"{correction_prefix}{correction_text}\n\n{original_response}"
        
        return corrected_response 