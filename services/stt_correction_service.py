# services/stt_correction_service.py
import re
import logging
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

class STTCorrectionService:
    """
    Service for correcting STT (Speech-to-Text) misspellings using both
    prompt-based and post-processing techniques.
    
    Based on research from OpenAI Cookbook and Mondegreen paper for
    handling ASR errors in voice queries.
    """
    
    def __init__(self, app_context):
        self.app_context = app_context
        self.logger = app_context.logger
        self.config = app_context.global_settings
        
        # Load game-specific terminology and common misspellings
        self.game_terminology = self._load_game_terminology()
        self.common_misspellings = self._load_common_misspellings()
        
        # STT correction settings
        self.correction_threshold = self.config.get('STT_CORRECTION_THRESHOLD', 0.7)
        self.max_corrections_per_text = self.config.get('STT_MAX_CORRECTIONS', 5)
        
        self.logger.info("[STTCorrectionService] Initialized with game terminology and misspelling corrections")
    
    def _load_game_terminology(self) -> Dict[str, List[str]]:
        """Load game-specific terminology from profiles."""
        terminology = {}
        
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
                            
                            # Extract terminology from profile
                            game_terms = []
                            
                            # Classes, items, locations, etc.
                            if 'classes' in profile_data:
                                game_terms.extend(profile_data['classes'])
                            if 'items' in profile_data:
                                game_terms.extend(profile_data['items'])
                            if 'locations' in profile_data:
                                game_terms.extend(profile_data['locations'])
                            if 'npcs' in profile_data:
                                game_terms.extend(profile_data['npcs'])
                            if 'spells' in profile_data:
                                game_terms.extend(profile_data['spells'])
                            
                            terminology[game_name.lower()] = game_terms
                            
                        except Exception as e:
                            self.logger.warning(f"[STTCorrectionService] Failed to load profile {profile_file}: {e}")
                            
        except Exception as e:
            self.logger.error(f"[STTCorrectionService] Error loading game terminology: {e}")
        
        return terminology
    
    def _load_common_misspellings(self) -> Dict[str, str]:
        """Load common STT misspellings and their corrections."""
        return {
            # Common STT errors
            'emberquest': 'everquest',
            'ever quest': 'everquest',
            'ever quests': 'everquest',
            'eq': 'everquest',
            'eq2': 'everquest 2',
            'eq1': 'everquest',
            'wow': 'world of warcraft',
            'warcraft': 'world of warcraft',
            'rim world': 'rimworld',
            'rim worlds': 'rimworld',
            
            # Common word misspellings
            'dont': "don't",
            'cant': "can't",
            'wont': "won't",
            'im': "I'm",
            'ive': "I've",
            'youre': "you're",
            'theyre': "they're",
            'were': "we're",
            'hes': "he's",
            'shes': "she's",
            'its': "it's",
            
            # Gaming terms
            'mob': 'mob',
            'npc': 'NPC',
            'hp': 'HP',
            'mp': 'MP',
            'exp': 'XP',
            'xp': 'XP',
            'lvl': 'level',
            'loot': 'loot',
            'aggro': 'aggro',
            'tank': 'tank',
            'heal': 'heal',
            'dps': 'DPS',
            'raid': 'raid',
            'dungeon': 'dungeon',
            'quest': 'quest',
            'guild': 'guild',
            'party': 'party',
            'group': 'group',
            'buff': 'buff',
            'debuff': 'debuff',
            'spell': 'spell',
            'ability': 'ability',
            'skill': 'skill',
            'class': 'class',
            'race': 'race',
            'character': 'character',
            'player': 'player',
            'game': 'game',
            'server': 'server',
            'zone': 'zone',
            'area': 'area',
            'town': 'town',
            'city': 'city',
            'village': 'village',
            'camp': 'camp',
            'outpost': 'outpost',
            'fortress': 'fortress',
            'castle': 'castle',
            'tower': 'tower',
            'cave': 'cave',
            'dungeon': 'dungeon',
            'temple': 'temple',
            'shrine': 'shrine',
            'altar': 'altar',
            'portal': 'portal',
            'gate': 'gate',
            'bridge': 'bridge',
            'road': 'road',
            'path': 'path',
            'trail': 'trail',
            'forest': 'forest',
            'mountain': 'mountain',
            'river': 'river',
            'lake': 'lake',
            'ocean': 'ocean',
            'desert': 'desert',
            'swamp': 'swamp',
            'jungle': 'jungle',
            'plains': 'plains',
            'hills': 'hills',
            'valley': 'valley',
            'canyon': 'canyon',
            'volcano': 'volcano',
            'island': 'island',
            'beach': 'beach',
            'cliff': 'cliff',
            'waterfall': 'waterfall',
            'spring': 'spring',
            'well': 'well',
            'fountain': 'fountain',
            'pond': 'pond',
            'stream': 'stream',
            'creek': 'creek',
            'brook': 'brook',
            'bay': 'bay',
            'gulf': 'gulf',
            'sea': 'sea',
            'coast': 'coast',
            'shore': 'shore',
            'harbor': 'harbor',
            'port': 'port',
            'dock': 'dock',
            'pier': 'pier',
            'wharf': 'wharf',
            'lighthouse': 'lighthouse',
            'buoy': 'buoy',
            'anchor': 'anchor',
            'ship': 'ship',
            'boat': 'boat',
            'raft': 'raft',
            'canoe': 'canoe',
            'kayak': 'kayak',
            'yacht': 'yacht',
            'ferry': 'ferry',
            'submarine': 'submarine',
            'aircraft': 'aircraft',
            'plane': 'plane',
            'helicopter': 'helicopter',
            'balloon': 'balloon',
            'airship': 'airship',
            'zeppelin': 'zeppelin',
            'glider': 'glider',
            'parachute': 'parachute',
            'rocket': 'rocket',
            'missile': 'missile',
            'torpedo': 'torpedo',
            'bomb': 'bomb',
            'explosive': 'explosive',
            'grenade': 'grenade',
            'mine': 'mine',
            'trap': 'trap',
            'snare': 'snare',
            'pit': 'pit',
            'spike': 'spike',
            'arrow': 'arrow',
            'bolt': 'bolt',
            'bullet': 'bullet',
            'shell': 'shell',
            'cannon': 'cannon',
            'catapult': 'catapult',
            'trebuchet': 'trebuchet',
            'ballista': 'ballista',
            'crossbow': 'crossbow',
            'bow': 'bow',
            'sword': 'sword',
            'axe': 'axe',
            'hammer': 'hammer',
            'mace': 'mace',
            'spear': 'spear',
            'dagger': 'dagger',
            'knife': 'knife',
            'staff': 'staff',
            'wand': 'wand',
            'orb': 'orb',
            'crystal': 'crystal',
            'gem': 'gem',
            'jewel': 'jewel',
            'ring': 'ring',
            'necklace': 'necklace',
            'bracelet': 'bracelet',
            'earring': 'earring',
            'crown': 'crown',
            'tiara': 'tiara',
            'helmet': 'helmet',
            'armor': 'armor',
            'shield': 'shield',
            'boots': 'boots',
            'gloves': 'gloves',
            'belt': 'belt',
            'cloak': 'cloak',
            'robe': 'robe',
            'tunic': 'tunic',
            'vest': 'vest',
            'jacket': 'jacket',
            'coat': 'coat',
            'pants': 'pants',
            'leggings': 'leggings',
            'greaves': 'greaves',
            'gauntlets': 'gauntlets',
            'bracers': 'bracers',
            'pauldrons': 'pauldrons',
            'cuirass': 'cuirass',
            'breastplate': 'breastplate',
            'chainmail': 'chainmail',
            'platemail': 'platemail',
            'leather': 'leather',
            'cloth': 'cloth',
            'silk': 'silk',
            'wool': 'wool',
            'cotton': 'cotton',
            'linen': 'linen',
            'velvet': 'velvet',
            'satin': 'satin',
            'lace': 'lace',
            'fur': 'fur',
            'hide': 'hide',
            'scale': 'scale',
            'bone': 'bone',
            'ivory': 'ivory',
            'ebony': 'ebony',
            'oak': 'oak',
            'pine': 'pine',
            'maple': 'maple',
            'birch': 'birch',
            'willow': 'willow',
            'elm': 'elm',
            'ash': 'ash',
            'yew': 'yew',
            'mahogany': 'mahogany',
            'teak': 'teak',
            'cedar': 'cedar',
            'cypress': 'cypress',
            'juniper': 'juniper',
            'fir': 'fir',
            'spruce': 'spruce',
            'hemlock': 'hemlock',
            'larch': 'larch',
            'poplar': 'poplar',
            'alder': 'alder',
            'hazel': 'hazel',
            'chestnut': 'chestnut',
            'walnut': 'walnut',
            'hickory': 'hickory',
            'beech': 'beech',
            'sycamore': 'sycamore',
            'plane': 'plane',
            'linden': 'linden',
            'basswood': 'basswood',
            'cottonwood': 'cottonwood',
            'aspen': 'aspen',
            'cherry': 'cherry',
            'plum': 'plum',
            'peach': 'peach',
            'apricot': 'apricot',
            'nectarine': 'nectarine',
            'apple': 'apple',
            'pear': 'pear',
            'quince': 'quince',
            'medlar': 'medlar',
            'loquat': 'loquat',
            'persimmon': 'persimmon',
            'mulberry': 'mulberry',
            'fig': 'fig',
            'pomegranate': 'pomegranate',
            'guava': 'guava',
            'mango': 'mango',
            'papaya': 'papaya',
            'banana': 'banana',
            'plantain': 'plantain',
            'coconut': 'coconut',
            'date': 'date',
            'prune': 'prune',
            'raisin': 'raisin',
            'currant': 'currant',
            'gooseberry': 'gooseberry',
            'elderberry': 'elderberry',
            'blackberry': 'blackberry',
            'raspberry': 'raspberry',
            'strawberry': 'strawberry',
            'blueberry': 'blueberry',
            'cranberry': 'cranberry',
            'lingonberry': 'lingonberry',
            'cloudberry': 'cloudberry',
            'salmonberry': 'salmonberry',
            'thimbleberry': 'thimbleberry',
            'wineberry': 'wineberry',
            'dewberry': 'dewberry',
            'boysenberry': 'boysenberry',
            'loganberry': 'loganberry',
            'tayberry': 'tayberry',
            'marionberry': 'marionberry',
            'olallieberry': 'olallieberry',
            'santiam': 'santiam',
            'chehalem': 'chehalem',
            'kotata': 'kotata',
            'blackcap': 'blackcap',
            'jostaberry': 'jostaberry',
        }
    
    def correct_transcription(self, text: str, game_context: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Correct STT transcription using multiple techniques.
        
        Args:
            text: Raw transcription text
            game_context: Optional game context for targeted corrections
            
        Returns:
            Tuple of (corrected_text, correction_log)
        """
        original_text = text
        corrections = []
        
        # Step 1: Basic text normalization
        text = self._normalize_text(text)
        
        # Step 2: Apply common misspelling corrections
        text, basic_corrections = self._apply_common_corrections(text)
        corrections.extend(basic_corrections)
        
        # Step 3: Game-specific terminology correction
        if game_context:
            text, game_corrections = self._apply_game_corrections(text, game_context)
            corrections.extend(game_corrections)
        
        # Step 4: Fuzzy matching for similar words
        text, fuzzy_corrections = self._apply_fuzzy_corrections(text)
        corrections.extend(fuzzy_corrections)
        
        # Step 5: Context-aware corrections
        text, context_corrections = self._apply_context_corrections(text)
        corrections.extend(context_corrections)
        
        if corrections:
            self.logger.info(f"[STTCorrectionService] Applied {len(corrections)} corrections to transcription")
            for correction in corrections:
                self.logger.debug(f"[STTCorrectionService] Correction: '{correction['original']}' -> '{correction['corrected']}' (confidence: {correction['confidence']:.2f})")
        
        return text, corrections
    
    def _normalize_text(self, text: str) -> str:
        """Basic text normalization."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,!?])', r'\1', text)
        text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
        
        return text
    
    def _apply_common_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply common misspelling corrections."""
        corrections = []
        corrected_text = text
        
        # Check each word against common misspellings
        words = corrected_text.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check exact matches first
            if word_lower in self.common_misspellings:
                correction = self.common_misspellings[word_lower]
                
                # Preserve original case if it was capitalized
                if word[0].isupper():
                    correction = correction.capitalize()
                
                words[i] = correction
                corrections.append({
                    'type': 'common_misspelling',
                    'original': word,
                    'corrected': correction,
                    'confidence': 0.95,
                    'position': i
                })
        
        return ' '.join(words), corrections
    
    def _apply_game_corrections(self, text: str, game_context: str) -> Tuple[str, List[Dict]]:
        """Apply game-specific terminology corrections."""
        corrections = []
        corrected_text = text
        
        game_terms = self.game_terminology.get(game_context.lower(), [])
        if not game_terms:
            return corrected_text, corrections
        
        words = corrected_text.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check against game terminology
            for term in game_terms:
                term_lower = term.lower()
                
                # Exact match
                if word_lower == term_lower:
                    # Preserve original case
                    if word[0].isupper():
                        words[i] = term.capitalize()
                    else:
                        words[i] = term
                    
                    corrections.append({
                        'type': 'game_terminology',
                        'original': word,
                        'corrected': words[i],
                        'confidence': 0.9,
                        'position': i,
                        'game_context': game_context
                    })
                    break
                
                # Fuzzy match for similar terms
                similarity = SequenceMatcher(None, word_lower, term_lower).ratio()
                if similarity > self.correction_threshold:
                    # Preserve original case
                    if word[0].isupper():
                        words[i] = term.capitalize()
                    else:
                        words[i] = term
                    
                    corrections.append({
                        'type': 'game_terminology_fuzzy',
                        'original': word,
                        'corrected': words[i],
                        'confidence': similarity,
                        'position': i,
                        'game_context': game_context
                    })
                    break
        
        return ' '.join(words), corrections
    
    def _apply_fuzzy_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply fuzzy matching corrections for similar words."""
        corrections = []
        corrected_text = text
        
        words = corrected_text.split()
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Skip short words and already corrected words
            if len(word) < 3:
                continue
            
            best_match = None
            best_similarity = 0
            
            # Check against all known terms
            all_terms = list(self.common_misspellings.keys())
            for term in all_terms:
                similarity = SequenceMatcher(None, word_lower, term).ratio()
                if similarity > best_similarity and similarity > self.correction_threshold:
                    best_similarity = similarity
                    best_match = term
            
            if best_match:
                # Preserve original case
                if word[0].isupper():
                    words[i] = best_match.capitalize()
                else:
                    words[i] = best_match
                
                corrections.append({
                    'type': 'fuzzy_match',
                    'original': word,
                    'corrected': words[i],
                    'confidence': best_similarity,
                    'position': i
                })
        
        return ' '.join(words), corrections
    
    def _apply_context_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        """Apply context-aware corrections."""
        corrections = []
        corrected_text = text
        
        # Fix common context issues
        # "is not a class that's in" -> "is not a class in"
        corrected_text = re.sub(r"is not a class that's in", "is not a class in", corrected_text)
        corrected_text = re.sub(r"is not a class thats in", "is not a class in", corrected_text)
        
        # "that's in" -> "in"
        corrected_text = re.sub(r"that's in", "in", corrected_text)
        corrected_text = re.sub(r"thats in", "in", corrected_text)
        
        # "there's no" -> "there is no"
        corrected_text = re.sub(r"there's no", "there is no", corrected_text)
        corrected_text = re.sub(r"theres no", "there is no", corrected_text)
        
        # "doesn't have" -> "does not have"
        corrected_text = re.sub(r"doesn't have", "does not have", corrected_text)
        corrected_text = re.sub(r"doesnt have", "does not have", corrected_text)
        
        if corrected_text != text:
            corrections.append({
                'type': 'context_correction',
                'original': text,
                'corrected': corrected_text,
                'confidence': 0.8,
                'position': -1
            })
        
        return corrected_text, corrections
    
    def get_correction_prompt(self, game_context: Optional[str] = None) -> str:
        """
        Generate a prompt for STT correction that can be used with Whisper.
        
        Based on OpenAI Cookbook technique of providing correct spellings
        in the prompt parameter.
        """
        prompt_terms = []
        
        # Add common gaming terms
        prompt_terms.extend([
            "EverQuest", "EmberQuest", "World of Warcraft", "RimWorld",
            "class", "race", "character", "player", "game", "server",
            "zone", "area", "town", "city", "village", "camp", "outpost",
            "fortress", "castle", "tower", "cave", "dungeon", "temple",
            "shrine", "altar", "portal", "gate", "bridge", "road", "path",
            "trail", "forest", "mountain", "river", "lake", "ocean",
            "desert", "swamp", "jungle", "plains", "hills", "valley",
            "canyon", "volcano", "island", "beach", "cliff", "waterfall",
            "spring", "well", "fountain", "pond", "stream", "creek",
            "brook", "bay", "gulf", "sea", "coast", "shore", "harbor",
            "port", "dock", "pier", "wharf", "lighthouse", "buoy",
            "anchor", "ship", "boat", "raft", "canoe", "kayak", "yacht",
            "ferry", "submarine", "aircraft", "plane", "helicopter",
            "balloon", "airship", "zeppelin", "glider", "parachute",
            "rocket", "missile", "torpedo", "bomb", "explosive",
            "grenade", "mine", "trap", "snare", "pit", "spike", "arrow",
            "bolt", "bullet", "shell", "cannon", "catapult", "trebuchet",
            "ballista", "crossbow", "bow", "sword", "axe", "hammer",
            "mace", "spear", "dagger", "knife", "staff", "wand", "orb",
            "crystal", "gem", "jewel", "ring", "necklace", "bracelet",
            "earring", "crown", "tiara", "helmet", "armor", "shield",
            "boots", "gloves", "belt", "cloak", "robe", "tunic", "vest",
            "jacket", "coat", "pants", "leggings", "greaves", "gauntlets",
            "bracers", "pauldrons", "cuirass", "breastplate", "chainmail",
            "platemail", "leather", "cloth", "silk", "wool", "cotton",
            "linen", "velvet", "satin", "lace", "fur", "hide", "scale",
            "bone", "ivory", "ebony", "oak", "pine", "maple", "birch",
            "willow", "elm", "ash", "yew", "mahogany", "teak", "cedar",
            "cypress", "juniper", "fir", "spruce", "hemlock", "larch",
            "poplar", "alder", "hazel", "chestnut", "walnut", "hickory",
            "beech", "sycamore", "plane", "linden", "basswood", "cottonwood",
            "aspen", "cherry", "plum", "peach", "apricot", "nectarine",
            "apple", "pear", "quince", "medlar", "loquat", "persimmon",
            "mulberry", "fig", "pomegranate", "guava", "mango", "papaya",
            "banana", "plantain", "coconut", "date", "prune", "raisin",
            "currant", "gooseberry", "elderberry", "blackberry", "raspberry",
            "strawberry", "blueberry", "cranberry", "lingonberry", "cloudberry",
            "salmonberry", "thimbleberry", "wineberry", "dewberry", "boysenberry",
            "loganberry", "tayberry", "marionberry", "olallieberry", "santiam",
            "chehalem", "kotata", "blackcap", "jostaberry"
        ])
        
        # Add game-specific terms
        if game_context and game_context.lower() in self.game_terminology:
            game_terms = self.game_terminology[game_context.lower()]
            prompt_terms.extend(game_terms)
        
        return ", ".join(prompt_terms) 