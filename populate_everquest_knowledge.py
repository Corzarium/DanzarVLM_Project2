#!/usr/bin/env python3
"""
Populate the RAG database with accurate EverQuest information
This will prevent the system from making up information about EverQuest classes
Source: https://everquest.allakhazam.com/wiki/EverQuest
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config_loader import load_global_settings, load_game_profile
from services.qdrant_service import QdrantService
import logging

def setup_logging():
    """Setup basic logging for the population script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("EQKnowledgePopulator")

def get_everquest_class_knowledge():
    """Returns accurate EverQuest class information from allakhazam.com"""
    return [
        {
            "category": "EverQuest Classes Overview",
            "content": """EverQuest has 16 distinct character classes, each with unique abilities, roles, and playstyles. 
            The classes are divided into four main archetypes: Melee (Warrior, Paladin, Shadow Knight, Ranger, Monk, Rogue, Beastlord, Berserker), 
            Priest (Cleric, Druid, Shaman), Caster (Wizard, Magician, Necromancer, Enchanter), and Hybrid classes that combine elements.""",
            "metadata": {"source": "allakhazam.com", "type": "class_overview", "game": "everquest"}
        },
        {
            "category": "EverQuest Warrior Class",
            "content": """Warrior: The primary tank class in EverQuest. Warriors have the highest hit points, best armor class, 
            and access to the widest variety of weapons and armor. They excel at taking damage and holding aggro from enemies. 
            Warriors can use all weapon types and have abilities like Taunt, Bash, Kick, and later Rampage and Defensive abilities.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "warrior", "game": "everquest"}
        },
        {
            "category": "EverQuest Paladin Class", 
            "content": """Paladin: A hybrid tank/healer class. Paladins are holy warriors who can tank effectively while providing 
            healing and beneficial spells. They have good hit points and armor, can use most weapons, and gain healing spells, 
            buffs, and undead-specific abilities like Turn Undead and Harm Touch (reverse). They can also Lay on Hands to heal.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "paladin", "game": "everquest"}
        },
        {
            "category": "EverQuest Shadow Knight Class",
            "content": """Shadow Knight: An evil hybrid tank/caster class. Shadow Knights combine melee combat with necromantic magic. 
            They can tank well, have good hit points and armor, and cast necromancer spells including damage over time, 
            fear, and undead summoning. They have a Harm Touch ability and can lifetap enemies to heal themselves.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "shadow knight", "game": "everquest"}
        },
        {
            "category": "EverQuest Ranger Class",
            "content": """Ranger: A hybrid melee/caster class specializing in archery and nature magic. Rangers are excellent at 
            ranged combat with bows, can track enemies, and cast druid spells including healing, buffs, and transportation. 
            They can dual wield weapons and are effective in outdoor environments with abilities like Foraging and tracking.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "ranger", "game": "everquest"}
        },
        {
            "category": "EverQuest Monk Class",
            "content": """Monk: A melee class that fights with martial arts techniques. Monks use hand-to-hand combat or blunt weapons, 
            have high agility and speed, and can feign death to avoid combat. They excel at pulling enemies and have abilities 
            like Dodge, Block, and various combat stances. Monks can also mend their wounds and have good movement speed.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "monk", "game": "everquest"}
        },
        {
            "category": "EverQuest Rogue Class",
            "content": """Rogue: A melee DPS class specializing in stealth and backstab attacks. Rogues can hide, sneak, pick locks, 
            disarm traps, and deal massive damage from behind enemies. They can dual wield piercing weapons effectively and 
            have abilities like Backstab, Sneak Attack, and later Assassinate. Rogues are the primary lockpicking class.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "rogue", "game": "everquest"}
        },
        {
            "category": "EverQuest Beastlord Class",
            "content": """Beastlord: A hybrid melee/caster class that fights alongside an animal companion. Beastlords can summon 
            and control various animal pets, cast shaman spells, and provide group support. They use hand-to-hand combat 
            and have good solo capabilities due to their pet. Added in the Shadows of Luclin expansion.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "beastlord", "game": "everquest"}
        },
        {
            "category": "EverQuest Berserker Class",
            "content": """Berserker: A melee DPS class focused on two-handed weapon combat and berserking abilities. Berserkers 
            can enter berserk mode for increased damage, have high hit points, and specialize in axes and two-handed weapons. 
            They have abilities like Frenzy and various berserker-specific combat techniques. Added in Gates of Discord expansion.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "berserker", "game": "everquest"}
        },
        {
            "category": "EverQuest Cleric Class",
            "content": """Cleric: The primary healing class in EverQuest. Clerics have the most powerful healing spells, 
            resurrection abilities, and can provide various beneficial buffs. They can turn undead, have good hit points 
            for a caster, and are essential for group healing and support. Clerics can also provide damage shields and protection spells.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "cleric", "game": "everquest"}
        },
        {
            "category": "EverQuest Druid Class",
            "content": """Druid: A priest class specializing in nature magic and outdoor abilities. Druids can heal, provide buffs, 
            cast damage over time spells, and have excellent transportation abilities like teleportation. They can track, 
            forage for food and reagents, and have both healing and damage capabilities. Druids are popular for their utility.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "druid", "game": "everquest"}
        },
        {
            "category": "EverQuest Shaman Class",
            "content": """Shaman: A priest class focusing on buffs, debuffs, and spiritual magic. Shamans provide powerful stat 
            buffs (like Haste), can slow enemies, heal effectively, and have unique abilities like Spirit of Wolf for movement speed. 
            They can also provide mana regeneration and have both beneficial and harmful magic capabilities.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "shaman", "game": "everquest"}
        },
        {
            "category": "EverQuest Wizard Class",
            "content": """Wizard: A pure caster class specializing in direct damage magic. Wizards have the highest burst damage 
            spells in the game, including powerful nukes and area effect spells. They can teleport themselves and groups 
            to various locations and have abilities like Gate to return to their bind point. Wizards are the premier DD casters.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "wizard", "game": "everquest"}
        },
        {
            "category": "EverQuest Magician Class",
            "content": """Magician: A caster class that specializes in summoning elementals and conjuring items. Magicians can 
            summon various elemental pets (air, earth, fire, water), conjure food, water, and magical items. They have 
            good direct damage spells and their pets can tank or provide DPS. Magicians are excellent soloers due to their pets.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "magician", "game": "everquest"}
        },
        {
            "category": "EverQuest Necromancer Class",
            "content": """Necromancer: A caster class specializing in death magic and damage over time spells. Necromancers can 
            summon undead pets, cast powerful DoT spells, fear enemies, and lifetap to convert damage to health. They have 
            excellent mana efficiency through lifetapping and can feign death. Necromancers are powerful soloers.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "necromancer", "game": "everquest"}
        },
        {
            "category": "EverQuest Enchanter Class",
            "content": """Enchanter: A caster class specializing in crowd control and mental magic. Enchanters can charm enemies 
            to fight for them, mesmerize multiple enemies, provide powerful buffs like Haste and mana regeneration, and 
            have illusion spells. They are masters of crowd control and can make charmed pets much more powerful than normal.""",
            "metadata": {"source": "allakhazam.com", "type": "class_detail", "class": "enchanter", "game": "everquest"}
        },
        {
            "category": "EverQuest Class Roles Summary",
            "content": """EverQuest class roles: Tanks (Warrior, Paladin, Shadow Knight) absorb damage and control enemies. 
            Healers (Cleric, Druid, Shaman) keep the group alive and provide support. DPS classes deal damage through 
            melee (Rogue, Monk, Ranger, Berserker, Beastlord) or magic (Wizard, Magician, Necromancer). 
            Support classes (Enchanter, Bard) provide crowd control and utility. All classes have expanded significantly through expansions.""",
            "metadata": {"source": "allakhazam.com", "type": "class_roles", "game": "everquest"}
        }
    ]

def main():
    logger = setup_logging()
    logger.info("Starting EverQuest knowledge population...")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        global_settings = load_global_settings()
        
        # Initialize Qdrant client
        qdrant_config = global_settings.get('QDRANT_SERVER', {})
        logger.info(f"Connecting to Qdrant at {qdrant_config.get('host', '127.0.0.1')}:{qdrant_config.get('port', 6333)}")
        
        qdrant = QdrantService(
            host=qdrant_config.get('host', '127.0.0.1'),
            port=qdrant_config.get('port', 6333),
            api_key=qdrant_config.get('api_key')
        )
        
        # Test connection
        try:
            collections = qdrant.client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}. Make sure Qdrant is running.")
            return False
            
        collection_name = qdrant_config.get('default_collection', 'danzar_knowledge')
        logger.info(f"Using collection: {collection_name}")
        
        # Get EverQuest knowledge
        eq_knowledge = get_everquest_class_knowledge()
        logger.info(f"Preparing to add {len(eq_knowledge)} EverQuest knowledge entries")
        
        # Initialize sentence transformer for embeddings
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        success_count = 0
        for i, knowledge in enumerate(eq_knowledge, 1):
            try:
                logger.info(f"Processing entry {i}/{len(eq_knowledge)}: {knowledge['category']}")
                
                # Generate embeddings
                text = f"{knowledge['category']}: {knowledge['content']}"
                embeddings = embedding_model.encode([text])
                
                # Convert to list format
                if hasattr(embeddings, 'tolist'):
                    embeddings = embeddings.tolist()
                
                # Add to Qdrant
                result = qdrant.add_texts(
                    collection_name=collection_name,
                    texts=[text],
                    vectors=embeddings,
                    metadatas=[knowledge['metadata']]
                )
                
                if result:
                    success_count += 1
                    logger.info(f"✓ Successfully added: {knowledge['category']}")
                else:
                    logger.error(f"✗ Failed to add: {knowledge['category']}")
                    
            except Exception as e:
                logger.error(f"Error processing {knowledge['category']}: {e}")
        
        logger.info(f"Knowledge population complete! Successfully added {success_count}/{len(eq_knowledge)} entries")
        
        # Test query
        logger.info("Testing knowledge retrieval...")
        test_query = "What classes are in EverQuest?"
        query_vector = embedding_model.encode(test_query)
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()
            
        results = qdrant.query(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3
        )
        
        logger.info(f"Test query returned {len(results)} results:")
        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. Score: {result.get('score', 'N/A'):.3f} - {result.get('text', 'No text')[:100]}...")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to populate knowledge: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 