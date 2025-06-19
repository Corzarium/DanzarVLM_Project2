#!/usr/bin/env python3
"""
Simple Knowledge Base Setup for DanzarVLM
Creates a Chroma vector database with basic EverQuest information
"""

import os
import sys
from pathlib import Path

def setup_basic_everquest_knowledge():
    """Set up a basic EverQuest knowledge base"""
    
    try:
        # Try to import required packages
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.docstore.document import Document
        
        print("✅ All required packages available")
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("\nTo install required packages, run:")
        print("pip install langchain chromadb sentence-transformers")
        return False
    
    # Create knowledge base directory
    kb_dir = Path("data/knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic EverQuest knowledge
    everquest_knowledge = [
        {
            "content": """EverQuest is a 3D fantasy-themed massively multiplayer online role-playing game (MMORPG) 
            originally developed by Verant Interactive and 989 Studios for Sony Online Entertainment. 
            Players move their character throughout the medieval fantasy world of Norrath, often fighting 
            monsters and enemies for treasure and experience points, and optionally mastering trade skills.""",
            "metadata": {"source": "everquest_overview", "category": "general"}
        },
        {
            "content": """EverQuest Classes: Warrior (tank), Cleric (healer), Paladin (tank/healer hybrid), 
            Ranger (melee/archery hybrid), Shadow Knight (tank/necromancer hybrid), Druid (healer/caster hybrid), 
            Monk (melee DPS), Bard (support/utility), Rogue (melee DPS), Wizard (pure caster DPS), 
            Magician (pet class/caster), Enchanter (crowd control/support), Necromancer (pet class/DoT caster), 
            Shaman (healer/support hybrid), Berserker (melee DPS), Beastlord (pet class/melee hybrid).""",
            "metadata": {"source": "everquest_classes", "category": "classes"}
        },
        {
            "content": """EverQuest Races: Human (balanced stats), Barbarian (high strength/stamina), 
            Dark Elf (high intelligence/dexterity), Dwarf (high stamina/constitution), 
            Erudite (highest intelligence), Gnome (high intelligence), Half Elf (balanced), 
            Halfling (high dexterity/agility), High Elf (high intelligence/wisdom), 
            Iksar (lizardman, regeneration), Ogre (highest strength), Troll (high strength/stamina, regeneration), 
            Wood Elf (high dexterity), Froglok (amphibious), Vah Shir (cat people).""",
            "metadata": {"source": "everquest_races", "category": "races"}
        },
        {
            "content": """EverQuest Zones: Qeynos (human starting city), Freeport (human starting city), 
            Kelethin (wood elf tree city), Kaladim (dwarf underground city), Ak'Anon (gnome clockwork city), 
            Oggok (ogre city), Grobb (troll city), Neriak (dark elf city), Erudin (erudite city), 
            Rivervale (halfling village), Halas (barbarian city), Cabilis (iksar city), 
            Shar Vahl (vah shir city). Popular hunting zones include Crushbone, Blackburrow, 
            Unrest, Mistmoore Castle, Sebilis, Chardok, and Plane of Fear.""",
            "metadata": {"source": "everquest_zones", "category": "zones"}
        },
        {
            "content": """EverQuest Gameplay: Players gain experience points (XP) by killing monsters, 
            completing quests, and discovering new areas. Death results in experience loss and 
            corpse retrieval. Groups of up to 6 players can adventure together. 
            Guilds provide social structure and raid opportunities. Player vs Player (PvP) 
            servers offer competitive gameplay. Trade skills include Smithing, Tailoring, 
            Pottery, Brewing, Baking, Fletching, and Jewelry Making.""",
            "metadata": {"source": "everquest_gameplay", "category": "gameplay"}
        }
    ]
    
    print("📚 Creating knowledge base documents...")
    
    # Create documents
    documents = []
    for item in everquest_knowledge:
        doc = Document(
            page_content=item["content"],
            metadata=item["metadata"]
        )
        documents.append(doc)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"📄 Created {len(split_docs)} document chunks")
    
    # Create embeddings (using a lightweight model)
    print("🔧 Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Use CPU for compatibility
    )
    
    # Create vector store
    print("💾 Creating Chroma vector database...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=str(kb_dir / "chroma_db"),
        collection_name="everquest_knowledge"
    )
    
    # Persist the database
    vectorstore.persist()
    
    print(f"✅ Knowledge base created successfully!")
    print(f"📁 Location: {kb_dir / 'chroma_db'}")
    print(f"📊 Documents: {len(documents)} original, {len(split_docs)} chunks")
    
    # Test the knowledge base
    print("\n🧪 Testing knowledge base...")
    results = vectorstore.similarity_search("What classes are in EverQuest?", k=2)
    
    if results:
        print("✅ Knowledge base test successful!")
        print(f"Sample result: {results[0].page_content[:100]}...")
    else:
        print("❌ Knowledge base test failed")
        return False
    
    # Create a simple configuration file
    config_content = f"""# DanzarVLM Knowledge Base Configuration
# Generated automatically by setup_knowledge_base.py

KNOWLEDGE_BASE_ENABLED: true
KNOWLEDGE_BASE_TYPE: "chroma"
KNOWLEDGE_BASE_PATH: "{kb_dir / 'chroma_db'}"
COLLECTION_NAME: "everquest_knowledge"
EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE: 500
CHUNK_OVERLAP: 50
TOP_K_RESULTS: 3
"""
    
    config_file = kb_dir / "kb_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Configuration saved to: {config_file}")
    
    return True

def main():
    """Main setup function"""
    print("🚀 DanzarVLM Knowledge Base Setup")
    print("=" * 50)
    
    if setup_basic_everquest_knowledge():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Install required packages if not already installed:")
        print("   pip install langchain chromadb sentence-transformers")
        print("2. Update your DanzarVLM configuration to use the knowledge base")
        print("3. Restart DanzarVLM to load the new knowledge base")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 