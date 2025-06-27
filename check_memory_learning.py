#!/usr/bin/env python3
"""
DanzarAI Memory Learning Checker
Checks if Danzar is learning by examining stored memories and RAG collections.
"""

import sqlite3
import os
from datetime import datetime
import json

def check_memory_database():
    """Check the SQLite memory database for stored conversations."""
    print("🧠 DanzarAI Memory Learning Analysis")
    print("=" * 50)
    
    # Check if memory database exists
    memory_db_path = os.path.join("data", "memory.db")
    if not os.path.exists(memory_db_path):
        print("❌ Memory database not found at data/memory.db")
        print("   Danzar is not storing any memories yet.")
        return
    
    try:
        conn = sqlite3.connect(memory_db_path)
        cursor = conn.cursor()
        
        # Get total memory count
        cursor.execute('SELECT COUNT(*) FROM memories')
        total_memories = cursor.fetchone()[0]
        
        print(f"📊 Total memories stored: {total_memories}")
        
        if total_memories == 0:
            print("❌ No memories found - Danzar is not learning yet")
            return
        
        # Get memories by source
        cursor.execute('SELECT source, COUNT(*) FROM memories GROUP BY source')
        sources = cursor.fetchall()
        
        print("\n📈 Memory breakdown by source:")
        for source, count in sources:
            print(f"   {source}: {count}")
        
        # Get recent memories
        cursor.execute('''
            SELECT content, source, timestamp, metadata 
            FROM memories 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_memories = cursor.fetchall()
        
        print(f"\n🕒 Recent memories (last 10):")
        for i, (content, source, timestamp, metadata) in enumerate(recent_memories, 1):
            dt = datetime.fromtimestamp(timestamp)
            content_preview = content[:80] + "..." if len(content) > 80 else content
            print(f"   {i}. [{source}] {content_preview}")
            print(f"      Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get memory stats
        cursor.execute('''
            SELECT 
                MIN(timestamp) as oldest,
                MAX(timestamp) as newest,
                COUNT(DISTINCT DATE(timestamp, 'unixepoch')) as unique_days
            FROM memories
        ''')
        stats = cursor.fetchone()
        
        if stats[0] and stats[1]:
            oldest = datetime.fromtimestamp(stats[0])
            newest = datetime.fromtimestamp(stats[1])
            days_active = stats[2]
            
            print(f"\n📅 Memory timeline:")
            print(f"   First memory: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Latest memory: {newest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Days active: {days_active}")
            print(f"   Average memories per day: {total_memories / max(days_active, 1):.1f}")
        
        # Check for conversation patterns
        cursor.execute('''
            SELECT content, source 
            FROM memories 
            WHERE source IN ('user_query', 'bot_response')
            ORDER BY timestamp DESC 
            LIMIT 20
        ''')
        conversations = cursor.fetchall()
        
        if conversations:
            print(f"\n💬 Recent conversation patterns:")
            user_queries = [c[0] for c in conversations if c[1] == 'user_query'][:5]
            bot_responses = [c[0] for c in conversations if c[1] == 'bot_response'][:5]
            
            print("   Recent user queries:")
            for i, query in enumerate(user_queries, 1):
                preview = query[:60] + "..." if len(query) > 60 else query
                print(f"     {i}. {preview}")
            
            print("   Recent bot responses:")
            for i, response in enumerate(bot_responses, 1):
                preview = response[:60] + "..." if len(response) > 60 else response
                print(f"     {i}. {preview}")
        
        conn.close()
        
        # Learning assessment
        print(f"\n🎯 Learning Assessment:")
        if total_memories > 100:
            print("   ✅ Danzar is actively learning and storing conversations")
        elif total_memories > 10:
            print("   🟡 Danzar is learning but has limited conversation history")
        else:
            print("   ❌ Danzar has minimal learning data")
        
        if total_memories > 0:
            print("   📚 Memory sources:")
            for source, count in sources:
                percentage = (count / total_memories) * 100
                print(f"      {source}: {percentage:.1f}%")
        
    except Exception as e:
        print(f"❌ Error reading memory database: {e}")

def check_rag_collections():
    """Check RAG collections for stored knowledge."""
    print(f"\n🔍 RAG Knowledge Base Check")
    print("=" * 30)
    
    try:
        # Try to connect to Qdrant
        import qdrant_client
        client = qdrant_client.QdrantClient("localhost", port=6333)
        
        collections = client.get_collections()
        if not collections.collections:
            print("❌ No RAG collections found")
            print("   Danzar is not storing knowledge in RAG yet")
            return
        
        print(f"📚 Found {len(collections.collections)} RAG collections:")
        for collection in collections.collections:
            collection_name = collection.name
            try:
                info = client.get_collection(collection_name)
                point_count = info.points_count
                print(f"   {collection_name}: {point_count} knowledge points")
            except:
                print(f"   {collection_name}: Unable to get point count")
        
    except ImportError:
        print("❌ Qdrant client not available")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    check_memory_database()
    check_rag_collections()
    
    print(f"\n🎉 Analysis complete!")
    print("   Danzar is learning if you see:")
    print("   - Multiple memories in the database")
    print("   - Recent conversation patterns")
    print("   - RAG collections with knowledge points") 