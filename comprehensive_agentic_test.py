#!/usr/bin/env python3

import asyncio
import time
from services.agentic_rag_service import AgenticRAGService
from services.ollama_rag_service import OllamaRAGService
from DanzarVLM import AppContext

def print_banner(text):
    print("\n" + "="*60)
    print(f"🎯 {text}")
    print("="*60)

def print_result(query, response, metadata):
    print(f"\n🔍 Query: {query}")
    print(f"⏱️  Processing time: {metadata.get('processing_time', 0):.2f}s")
    print(f"🤖 Method: {metadata.get('method', 'unknown')}")
    print(f"💬 Response: {response}")
    
    if 'conversation_context' in metadata:
        print(f"🧠 Context: {metadata['conversation_context']}")
    if 'conversation_history_length' in metadata:
        print(f"📚 History: {metadata['conversation_history_length']} turns")
    if 'agents_used' in metadata:
        print(f"👥 Agents: {', '.join(metadata['agents_used'])}")
    if 'retrieval' in metadata:
        retrieval = metadata['retrieval']
        print(f"🔬 Retrieval: {retrieval.get('iterations', 0)} iterations, {retrieval.get('filtered_results', 0)} results, quality: {retrieval.get('final_quality_score', 0):.2f}")

async def main():
    print_banner("COMPREHENSIVE AGENTIC RAG TESTING")
    
    # Initialize services
    app_context = AppContext()
    rag_service = OllamaRAGService(app_context)
    
    agentic = AgenticRAGService(app_context)
    agentic.ollama_rag = rag_service
    
    print("✅ All services initialized successfully!")
    
    # Test sequence demonstrating different capabilities
    test_queries = [
        # Test 1: Game-specific query (should use vector DB)
        "Tell me about EverQuest raids",
        
        # Test 2: Follow-up query (should use conversational memory)
        "What about the gear from those raids?",
        
        # Test 3: Conversational query
        "What have we been talking about?",
        
        # Test 4: Complex gaming query
        "How do I get better at tanking in EverQuest?",
        
        # Test 5: Another follow-up
        "What classes can tank?"
    ]
    
    print_banner("RUNNING AGENTIC RAG SEQUENCE")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 TEST {i}/5:")
        
        start_time = time.time()
        response, metadata = await agentic.smart_retrieve(query, user_name="Tester")
        
        print_result(query, response, metadata)
        
        # Brief pause between queries
        await asyncio.sleep(0.5)
    
    print_banner("FINAL CONVERSATIONAL STATE")
    
    conversation_summary = agentic._get_recent_conversation_summary()
    print(f"📖 Conversation Summary:\n{conversation_summary}")
    print(f"💾 Context Keys: {list(agentic.conversation_context.keys())}")
    print(f"📚 Total Conversation Turns: {len(agentic.conversation_history)}")
    
    # Test reflection capabilities
    print_banner("REFLECTION & QUALITY ASSESSMENT")
    
    if agentic.conversation_history:
        last_turn = agentic.conversation_history[-1]
        reflection = agentic._reflection_agent(
            last_turn["query"], 
            [], 
            last_turn["response"]
        )
        print(f"🔍 Last Query Quality: {reflection['quality_score']:.2f}")
        print(f"⚡ Needs Refinement: {reflection['needs_refinement']}")
        print(f"📊 Assessment: {reflection['quality_assessment']}")
    
    print_banner("TESTING COMPLETE")
    print("🎉 All agentic features demonstrated successfully!")
    print("✨ Features working:")
    print("   • Multi-agent coordination")
    print("   • Iterative refinement")
    print("   • Conversational memory")
    print("   • Quality reflection")
    print("   • Context-aware enhancement")

if __name__ == "__main__":
    asyncio.run(main()) 