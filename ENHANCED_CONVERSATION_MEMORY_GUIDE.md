# Enhanced Conversation Memory System Guide

## Overview

The Enhanced Conversation Memory System provides a sophisticated memory architecture that combines **Short-Term Memory (STM) in RAM** with **Long-Term Memory (LTM) in RAG** to maintain proper conversation context and awareness.

## Architecture

### 🧠 Short-Term Memory (STM) - RAM-Based
- **Location**: In-memory storage for fast access
- **Purpose**: Immediate conversation context and recent interactions
- **Features**:
  - Per-user conversation sessions
  - Real-time context window management
  - Visual context integration
  - Automatic cleanup of old sessions
  - Thread-safe operations

### 🗄️ Long-Term Memory (LTM) - RAG-Based
- **Location**: Qdrant vector database with llama.cpp embeddings
- **Purpose**: Persistent storage of important conversation data
- **Features**:
  - Vector-based similarity search
  - Conversation consolidation and summarization
  - Game context and topic tracking
  - Cross-session memory retrieval

## Key Components

### 1. EnhancedConversationMemory Service
```python
class EnhancedConversationMemory:
    - STM Storage (RAM): active_sessions, global_stm
    - LTM Integration: rag_service, ltm_collection
    - Background Tasks: cleanup, consolidation
    - Context Management: session tracking, visual context
```

### 2. ConversationSession
```python
@dataclass
class ConversationSession:
    - session_id: str
    - user_name: str
    - stm_entries: deque(maxlen=50)  # Last 50 entries in RAM
    - conversation_summary: str
    - current_topic: str
    - game_context: Optional[str]
    - visual_context_history: deque(maxlen=10)
```

### 3. STMEntry
```python
@dataclass
class STMEntry:
    - content: str
    - timestamp: float
    - user_name: str
    - entry_type: str  # 'user_input', 'bot_response', 'system_event', 'visual_context'
    - visual_context: Optional[Dict[str, Any]]
    - importance: float
```

## Configuration

### Global Settings (`config/global_settings.yaml`)
```yaml
CONVERSATION_MEMORY:
  stm_window_minutes: 30  # STM retention window
  max_active_sessions: 10  # Maximum concurrent sessions
  consolidation_threshold: 20  # Entries before LTM consolidation
  context_window_minutes: 60  # Conversation context window

VISION_PROCESSING:
  fps: 1  # Reduced for system stability
  visual_update_interval: 10.0  # Visual context update frequency
  max_visual_elements: 3  # Visual elements to track
  clip_processing_frequency: 5  # CLIP processing frequency
```

## Memory Flow

### 1. Conversation Entry Addition
```python
# Add user input to STM
conversation_memory.add_conversation_entry(
    user_name="VirtualAudio",
    content="What's happening in the game?",
    entry_type='user_input',
    visual_context=current_visual_context
)

# Add bot response to STM
conversation_memory.add_conversation_entry(
    user_name="VirtualAudio", 
    content="I can see you're in combat...",
    entry_type='bot_response',
    visual_context=current_visual_context
)
```

### 2. Context Retrieval
```python
# Get comprehensive conversation context
context = conversation_memory.get_conversation_context(
    user_name="VirtualAudio",
    include_ltm=True,
    max_stm_entries=10,
    max_ltm_results=3
)

# Returns:
{
    'user_name': 'VirtualAudio',
    'stm_entries': [recent_conversation_entries],
    'conversation_summary': 'Recent conversation summary',
    'current_topic': 'combat',
    'ltm_results': [relevant_long_term_memories],
    'visual_context': current_visual_context
}
```

### 3. Memory Consolidation
- **Automatic**: When STM entries reach threshold (20 entries)
- **Background**: Every 10 minutes for all sessions
- **Process**: 
  1. Create consolidation entry with conversation summary
  2. Store in RAG with vector embeddings
  3. Clear old STM entries
  4. Update conversation summary

## Integration with Vision-Aware Conversation

### Enhanced Prompt Building
```python
def _build_enhanced_context_prompt(self, turn, conversation_context):
    prompt_parts = []
    
    # System context
    prompt_parts.append("You are DanzarAI, a gaming commentary assistant with vision capabilities.")
    
    # Visual context
    if turn.visual_context:
        prompt_parts.append(f"Current visual context: {turn.visual_context.scene_summary}")
    
    # STM entries (recent conversation)
    stm_entries = conversation_context.get('stm_entries', [])
    if stm_entries:
        prompt_parts.append("Recent conversation:")
        for entry in stm_entries[-5:]:
            if entry.entry_type == 'user_input':
                prompt_parts.append(f"User: {entry.content}")
            elif entry.entry_type == 'bot_response':
                prompt_parts.append(f"You: {entry.content}")
    
    # LTM results (long-term memory)
    ltm_results = conversation_context.get('ltm_results', [])
    if ltm_results:
        prompt_parts.append("Relevant long-term memory:")
        for result in ltm_results[:2]:
            prompt_parts.append(f"- {result}")
    
    # Current user input
    prompt_parts.append(f"User: {turn.user_input}")
    prompt_parts.append("You:")
    
    return "\n".join(prompt_parts)
```

## Benefits

### 1. **Context Awareness**
- Maintains conversation flow across multiple interactions
- Remembers user preferences and game context
- Provides relevant historical information

### 2. **Performance Optimization**
- Fast STM access for immediate responses
- Efficient LTM retrieval for relevant context
- Reduced system load with optimized processing

### 3. **Memory Persistence**
- Important conversations stored in RAG
- Cross-session memory retrieval
- Game-specific context preservation

### 4. **Visual Integration**
- Visual context stored with conversation entries
- CLIP insights integrated into memory
- Game state awareness maintained

## Usage Examples

### Basic Conversation Flow
```python
# 1. User asks about game state
user_input = "What's my health like?"
visual_context = get_current_visual_context()

# 2. Add to STM
conversation_memory.add_conversation_entry(
    user_name="VirtualAudio",
    content=user_input,
    entry_type='user_input',
    visual_context=visual_context
)

# 3. Get context for response generation
context = conversation_memory.get_conversation_context("VirtualAudio")

# 4. Generate response with full context
response = await generate_response_with_context(user_input, context)

# 5. Store response in STM
conversation_memory.add_conversation_entry(
    user_name="VirtualAudio",
    content=response,
    entry_type='bot_response',
    visual_context=visual_context
)
```

### Memory Consolidation Example
```python
# When STM reaches threshold, automatically consolidate to LTM
if len(session.stm_entries) >= consolidation_threshold:
    # Create consolidation entry
    consolidation_data = {
        'session_id': session.session_id,
        'user_name': session.user_name,
        'conversation_summary': session.conversation_summary,
        'current_topic': session.current_topic,
        'game_context': session.game_context,
        'entries': [entry data...]
    }
    
    # Store in RAG
    rag_service.ingest_text(
        collection="conversation_memory",
        text=f"Conversation with {session.user_name}: {session.conversation_summary}",
        metadata=consolidation_data
    )
```

## Monitoring and Debugging

### Memory Statistics
```python
stats = conversation_memory.get_memory_stats()
# Returns:
{
    'active_sessions': 3,
    'global_stm_entries': 45,
    'ltm_collection': 'conversation_memory',
    'rag_service_available': True,
    'sessions': {
        'VirtualAudio_Session': {
            'user_name': 'VirtualAudio',
            'stm_entries': 15,
            'current_topic': 'combat',
            'game_context': 'EverQuest'
        }
    }
}
```

### Discord Commands
```bash
!memory status  # Show memory statistics
!memory clear VirtualAudio  # Clear specific user memory
!memory stats  # Detailed memory information
```

## Troubleshooting

### Common Issues

1. **Memory Not Persisting**
   - Check RAG service connection
   - Verify Qdrant is running
   - Check consolidation threshold settings

2. **Context Loss**
   - Verify STM window settings
   - Check session cleanup intervals
   - Monitor memory usage

3. **Performance Issues**
   - Reduce STM window size
   - Lower consolidation threshold
   - Increase cleanup frequency

### Debug Logging
```python
# Enable debug logging for memory operations
logger.debug(f"[EnhancedConversationMemory] Added {entry_type} for {user_name}")
logger.debug(f"[EnhancedConversationMemory] Retrieved context: {len(stm_entries)} STM, {len(ltm_results)} LTM")
logger.info(f"[EnhancedConversationMemory] Consolidated session {session_id} to LTM")
```

## Future Enhancements

### Planned Features
1. **Semantic Memory**: Advanced topic clustering and relationship mapping
2. **Emotional Memory**: Sentiment tracking and emotional context
3. **Procedural Memory**: Game action sequences and patterns
4. **Episodic Memory**: Detailed event sequences and timelines
5. **Memory Compression**: Advanced summarization and compression algorithms

### Performance Optimizations
1. **Memory Pooling**: Shared memory pools for better resource management
2. **Predictive Loading**: Pre-load relevant memories based on context
3. **Memory Indexing**: Advanced indexing for faster retrieval
4. **Distributed Memory**: Multi-node memory distribution

## Conclusion

The Enhanced Conversation Memory System provides a robust foundation for maintaining conversation context and awareness in DanzarAI. By combining fast STM access with persistent LTM storage, the system ensures that the AI can maintain meaningful conversations while preserving important information for future interactions.

The integration with vision processing and CLIP insights further enhances the system's ability to understand and respond to the current game state, making DanzarAI a truly context-aware gaming companion. 