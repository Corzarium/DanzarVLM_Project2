# LangChain Tools Integration for DanzarAI

## Overview

This document describes the LangChain tools integration for DanzarAI, which enables agentic behavior and natural tool usage by the LLM. The integration provides a seamless way for the AI to use vision, memory, and system tools through LangChain's agent framework.

## Architecture

### Components

1. **LangChain Tools Service** (`services/langchain_tools_service.py`)
   - Wraps existing DanzarAI capabilities as LangChain tools
   - Manages agent initialization and execution
   - Provides tool descriptions and schemas

2. **LangChain Model Client** (`services/langchain_model_client.py`)
   - Wraps the existing Ollama model client for LangChain compatibility
   - Handles message format conversion
   - Provides async generation capabilities

3. **App Context Integration** (`core/app_context.py`)
   - Initializes LangChain tools and agent
   - Provides tools status information
   - Manages tool lifecycle

4. **Conversational AI Integration** (`services/conversational_ai_service.py`)
   - Routes messages to LangChain agent when available
   - Falls back to standard processing if agent unavailable
   - Maintains conversation context

## Available Tools

### Vision Tools
- **capture_screenshot**: Capture a screenshot of the current game screen
- **analyze_screenshot**: Analyze a screenshot to understand what's happening
- **get_vision_summary**: Get a summary of recent visual activity
- **check_vision_capabilities**: Check what vision capabilities are available

### Memory Tools
- **search_memory**: Search for relevant memories and past interactions
- **store_memory**: Store new information in memory for future reference

### Game Context Tools
- **get_game_context**: Get information about the current game context
- **set_game_context**: Set or update the current game context

### System Tools
- **get_system_status**: Get current system status and capabilities
- **get_conversation_history**: Get recent conversation history

## Installation

### 1. Install LangChain Dependencies

```bash
pip install langchain==0.3.0 langchain-core==0.3.0 langchain-community==0.3.0 langgraph==0.2.0
```

### 2. Update Requirements

The requirements.txt file has been updated to include LangChain dependencies:

```txt
# LangChain for tool integration and agentic behavior
langchain==0.3.0
langchain-core==0.3.0
langchain-community==0.3.0
langgraph==0.2.0
```

## Usage

### Automatic Integration

The LangChain tools are automatically integrated into the conversational AI service. When a user sends a message, the system will:

1. Check if LangChain agent is available
2. If available, route the message to the agent for processing
3. If not available, fall back to standard processing
4. Store agent responses in RAG memory

### Discord Commands

The following Discord commands are available for managing LangChain tools:

#### `!langchain status`
Shows the status of LangChain tools and agent.

#### `!langchain init`
Initializes LangChain tools and agent.

#### `!langchain test`
Tests all available tools.

#### `!langchain agent <message>`
Tests the agent with a specific message.

### Example Usage

```
User: "What can you see on my screen right now?"

Agent: [Uses capture_screenshot tool]
       [Uses analyze_screenshot tool]
       "I can see you're playing EverQuest! There's a character standing in what looks like the Commonlands, 
        with a health bar at about 80% and some chat text visible. The UI shows typical EverQuest elements 
        like the spell bar and inventory slots."
```

## Configuration

### Agent Personality

The LangChain agent is configured with Danzar's personality:

- Sharp wit and sarcastic humor
- Delightfully unhinged responses
- Natural tool usage without mentioning tools
- Context-aware responses

### Tool Usage Guidelines

- Tools are used naturally when needed
- Tool usage is not explicitly mentioned unless part of snarky commentary
- Results are combined with personality for engaging responses
- Vision tools are used proactively when users ask about screen content

## Testing

### Run the Test Script

```bash
python test_langchain_tools.py
```

This script will:
1. Test LangChain requirements availability
2. Test tools initialization
3. Test agent functionality
4. Provide detailed status information

### Expected Output

```
🚀 Starting LangChain Tools Integration Test
📦 Testing LangChain requirements...
✅ LangChain version: 0.3.0
✅ LangChain Core version: 0.3.0
✅ LangChain Community version: 0.3.0
✅ LangGraph version: 0.2.0
🧪 Starting LangChain Tools Test
✅ Settings loaded
✅ App context created
🔧 Testing LangChain tools initialization...
✅ LangChain tools initialized successfully
📊 Tools Info: {'total_tools': 8, 'agent_ready': True, ...}
🧪 Testing LangChain tools...
📊 Test Results: {...}
🤖 Testing LangChain agent...
🤖 Agent Response: "I have several tools available..."
✅ LangChain Tools Test Completed
✅ All tests completed
```

## Troubleshooting

### Common Issues

#### 1. LangChain Not Installed
```
⚠️ LangChain not installed
💡 Install with: pip install langchain==0.3.0
```

**Solution**: Install the required dependencies as shown in the Installation section.

#### 2. Agent Initialization Failed
```
❌ LangChain Tools initialization failed
```

**Possible Causes**:
- Model client not available
- Tool dependencies missing
- Configuration issues

**Solution**: Check that all services are properly initialized before LangChain tools.

#### 3. Tool Execution Errors
```
❌ Error executing capture_screenshot: ...
```

**Possible Causes**:
- Vision services not available
- NDI stream not accessible
- Model client issues

**Solution**: Ensure vision integration service is properly initialized.

### Debug Mode

Enable debug logging to see detailed tool execution:

```python
logging.getLogger("LangChainTools").setLevel(logging.DEBUG)
```

## Benefits

### 1. Natural Tool Usage
The LLM can naturally decide when to use tools without explicit prompting.

### 2. Agentic Behavior
The system can take proactive actions like capturing screenshots when users ask about screen content.

### 3. Memory Integration
Tools can access and store information in RAG memory for future reference.

### 4. Context Awareness
The agent maintains conversation context and can use tools based on conversation history.

### 5. Fallback Support
If LangChain tools are unavailable, the system gracefully falls back to standard processing.

## Future Enhancements

### Planned Features

1. **More Tools**: Additional tools for game-specific actions
2. **Tool Chaining**: Complex workflows using multiple tools
3. **Custom Tools**: User-defined tools for specific use cases
4. **Tool Learning**: Tools that improve based on usage patterns
5. **Multi-Agent**: Multiple specialized agents for different tasks

### Integration Opportunities

1. **Discord Slash Commands**: Native Discord slash command integration
2. **Web Interface**: Web-based tool management interface
3. **API Endpoints**: REST API for external tool access
4. **Plugin System**: Plugin architecture for custom tools

## Conclusion

The LangChain tools integration provides a powerful foundation for agentic behavior in DanzarAI. It enables natural tool usage, maintains the system's personality, and provides a robust fallback mechanism. The integration is designed to be extensible and maintainable, allowing for future enhancements and customizations.

For questions or issues, refer to the troubleshooting section or check the logs for detailed error information. 