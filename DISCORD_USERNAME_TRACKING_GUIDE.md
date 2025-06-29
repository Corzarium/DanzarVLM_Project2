# Discord Username Tracking Guide

## Overview

The DanzarAI system has been enhanced to properly track and use actual Discord usernames instead of generic "User{id}" names. This provides a much more natural and personalized experience when interacting with the VLM.

## Key Improvements

### ✅ Enhanced User Tracking
- **Real Usernames**: Uses actual Discord display names instead of generic IDs
- **User Mapping**: Maintains a mapping of user IDs to display names
- **Consistent Naming**: Ensures usernames are consistent throughout the conversation
- **Fallback Handling**: Gracefully handles cases where user objects aren't available

### ✅ Improved Voice Processing
- **Individual Speaker Recognition**: Tracks each Discord user separately
- **Username in Responses**: VLM responses reference actual usernames
- **Personalized Interactions**: Creates more natural conversations
- **User State Tracking**: Monitors speaking status for each user

## How It Works

### 1. User Registration
When a Discord user speaks, the system:
```python
# Updates user information and gets display name
display_name = voice_receiver.update_user_info(user_id, user_object)

# Stores mapping for future reference
voice_receiver.user_names[user_id] = display_name
```

### 2. Voice Processing
During voice processing:
```python
# Gets actual username instead of generic name
display_name = voice_receiver.get_user_display_name(user_id)

# Processes speech with real username
await voice_receiver._process_user_speech(user_id, display_name, reason)
```

### 3. LLM Integration
The VLM receives the actual username:
```python
# LLM gets real username for personalized responses
response = await llm_service.handle_user_text_query(
    user_text=transcription,
    user_name=display_name  # Real Discord username
)
```

## Usage Examples

### Before (Generic Names)
```
🎤 Processing User123456789: 2.5s, max_vol: 0.8 - silence timeout
🤖 DanzarAI: Hello User123456789, how can I help you today?
```

### After (Real Usernames)
```
🎤 Processing GamerDude: 2.5s, max_vol: 0.8 - silence timeout
🤖 DanzarAI: Hey GamerDude! What's happening in your game right now?
```

## Commands

### `!users list`
Shows all currently tracked Discord users:
```
👥 Tracked Discord Users
Users currently being tracked for voice processing

👤 GamerDude
ID: 123456789 | Status: 🎤 Speaking

👤 StreamerGirl  
ID: 987654321 | Status: 🔇 Silent

Total users: 2
```

### `!users clear`
Clears all user tracking data (useful for privacy or troubleshooting).

## Technical Implementation

### Enhanced DiscordVoiceReceiver
```python
class DiscordVoiceReceiver:
    def __init__(self, app_context):
        # Enhanced user tracking
        self.user_names: Dict[int, str] = {}  # Map user_id to display_name
        self.user_cache: Dict[int, Any] = {}  # Cache Discord user objects
    
    def update_user_info(self, user_id: int, user_object: Any) -> str:
        """Update user information and return display name."""
        # Prefer display_name over name
        if hasattr(user_object, 'display_name') and user_object.display_name:
            display_name = user_object.display_name
        elif hasattr(user_object, 'name') and user_object.name:
            display_name = user_object.name
        else:
            display_name = f"User{user_id}"
        
        self.user_names[user_id] = display_name
        return display_name
    
    def get_user_display_name(self, user_id: int) -> str:
        """Get display name for a user ID."""
        return self.user_names.get(user_id, f"User{user_id}")
```

### NativeDiscordSink Integration
```python
class NativeDiscordSink:
    def write(self, data, user):
        # Handle both user objects and user IDs
        if isinstance(user, int):
            user_id = user
            display_name = self.voice_receiver.get_user_display_name(user_id)
        else:
            user_id = user.id
            display_name = self.voice_receiver.update_user_info(user_id, user)
        
        # Process with real username
        # ... rest of processing
```

## Benefits

### 🎯 Personalization
- VLM responses feel more natural and personal
- Users are addressed by their actual names
- Conversation context includes real usernames

### 🔍 Better Tracking
- Monitor who is speaking in voice channels
- Track individual user interaction patterns
- Debug voice processing issues more easily

### 🛡️ Privacy & Control
- Clear user tracking data with `!users clear`
- No permanent storage of user information
- Respects Discord privacy settings

### 🎮 Gaming Context
- Better integration with game-specific responses
- Personalized gaming advice and commentary
- Enhanced role-playing and immersion

## Testing

Run the test script to see the enhanced functionality:
```bash
python test_discord_voice_users.py
```

This demonstrates:
- User registration and tracking
- Username mapping and retrieval
- Voice sink processing
- Cleanup and resource management

## Configuration

The enhanced username tracking is enabled by default. No additional configuration is required.

### Optional Settings
```yaml
# In global_settings.yaml (if needed for future customization)
VOICE_PROCESSING:
  track_usernames: true
  cache_user_objects: true
  fallback_to_generic: true
```

## Troubleshooting

### Users Not Appearing
1. Check if voice services are initialized: `!status`
2. Verify Discord connection: `!status`
3. Check voice channel permissions
4. Use `!users list` to see current tracking

### Generic Names Still Showing
1. Ensure Discord.py is properly loaded
2. Check user object availability
3. Verify voice receiver initialization
4. Clear and restart voice services

### Privacy Concerns
1. Use `!users clear` to remove tracking data
2. Restart the bot to clear all caches
3. Check Discord privacy settings
4. Review voice channel permissions

## Future Enhancements

### Planned Features
- **Voice Cloning**: Use individual user voice characteristics
- **User Preferences**: Store individual user settings
- **Conversation History**: Per-user conversation tracking
- **Role-Based Responses**: Different responses based on user roles

### Integration Opportunities
- **Game Profiles**: Link usernames to game-specific settings
- **Memory Service**: Associate memories with specific users
- **TTS Personalization**: Different voice styles per user
- **Access Control**: Role-based command permissions

## Conclusion

The enhanced Discord username tracking significantly improves the user experience by providing personalized, natural interactions with the VLM. Users are now addressed by their actual names, making conversations feel more engaging and immersive.

This enhancement maintains privacy and control while providing the foundation for future personalized features and integrations. 