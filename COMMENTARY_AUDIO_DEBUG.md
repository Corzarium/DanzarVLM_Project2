# Vision Commentary Audio Debug Guide

## Issue Summary
Vision commentary is being generated and TTS is working, but audio is not being heard.

## Current Status (from logs)
- ✅ Commentary generation: Working
- ✅ TTS synthesis: Working  
- ✅ Audio playback: Attempted
- ❌ Audio output: Not heard

## Debug Steps

### 1. Check Voice Connection
```
!join
```
Make sure bot joins voice channel successfully.

### 2. Test Basic TTS
```
!tts This is a test of the audio system
```
If you can't hear this, the issue is with Discord audio setup, not vision commentary.

### 3. Check Discord Audio Settings
- Settings → Voice & Video
- Output Device: Set to your speakers/headphones
- User Volume: Make sure bot volume is not 0%
- Try "Legacy Audio Subsystem" in Advanced

### 4. Check Windows Audio
- Right-click speaker → Sound settings
- Output: Make sure correct device is selected
- Test with Windows sound test

### 5. Check Bot Volume in Discord
- Right-click bot in voice channel
- User Volume: Make sure it's not 0%
- Server Deafened: Make sure bot is not deafened

### 6. Test Vision Commentary TTS
```
!watch
```
Then trigger some vision events and check:
- Do you see commentary text in Discord?
- Do you hear any audio at all?
- Are there any error messages in the logs?

### 7. Check Audio Feedback Prevention
The system has audio feedback prevention that might be interfering:
- Check if TTS is being blocked by feedback prevention
- Look for logs about "feedback prevention" or "TTS echo"

### 8. Restart Services
1. Stop the application
2. Close Discord completely
3. Restart Discord
4. Start the application again
5. Test with `!tts test` first
6. Then test with `!watch`

## Expected Behavior
When `!watch` is active:
1. You should see commentary text in Discord: "👁️ **Vision Commentary**: [text]"
2. You should hear TTS audio for each commentary
3. Commentary should be generated every 10-15 seconds when events are detected

## Common Issues
1. **Discord audio settings**: Wrong output device or muted bot
2. **Windows audio**: Wrong default output device
3. **Feedback prevention**: TTS being blocked as echo
4. **Voice connection**: Bot not properly connected to voice channel
5. **Audio overlap**: Too frequent commentary causing audio cutoff

## Next Steps
1. Test basic TTS first with `!tts test`
2. If basic TTS works, the issue is with vision commentary specifically
3. If basic TTS doesn't work, fix Discord/Windows audio settings first
4. Check logs for any error messages during commentary generation 