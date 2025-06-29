# Vision Commentary Fix Summary

## Issue Analysis

**Problem**: Vision commentary is not being generated despite vision events being detected.

**Root Cause**: Two separate vision pipelines are running:
1. **Main vision pipeline** (detecting events correctly)
2. **Vision integration service pipeline** (not receiving events)

**Evidence from logs**:
- ✅ Vision events detected: "Detected event: yolo - health_bar"
- ✅ Vision integration service initialized: "Vision Integration Service initialized"
- ✅ Event processor loop running: "Event processor loop started"
- ❌ No commentary generated: No "Generated commentary" logs

## The Fix

The vision integration service needs to use the existing vision pipeline from the app context instead of creating its own.

### Current Problem
```python
# In vision_integration_service.py - creates its own pipeline
self.vision_pipeline = VisionPipeline(
    event_callback=self._handle_vision_event,
    clip_callback=self._handle_clip_update,
    config_path="config/vision_config.yaml"
)
```

### Solution
```python
# Use existing pipeline from app context
if hasattr(self.app_context, 'vision_pipeline') and self.app_context.vision_pipeline:
    self.vision_pipeline = self.app_context.vision_pipeline
    # Set callbacks on existing pipeline
    self.vision_pipeline.set_event_callback(self._handle_vision_event)
    self.vision_pipeline.set_clip_callback(self._handle_clip_update)
else:
    # Create new pipeline only if none exists
    self.vision_pipeline = VisionPipeline(...)
```

## Implementation Steps

1. **Modify vision integration service** to use existing pipeline
2. **Add callback setting methods** to vision pipeline
3. **Test commentary generation** with `!watch` command

## Expected Result

After the fix:
- Vision events will be properly routed to the vision integration service
- Commentary will be generated and displayed in Discord
- TTS will play the commentary audio

## Testing

1. Start the application
2. Use `!watch` command
3. Verify commentary appears in Discord text
4. Verify TTS audio plays

## Files to Modify

- `services/vision_integration_service.py` - Use existing pipeline
- `vision_pipeline.py` - Add callback setting methods (if needed) 