# TTS Echo Detection System Guide

## Overview

DanzarAI uses an intelligent, multi-layered TTS echo detection system to prevent feedback loops while allowing legitimate user speech. The system has evolved from simple regex patterns to sophisticated heuristics that can distinguish between actual TTS echoes and legitimate user input.

## System Architecture

### 1. Primary Echo Detection (`is_likely_tts_echo`)

The main echo detection method uses multiple strategies:

#### Length-Based Filtering
- **Long transcriptions (>100 chars)**: Automatically allowed as unlikely to be echoes
- **Short transcriptions**: Subject to more scrutiny

#### Exact Match Detection
- Compares transcription with recent TTS outputs (last 30 seconds)
- **Exact matches**: Immediately flagged as echoes
- **High word overlap (90%+)**: Flagged as potential echoes

#### Intelligent Heuristic Analysis (`_heuristic_echo_detection`)

The system uses 6 different heuristics to analyze transcriptions:

##### Heuristic 1: TTS-Specific Phrases
Detects phrases that are clearly TTS artifacts:
```python
tts_specific_phrases = [
    # System identification
    "i'm danzar", "danzarai", "gaming assistant", "ai assistant",
    "i'm an ai", "i'm an assistant", "i'm here to help",
    
    # TTS response patterns
    "that's interesting", "let me help you", "i understand",
    "first they mentioned", "user is confident", "prep expansions ready",
    
    # Processing phrases
    "processing that", "trouble processing", "try again",
    "please try again", "error processing", "audio transcription failed"
]
```

##### Heuristic 2: Similarity Analysis
- Calculates word-based similarity with recent TTS outputs
- Uses Jaccard similarity (intersection/union of word sets)
- Flags transcriptions with >80% similarity as potential echoes

##### Heuristic 3: Repetition Detection
- Identifies excessive word repetition (any word >50% of total)
- TTS echoes often have unnatural repetition patterns

##### Heuristic 4: Pattern Recognition
Uses regex patterns to detect TTS artifacts:
```python
tts_artifact_patterns = [
    r"\b(i am|i'm) (an|a) (ai|assistant|bot)\b",
    r"\b(that is|that's) (very|quite) (interesting|fascinating)\b",
    r"\b(let me) (help|assist) (you|you with)\b",
    r"\b(first|initially) (they|the user) (mentioned|said)\b"
]
```

##### Heuristic 5: Length and Complexity
- Very short transcriptions (≤20 chars) with common echo words
- Checks against known system words: "danzar", "assistant", "ai", "bot", "help"

##### Heuristic 6: Timing Analysis
- Analyzes timing relative to recent TTS output
- Short transcriptions appearing within 5 seconds of TTS are more suspicious
- Checks for common TTS artifacts in short transcriptions

### 2. Secondary Noise Filtering (`process_transcription_queue`)

Additional filtering in the transcription queue processor:

#### Single-Word Noise Detection
```python
single_word_noise = ["um", "uh", "ah", "oh", "mm", "hmm", "hm", "eh", "er"]
```
- Only blocks exact single-word matches
- Allows legitimate speech containing these words

#### Phrase Pattern Detection
```python
phrase_noise_patterns = [
    "i'm having trouble processing", "please try again", "error processing",
    "audio transcription failed", "i encountered an error"
]
```
- Uses exact phrase matching, not substring matching
- Prevents false positives on legitimate speech

#### Length Override
- **Long transcriptions (>50 chars)**: Automatically allowed regardless of pattern matches
- Prevents blocking legitimate long speech

## Key Improvements Over Regex-Only Approach

### 1. Context Awareness
- Considers recent TTS history
- Analyzes timing relationships
- Uses similarity scoring instead of exact matches

### 2. Intelligent Pattern Recognition
- Distinguishes between TTS artifacts and legitimate speech patterns
- Uses multiple heuristics instead of single regex rules
- Considers speech complexity and naturalness

### 3. Conservative Default Behavior
- **Default action**: Allow transcription unless clearly identified as echo
- Prevents false positives that block legitimate user input
- Uses length-based overrides for safety

### 4. Multi-Layer Protection
- Primary echo detection in `AudioFeedbackPrevention`
- Secondary noise filtering in transcription queue
- Length-based safety overrides

## Example Scenarios

### Scenario 1: Legitimate User Speech
**Input**: "Does that sound like a good group to you?"
- **Length check**: 35 chars (< 100) - proceed to analysis
- **Exact match**: No matches with recent TTS
- **Similarity**: Low similarity with recent TTS outputs
- **Patterns**: No TTS-specific patterns detected
- **Result**: ✅ **ALLOWED** - legitimate user speech

### Scenario 2: TTS Echo
**Input**: "I'm Danzar, your gaming assistant"
- **Length check**: 32 chars (< 100) - proceed to analysis
- **Exact match**: No exact matches
- **Patterns**: Matches TTS-specific phrase "i'm danzar"
- **Result**: ❌ **BLOCKED** - TTS echo detected

### Scenario 3: Long Legitimate Speech
**Input**: "So I'm thinking about starting a team that's going to be a Shadow Knight, Bard, Cleric, probably Necromancer, Mage, and Enchanter. Does that sound like a good group to you?"
- **Length check**: 156 chars (> 100) - automatically allowed
- **Result**: ✅ **ALLOWED** - long transcription unlikely to be echo

## Configuration and Tuning

### Adjustable Parameters

1. **Similarity Threshold**: Currently 80% for high similarity detection
2. **Timing Window**: 30 seconds for exact match checking, 60 seconds for similarity analysis
3. **Length Thresholds**: 100 chars for auto-allow, 50 chars for pattern override
4. **Repetition Threshold**: 50% for excessive word repetition

### Adding New Patterns

To add new TTS-specific phrases:

1. **Primary Detection**: Add to `tts_specific_phrases` in `_heuristic_echo_detection`
2. **Secondary Filtering**: Add to `phrase_noise_patterns` in `process_transcription_queue`

### Debugging Echo Detection

Enable debug logging to see echo detection decisions:
```python
# In global_settings.yaml
VLM_DEBUG_MODE: true
```

## Best Practices

### 1. Conservative Approach
- Always err on the side of allowing legitimate speech
- Use length-based overrides for safety
- Default to allowing uncertain cases

### 2. Pattern Specificity
- Use exact phrases, not generic terms
- Avoid patterns that could match legitimate user speech
- Test patterns with real user input

### 3. Regular Monitoring
- Monitor false positives and false negatives
- Adjust patterns based on real usage
- Consider user feedback for pattern refinement

### 4. Performance Considerations
- Echo detection runs synchronously for each transcription
- Keep heuristics lightweight and efficient
- Use early returns for obvious cases

## Troubleshooting

### Common Issues

1. **False Positives**: Legitimate speech being blocked
   - **Solution**: Add length-based overrides or refine patterns
   - **Check**: Look for overly broad pattern matches

2. **False Negatives**: TTS echoes getting through
   - **Solution**: Add specific patterns for detected echoes
   - **Check**: Analyze similarity scores and timing

3. **Performance Issues**: Slow echo detection
   - **Solution**: Optimize similarity calculations
   - **Check**: Reduce analysis window or simplify heuristics

### Debugging Commands

Use Discord commands to monitor echo detection:
```bash
!tts status  # Check TTS service status
!memory status  # Check memory for recent interactions
```

## Future Enhancements

### Potential Improvements

1. **Machine Learning**: Train a model on echo vs legitimate speech patterns
2. **Audio Analysis**: Use audio characteristics to detect TTS vs human speech
3. **Context Awareness**: Consider conversation context in echo detection
4. **User Feedback**: Allow users to report false positives/negatives

### Integration with Other Systems

- **Memory Service**: Use conversation history for better context
- **Vision Service**: Consider visual context in echo detection
- **User Profiles**: Personalize echo detection per user

## Conclusion

The intelligent TTS echo detection system provides robust protection against feedback loops while minimizing false positives. The multi-layered approach ensures that legitimate user speech is rarely blocked while effectively preventing TTS echoes from being processed as user input.

The system is designed to be conservative and user-friendly, with multiple safety mechanisms to prevent blocking legitimate speech. Regular monitoring and pattern refinement help maintain optimal performance. 