# Azure TTS Setup Guide

This guide will help you replace Chatterbox TTS with Microsoft Azure Text-to-Speech using the `en-US-AdamMultilingualNeural` voice.

## 🚀 Quick Setup

### 1. Create Azure Speech Service

1. Go to [Azure Portal](https://portal.azure.com)
2. Click "Create a resource"
3. Search for "Speech service"
4. Click "Create"
5. Fill in the details:
   - **Subscription**: Your Azure subscription
   - **Resource group**: Create new or use existing
   - **Region**: Choose closest to you (e.g., East US)
   - **Name**: `danzar-tts` (or any name you prefer)
   - **Pricing tier**: Free (F0) for testing, Standard (S0) for production
6. Click "Review + create" then "Create"
7. Wait for deployment to complete
8. Go to the resource and copy the **Key 1** from "Keys and Endpoint"

### 2. Configure Environment Variables

Add these to your `.env` file:

```env
# Azure TTS Configuration
AZURE_TTS_SUBSCRIPTION_KEY=your_subscription_key_here
AZURE_TTS_REGION=eastus
AZURE_TTS_VOICE=en-US-AdamMultilingualNeural
AZURE_TTS_SPEECH_RATE=+0%
AZURE_TTS_PITCH=+0%
AZURE_TTS_VOLUME=+0%
```

### 3. Test Configuration

Run the test script to verify your setup:

```bash
python test_azure_tts.py
```

### 4. Start DanzarVLM

```bash
python DanzarVLM.py
```

## 🎤 Available Voices

Azure TTS offers many high-quality voices. Here are some popular options:

### English (US) Neural Voices
- `en-US-AdamMultilingualNeural` - Male, cheerful, multilingual
- `en-US-JennyMultilingualNeural` - Female, friendly, multilingual
- `en-US-GuyNeural` - Male, professional
- `en-US-AriaNeural` - Female, professional
- `en-US-DavisNeural` - Male, warm
- `en-US-JasonNeural` - Male, calm
- `en-US-SaraNeural` - Female, warm
- `en-US-TonyNeural` - Male, energetic

### Other Languages
- `en-GB-RyanNeural` - British English, male
- `en-GB-SoniaNeural` - British English, female
- `es-ES-ElviraNeural` - Spanish, female
- `fr-FR-DeniseNeural` - French, female
- `de-DE-KatjaNeural` - German, female

## 💰 Pricing

### Free Tier (F0)
- **500,000 characters per month**
- Perfect for testing and light usage
- No credit card required

### Standard Tier (S0)
- **$16.00 per 1 million characters**
- Pay-as-you-go pricing
- No monthly commitment

### Cost Estimation
- **1 hour of speech** ≈ 9,000 characters
- **Free tier** ≈ 55 hours of speech per month
- **$1** ≈ 62,500 characters ≈ 7 hours of speech

## 🔧 Configuration Options

### Speech Rate
- `+0%` - Normal speed
- `+10%` - 10% faster
- `-10%` - 10% slower
- Range: `-50%` to `+50%`

### Pitch
- `+0%` - Normal pitch
- `+10%` - Higher pitch
- `-10%` - Lower pitch
- Range: `-50%` to `+50%`

### Volume
- `+0%` - Normal volume
- `+10%` - Louder
- `-10%` - Quieter
- Range: `-50%` to `+50%`

## 🎯 Voice Styles

Azure TTS supports different speaking styles for some voices:

### Available Styles
- `cheerful` - Happy and enthusiastic
- `sad` - Melancholic
- `angry` - Irritated
- `fearful` - Nervous
- `disgruntled` - Unhappy
- `serious` - Professional
- `friendly` - Warm and approachable
- `hopeful` - Optimistic
- `shouting` - Loud and excited
- `terrified` - Scared
- `unfriendly` - Cold
- `whispering` - Quiet and secretive

### Example Configuration
```env
AZURE_TTS_VOICE=en-US-AdamMultilingualNeural
# Style is set in the SSML: <mstts:express-as style="cheerful">
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Subscription key not configured"
- Check that `AZURE_TTS_SUBSCRIPTION_KEY` is set in your `.env` file
- Verify the key is correct in Azure Portal

#### 2. "Azure TTS connection test failed"
- Check your internet connection
- Verify the region matches your Azure Speech Service
- Ensure the subscription key is valid

#### 3. "Voice not found"
- Check the voice name spelling
- Verify the voice is available in your region
- Try a different voice from the list above

#### 4. "Rate limit exceeded"
- You've exceeded the free tier limit (500K characters/month)
- Upgrade to Standard tier or wait for next month

### Testing Commands

Test TTS functionality in Discord:
```
!tts test
```

Check TTS status:
```
!tts status
```

## 🆚 Comparison: Azure TTS vs Chatterbox

| Feature | Azure TTS | Chatterbox TTS |
|---------|-----------|----------------|
| **Quality** | ⭐⭐⭐⭐⭐ Professional | ⭐⭐⭐⭐ Good |
| **Reliability** | ⭐⭐⭐⭐⭐ 99.9% uptime | ⭐⭐⭐ Self-hosted |
| **Setup** | ⭐⭐⭐⭐ Easy | ⭐⭐ Complex |
| **Cost** | ⭐⭐⭐ Pay-per-use | ⭐⭐⭐⭐⭐ Free |
| **Voices** | ⭐⭐⭐⭐⭐ 400+ voices | ⭐⭐⭐ Limited |
| **Languages** | ⭐⭐⭐⭐⭐ 140+ languages | ⭐⭐ English only |
| **Customization** | ⭐⭐⭐⭐⭐ SSML, styles | ⭐⭐⭐ Basic |

## 🎉 Benefits of Azure TTS

1. **Professional Quality**: Enterprise-grade TTS used by major companies
2. **High Reliability**: 99.9% uptime SLA
3. **Multiple Voices**: 400+ neural voices across 140+ languages
4. **Easy Setup**: No local GPU requirements
5. **Scalable**: Pay only for what you use
6. **SSML Support**: Advanced text-to-speech markup
7. **Voice Styles**: Emotional and contextual speaking styles
8. **Multilingual**: Support for many languages and accents

## 🔄 Migration from Chatterbox

The migration is automatic! DanzarVLM will:

1. Try to initialize Azure TTS first
2. Fall back to the default TTS service if Azure fails
3. Log which service is being used
4. Use the same interface for both services

No code changes needed - just configure the environment variables and restart DanzarVLM.

## 📞 Support

If you need help:

1. Run `python test_azure_tts.py` for diagnostics
2. Check the Azure Speech Service documentation
3. Verify your subscription key and region
4. Test with a different voice

Happy TTS-ing! 🎤✨ 