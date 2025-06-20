# SearxNG Web Search Setup for DanzarAI

## Overview

DanzarAI has been updated to use **SearxNG** instead of DuckDuckGo for web search functionality. This change resolves the rate limiting issues that were preventing the bot from verifying information and performing fact-checking.

## Why SearxNG?

- **No Rate Limits**: Self-hosted solution eliminates external API rate limiting
- **Privacy**: No tracking or data collection  
- **Multiple Search Engines**: Aggregates results from Google, Bing, DuckDuckGo, and others
- **Customizable**: Full control over search engines, categories, and filtering
- **Reliable**: Consistent performance without external dependencies

## Quick Setup (Docker)

1. **Install Docker** if not already installed
2. **Run SearxNG container**:
   ```bash
   docker run -d --name searxng -p 8080:8080 searxng/searxng:latest
   ```
3. **Verify it's working**: Visit `http://localhost:8080`
4. **Start DanzarAI**: The bot will automatically use SearxNG for web searches

## Configuration

The bot is configured in `config/global_settings.yaml`:

```yaml
# SearxNG Web Search Configuration
SEARXNG_SETTINGS:
  enabled: true
  endpoint: http://localhost:8080
  timeout: 30
  max_results: 6
  categories: general
  engines: google,bing,duckduckgo
  format: json
  safesearch: 1
```

## Benefits Over DuckDuckGo

| Feature | DuckDuckGo | SearxNG |
|---------|------------|---------|
| Rate Limits | ❌ Strict limits | ✅ No limits |
| Privacy | ✅ Good | ✅ Excellent |
| Search Engines | ❌ Single source | ✅ Multiple sources |
| Customization | ❌ Limited | ✅ Full control |
| Reliability | ❌ External dependency | ✅ Self-hosted |

## Troubleshooting

- **No search results**: Check if SearxNG is running at `http://localhost:8080`
- **Connection errors**: Verify the endpoint in `global_settings.yaml`
- **Slow responses**: Increase the timeout value in configuration
