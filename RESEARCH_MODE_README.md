# DanzarVLM Research Mode 🔍

**A comprehensive web crawler and RAG storage system for DanzarVLM that automatically extracts, embeds, and stores knowledge from websites.**

## Overview

Research Mode enables DanzarVLM to crawl websites, extract content, generate embeddings, and store the knowledge in a vector database for later retrieval. This allows the AI to access and reference accurate, up-to-date information from specific sources during conversations.

## Features ✨

### 🧵 **Discord Integration**
- `!danzar research <url> Category <name> [MaxPages <num>]` - Start new research
- `!danzar resume <category>` - Resume from checkpoint

### 🌐 **Smart Web Crawler**
- Recursive crawling following internal links (max depth 3)
- Respects robots.txt (with logging)
- Filters out non-content URLs (PDFs, images, admin pages, etc.)
- Rate limiting (1-2 requests/second)
- Retry logic with exponential backoff

### ✂️ **Content Processing**
- Extracts text from HTML using BeautifulSoup
- Removes navigation, ads, headers/footers
- Normalizes whitespace and cleans text
- Chunks into 1024-token segments with 128-token overlap

### 🧠 **AI-Powered Embeddings**
- Uses local Ollama `nomic-embed-text:latest` model
- Generates 768-dimensional embeddings
- Stores in Qdrant vector database with metadata

### 🔁 **Resume & Checkpoint System**
- Automatic checkpoint saving every 10 pages
- Resume interrupted crawls from last saved state
- Persistent storage of visited URLs and progress

### 📊 **Progress Monitoring**
- Real-time console feedback per page
- Final summary with statistics
- Export crawl summaries to JSON

## Installation

### Prerequisites
- **Ollama** with `nomic-embed-text:latest` model
- **Qdrant** vector database running on localhost:6333
- **Python packages**: `aiohttp`, `beautifulsoup4`, `qdrant-client`, `tiktoken`

### Setup Steps

1. **Install Ollama model:**
   ```bash
   ollama pull nomic-embed-text:latest
   ```

2. **Start Qdrant:**
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Install Python dependencies:**
   ```bash
   pip install aiohttp beautifulsoup4 qdrant-client tiktoken
   ```

## Usage

### Discord Commands

#### Start Research
```
!danzar research <url> Category <category_name> [MaxPages <num>]
```

**Examples:**
```
!danzar research https://everquest.allakhazam.com/wiki/EverQuest Category Everquest
!danzar research https://docs.python.org/ Category Python MaxPages 50
!danzar research https://www.gamedev.net/ Category GameDev MaxPages 100
```

#### Resume Research
```
!danzar resume <category_name>
```

**Example:**
```
!danzar resume Everquest
```

### Programmatic Usage

```python
import asyncio
from danzar_researcher import start_research, resume_research

# Start new research
success = await start_research(
    url="https://example.com",
    category_name="ExampleSite",
    max_depth=3,
    max_pages=100
)

# Resume from checkpoint
success = await resume_research("ExampleSite")
```

## Configuration

The researcher automatically configures itself with sensible defaults:

```python
# Default Configuration
ollama_endpoint = "http://localhost:11434"
embedding_model = "nomic-embed-text:latest"
qdrant_host = "localhost"
qdrant_port = 6333
chunk_size = 1024  # tokens
overlap_size = 128  # tokens
rate_limit_delay = 1.0  # seconds between requests
max_retries = 3
```

## Output Files

### Checkpoints
- **Location**: `./research_checkpoints/<category>.json`
- **Contains**: Progress data, visited URLs, pending URLs, statistics
- **Purpose**: Resume interrupted crawls

### Export Summaries
- **Location**: `./research_exports/<category>/crawl_summary_<timestamp>.json`
- **Contains**: Final statistics, URL list, timing information
- **Purpose**: Analysis and documentation

## Console Output Examples

### During Crawling
```
[Danzar Research] URL visited: https://example.com/page1
[Danzar Research] Characters processed: 2,341 | Embeddings created: 3
[Danzar Research] Stored in collection: example_site
```

### Completion Summary
```
[Danzar Research] Crawl completed!
[Danzar Research] Pages crawled: 47 | Docs created: 156 | Elapsed: 289.3 seconds
```

## RAG Integration

Research data integrates seamlessly with DanzarVLM's existing RAG system:

1. **Storage**: Documents stored in Qdrant collections named by category
2. **Retrieval**: SmartRAGService searches relevant collections during conversations
3. **Metadata**: Each document includes URL, title, timestamp, and chunk information

### Example Vector Database Structure
```json
{
  "id": "uuid-string",
  "vector": [768-dimensional embedding],
  "payload": {
    "text": "Extracted content chunk...",
    "url": "https://source.com/page",
    "title": "Page Title",
    "chunk_index": 0,
    "total_chunks": 5,
    "crawl_depth": 2,
    "timestamp": "2025-06-09T17:58:13.238630"
  }
}
```

## Troubleshooting

### Common Issues

**❌ "Failed to initialize researcher"**
- Check Ollama is running: `ollama list`
- Check Qdrant is accessible: `curl http://localhost:6333/`
- Verify `nomic-embed-text:latest` model is available

**❌ "Robots.txt disallows crawling"**
- Warning only - crawl continues
- Respect website policies
- Consider reaching out to site owners

**❌ "Insufficient content on page"**
- Normal for pages with <100 characters
- Indicates successful filtering of low-content pages

**❌ "Failed to store document"**
- Check Qdrant connection and disk space
- Verify collection creation permissions

### Rate Limiting

The crawler automatically rate limits to 1 request per second. For faster crawling on permissive sites, adjust:

```python
researcher.rate_limit_delay = 0.5  # 500ms between requests
```

### Memory Usage

- **Embeddings**: ~3KB per 1024-token chunk
- **Checkpoints**: ~1KB per 100 visited URLs
- **Qdrant Storage**: ~4KB per document with metadata

## Best Practices

### Choosing Categories
- Use descriptive, consistent names
- Examples: `Everquest`, `Python_Docs`, `GameDev_Tutorials`
- Categories become collection names in Qdrant

### Setting Limits
- **Small sites**: `MaxPages 10-20`
- **Medium sites**: `MaxPages 50-100`
- **Large sites**: `MaxPages 200+`
- **Max depth**: Usually 2-3 is sufficient

### Content Quality
- Choose authoritative, well-structured sites
- Avoid forums/discussion sites (lots of noise)
- Prefer documentation, wikis, and educational content

### Monitoring
- Watch console output for errors
- Check export summaries for completion statistics
- Monitor Qdrant disk usage for large crawls

## Integration with Game Profiles

Research collections can be referenced in game profiles:

```yaml
# config/profiles/everquest.yaml
rag_collection_name: "Everquest"  # Matches research category
```

This allows the AI to automatically search relevant research data when answering game-specific questions.

## Security Considerations

- **Respect robots.txt**: Crawler checks but doesn't strictly enforce
- **Rate limiting**: Default 1 req/sec is conservative
- **Content filtering**: Automatically skips login/admin pages
- **Local storage**: All data stored locally in Qdrant

## Examples

### Research EverQuest Wiki
```
!danzar research https://everquest.allakhazam.com/wiki/EverQuest Category Everquest MaxPages 100
```

Result: 100 pages of EverQuest knowledge embedded and searchable

### Research Python Documentation
```
!danzar research https://docs.python.org/3/ Category Python_Docs MaxPages 200
```

Result: Comprehensive Python documentation for code assistance

### Resume Interrupted Crawl
```
!danzar resume Everquest
```

Result: Continues from last checkpoint, no duplicate work

---

**Need help?** Check the console output for detailed logging, or examine checkpoint files for crawl progress. 