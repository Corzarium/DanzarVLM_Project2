"""
DanzarVLM Research Mode - Web Crawler and RAG Storage System

This module implements a comprehensive research crawler that:
1. Crawls websites recursively following internal links
2. Extracts and cleans text content 
3. Generates embeddings using local Ollama
4. Stores in RAG vector database with resume support
5. Integrates with Discord commands

Usage:
    !danzarresearch <url> Category <category_name>
    !danzarresume <category>
"""

import asyncio
import json
import logging
import os
import re
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass, asdict

import aiohttp
import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import tiktoken


@dataclass
class CrawlCheckpoint:
    """Checkpoint data for resuming crawls"""
    category: str
    start_url: str
    visited_urls: Set[str]
    pending_urls: List[Tuple[str, int]]  # (url, depth)
    max_depth: int
    max_pages: Optional[int]
    start_time: float
    last_update: float
    pages_crawled: int
    docs_created: int
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        data = asdict(self)
        data['visited_urls'] = list(self.visited_urls)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CrawlCheckpoint':
        """Create from JSON dict"""
        data['visited_urls'] = set(data['visited_urls'])
        data['pending_urls'] = [tuple(item) for item in data['pending_urls']]
        return cls(**data)


class DanzarResearcher:
    """Main research crawler and RAG storage system"""
    
    def __init__(self, app_context=None):
        self.app_context = app_context
        self.logger = logging.getLogger("DanzarVLM.DanzarResearcher")
        
        # Configuration - Updated for LM Studio
        self.lm_studio_endpoint = "http://192.168.0.102:1234"  # LM Studio endpoint
        self.embedding_model = "text-embedding-nomic-embed-text-v1.5"  # LM Studio embedding model
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.chunk_size = 1024  # tokens
        self.overlap_size = 128  # tokens
        self.rate_limit_delay = 1.0  # seconds between requests
        self.request_timeout = 30  # seconds
        self.max_retries = 3
        
        # Initialize components
        self.qdrant_client = None
        self.session = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Checkpoint management
        self.checkpoint_dir = Path("./research_checkpoints")
        self.export_dir = Path("./research_exports")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.export_dir.mkdir(exist_ok=True)
        
    async def initialize(self) -> bool:
        """Initialize the researcher with required connections"""
        try:
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port
            )
            
            # Test Qdrant connection
            collections = self.qdrant_client.get_collections()
            self.logger.info(f"[DanzarResearcher] Connected to Qdrant with {len(collections.collections)} collections")
            
            # Initialize HTTP session
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=2)
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'DanzarVLM-Researcher/1.0'}
            )
            
            # Test LM Studio connection
            if await self._test_lm_studio_connection():
                self.logger.info("[DanzarResearcher] Successfully initialized all connections")
                return True
            else:
                self.logger.error("[DanzarResearcher] Failed to connect to LM Studio")
                return False
                
        except Exception as e:
            self.logger.error(f"[DanzarResearcher] Initialization failed: {e}")
            return False
    
    async def _test_lm_studio_connection(self) -> bool:
        """Test connection to LM Studio embedding endpoint"""
        try:
            # Test with LM Studio's OpenAI-compatible API
            async with self.session.post(
                f"{self.lm_studio_endpoint}/v1/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": "test connection"
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and len(data['data']) > 0:
                        self.logger.info(f"[DanzarResearcher] LM Studio connection successful, embedding dimension: {len(data['data'][0]['embedding'])}")
                        return True
                    else:
                        self.logger.error(f"[DanzarResearcher] LM Studio returned unexpected format: {data}")
                        return False
                else:
                    self.logger.error(f"[DanzarResearcher] LM Studio responded with status {response.status}")
                    return False
        except Exception as e:
            self.logger.error(f"[DanzarResearcher] LM Studio connection test failed: {e}")
            return False
    
    async def start_research(self, url: str, category_name: str, max_depth: int = 3, max_pages: Optional[int] = None) -> bool:
        """
        Start a new research crawl
        
        Args:
            url: Starting URL to crawl
            category_name: Category name (used as collection name)
            max_depth: Maximum depth to crawl (default 3)
            max_pages: Maximum pages to crawl (optional limit)
        
        Returns:
            bool: True if crawl completed successfully
        """
        self.logger.info(f"[Danzar Research] Starting research crawl for category: {category_name}")
        self.logger.info(f"[Danzar Research] Start URL: {url}")
        self.logger.info(f"[Danzar Research] Max depth: {max_depth}, Max pages: {max_pages}")
        
        # Validate URL
        if not self._is_valid_url(url):
            self.logger.error(f"[Danzar Research] Invalid URL: {url}")
            return False
        
        # Check robots.txt
        if not await self._check_robots_txt(url):
            self.logger.warning(f"[Danzar Research] Robots.txt disallows crawling: {url}")
            # Continue anyway but log the warning
        
        # Initialize checkpoint
        checkpoint = CrawlCheckpoint(
            category=category_name,
            start_url=url,
            visited_urls=set(),
            pending_urls=[(url, 0)],
            max_depth=max_depth,
            max_pages=max_pages,
            start_time=time.time(),
            last_update=time.time(),
            pages_crawled=0,
            docs_created=0
        )
        
        return await self._run_crawl(checkpoint)
    
    async def resume_research(self, category_name: str) -> bool:
        """
        Resume a research crawl from checkpoint
        
        Args:
            category_name: Category name to resume
            
        Returns:
            bool: True if resume completed successfully
        """
        checkpoint_file = self.checkpoint_dir / f"{category_name}.json"
        
        if not checkpoint_file.exists():
            self.logger.error(f"[Danzar Research] No checkpoint found for category: {category_name}")
            return False
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            checkpoint = CrawlCheckpoint.from_dict(checkpoint_data)
            self.logger.info(f"[Danzar Research] Resuming crawl for category: {category_name}")
            self.logger.info(f"[Danzar Research] Progress: {checkpoint.pages_crawled} pages, {checkpoint.docs_created} docs")
            
            return await self._run_crawl(checkpoint)
            
        except Exception as e:
            self.logger.error(f"[Danzar Research] Failed to load checkpoint: {e}")
            return False
    
    async def _run_crawl(self, checkpoint: CrawlCheckpoint) -> bool:
        """Run the main crawling loop"""
        try:
            # Ensure collection exists
            await self._ensure_collection_exists(checkpoint.category)
            
            while checkpoint.pending_urls and (
                checkpoint.max_pages is None or 
                checkpoint.pages_crawled < checkpoint.max_pages
            ):
                url, depth = checkpoint.pending_urls.pop(0)
                
                if url in checkpoint.visited_urls:
                    continue
                
                if depth > checkpoint.max_depth:
                    continue
                
                # Process the page
                success = await self._process_page(url, depth, checkpoint)
                
                if success:
                    checkpoint.pages_crawled += 1
                    checkpoint.visited_urls.add(url)
                
                # Save checkpoint every 10 pages
                if checkpoint.pages_crawled % 10 == 0:
                    self._save_checkpoint(checkpoint)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
            
            # Final checkpoint save
            self._save_checkpoint(checkpoint)
            
            # Log completion summary
            elapsed = time.time() - checkpoint.start_time
            self.logger.info(f"[Danzar Research] Crawl completed!")
            self.logger.info(f"[Danzar Research] Pages crawled: {checkpoint.pages_crawled} | Docs created: {checkpoint.docs_created} | Elapsed: {elapsed:.1f} seconds")
            
            # Export summary
            await self._export_summary(checkpoint)
            
            return True
            
        except Exception as e:
            self.logger.error(f"[Danzar Research] Crawl failed: {e}")
            return False
        finally:
            if self.session:
                await self.session.close()
    
    async def _process_page(self, url: str, depth: int, checkpoint: CrawlCheckpoint) -> bool:
        """Process a single page: fetch, extract, embed, store"""
        try:
            self.logger.info(f"[Danzar Research] URL visited: {url}")
            
            # Fetch page content
            html_content = await self._fetch_page(url)
            if not html_content:
                return False
            
            # Extract text and metadata
            text_content, title, links = self._extract_content(html_content, url)
            if not text_content or len(text_content.strip()) < 100:
                self.logger.warning(f"[Danzar Research] Insufficient content on page: {url}")
                return False
            
            # Add new links to pending queue
            if depth < checkpoint.max_depth:
                for link in links:
                    if link not in checkpoint.visited_urls and (link, depth + 1) not in checkpoint.pending_urls:
                        checkpoint.pending_urls.append((link, depth + 1))
            
            # Chunk text for embedding
            chunks = self._chunk_text(text_content)
            
            # Generate embeddings and store
            docs_created = 0
            for i, chunk in enumerate(chunks):
                embedding = await self._generate_embedding(chunk)
                if embedding:
                    await self._store_document(
                        text=chunk,
                        embedding=embedding,
                        metadata={
                            'url': url,
                            'title': title,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'crawl_depth': depth,
                            'timestamp': datetime.now().isoformat()
                        },
                        collection_name=checkpoint.category
                    )
                    docs_created += 1
            
            checkpoint.docs_created += docs_created
            
            self.logger.info(f"[Danzar Research] Characters processed: {len(text_content)} | Embeddings created: {docs_created}")
            self.logger.info(f"[Danzar Research] Stored in collection: {checkpoint.category}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"[Danzar Research] Failed to process page {url}: {e}")
            return False
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL with retries"""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return content
                    else:
                        self.logger.warning(f"[Danzar Research] HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"[Danzar Research] Failed to fetch {url} after {self.max_retries} attempts: {e}")
        return None
    
    def _extract_content(self, html_content: str, base_url: str) -> Tuple[str, str, List[str]]:
        """Extract text content, title, and internal links from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else urlparse(base_url).netloc
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
            element.decompose()
        
        # Extract main content (prefer main, article, or body)
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if not main_content:
            main_content = soup
        
        # Extract text
        text = main_content.get_text()
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Extract internal links
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)
            
            # Only follow internal links
            if parsed_url.netloc == base_domain and parsed_url.scheme in ['http', 'https']:
                # Clean URL (remove fragments)
                clean_url = urlunparse((
                    parsed_url.scheme,
                    parsed_url.netloc,
                    parsed_url.path,
                    parsed_url.params,
                    parsed_url.query,
                    ''  # Remove fragment
                ))
                
                if clean_url not in links and self._is_crawlable_url(clean_url):
                    links.append(clean_url)
        
        return text, title, links
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for embedding"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - self.overlap_size
        
        return chunks
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using LM Studio's OpenAI-compatible API"""
        try:
            async with self.session.post(
                f"{self.lm_studio_endpoint}/v1/embeddings",
                json={
                    "model": self.embedding_model,
                    "input": text
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # LM Studio uses OpenAI format: data.data[0].embedding
                    if 'data' in data and len(data['data']) > 0 and 'embedding' in data['data'][0]:
                        return data['data'][0]['embedding']
                    else:
                        self.logger.error(f"[DanzarResearcher] Unexpected embedding response format: {data}")
                        return None
                else:
                    self.logger.error(f"[DanzarResearcher] Embedding API returned status {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"[DanzarResearcher] Failed to generate embedding: {e}")
            return None
    
    async def _ensure_collection_exists(self, collection_name: str):
        """Ensure Qdrant collection exists for the category"""
        try:
            collections = self.qdrant_client.get_collections()
            existing_names = [col.name for col in collections.collections]
            
            if collection_name not in existing_names:
                # Create collection with appropriate vector size (nomic-embed-text is 768 dimensional)
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                self.logger.info(f"[Danzar Research] Created new collection: {collection_name}")
            else:
                self.logger.info(f"[Danzar Research] Using existing collection: {collection_name}")
                
        except Exception as e:
            self.logger.error(f"[Danzar Research] Failed to ensure collection exists: {e}")
            raise
    
    async def _store_document(self, text: str, embedding: List[float], metadata: Dict, collection_name: str):
        """Store document in Qdrant vector database"""
        try:
            import uuid
            # Generate a UUID for the point ID to ensure compatibility with Qdrant
            point_id_str = f"{metadata['url']}_{metadata['chunk_index']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_str))
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'text': text,
                    'url': metadata['url'],
                    'title': metadata['title'],
                    'chunk_index': metadata['chunk_index'],
                    'total_chunks': metadata['total_chunks'],
                    'crawl_depth': metadata['crawl_depth'],
                    'timestamp': metadata['timestamp']
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
        except Exception as e:
            self.logger.error(f"[Danzar Research] Failed to store document: {e}")
            raise
    
    def _save_checkpoint(self, checkpoint: CrawlCheckpoint):
        """Save crawl checkpoint to disk"""
        try:
            checkpoint.last_update = time.time()
            checkpoint_file = self.checkpoint_dir / f"{checkpoint.category}.json"
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"[Danzar Research] Failed to save checkpoint: {e}")
    
    async def _export_summary(self, checkpoint: CrawlCheckpoint):
        """Export crawl summary to file"""
        try:
            export_path = self.export_dir / checkpoint.category
            export_path.mkdir(exist_ok=True)
            
            summary = {
                'category': checkpoint.category,
                'start_url': checkpoint.start_url,
                'pages_crawled': checkpoint.pages_crawled,
                'docs_created': checkpoint.docs_created,
                'max_depth': checkpoint.max_depth,
                'start_time': datetime.fromtimestamp(checkpoint.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(checkpoint.last_update).isoformat(),
                'elapsed_seconds': checkpoint.last_update - checkpoint.start_time,
                'visited_urls': list(checkpoint.visited_urls)
            }
            
            summary_file = export_path / f"crawl_summary_{int(checkpoint.start_time)}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"[Danzar Research] Exported summary to: {summary_file}")
            
        except Exception as e:
            self.logger.error(f"[Danzar Research] Failed to export summary: {e}")
    
    async def _check_robots_txt(self, url: str) -> bool:
        """Check if robots.txt allows crawling"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    
                    rp = urllib.robotparser.RobotFileParser()
                    rp.set_url(robots_url)
                    rp.read()
                    
                    return rp.can_fetch('*', url)
                else:
                    # If no robots.txt, assume allowed
                    return True
                    
        except Exception:
            # If can't check robots.txt, assume allowed
            return True
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _is_crawlable_url(self, url: str) -> bool:
        """Check if URL should be crawled (filter out non-content URLs)"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Skip common non-content extensions
        skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                          '.zip', '.rar', '.tar', '.gz', '.exe', '.dmg', '.iso',
                          '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
                          '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.wav'}
        
        for ext in skip_extensions:
            if path.endswith(ext):
                return False
        
        # Skip common dynamic/spam patterns
        skip_patterns = ['login', 'register', 'cart', 'checkout', 'admin', 'api/', 
                        'ajax', 'search?', 'print=', 'share=', 'download=']
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True


# Discord Integration Functions
async def handle_research_command(app_context, message, args: List[str]) -> str:
    """
    Handle !danzarresearch command from Discord
    
    Expected format: !danzarresearch <url> Category <category_name> [MaxPages <num>]
    """
    try:
        if len(args) < 3:
            return "❌ Usage: `!danzarresearch <url> Category <category_name> [MaxPages <num>]`"
        
        url = args[0]
        
        if args[1].lower() != 'category':
            return "❌ Usage: `!danzarresearch <url> Category <category_name> [MaxPages <num>]`"
        
        category_name = args[2]
        
        # Parse optional MaxPages parameter
        max_pages = None
        if len(args) >= 5 and args[3].lower() == 'maxpages':
            try:
                max_pages = int(args[4])
            except ValueError:
                return "❌ MaxPages must be a number"
        
        # Initialize researcher if not exists
        if not hasattr(app_context, 'researcher') or not app_context.researcher:
            app_context.researcher = DanzarResearcher(app_context)
            if not await app_context.researcher.initialize():
                return "❌ Failed to initialize researcher. Check LM Studio and Qdrant connections."
        
        # Start research in background task
        asyncio.create_task(
            app_context.researcher.start_research(url, category_name, max_pages=max_pages)
        )
        
        return f"🔍 **Research Started!**\n📝 Category: `{category_name}`\n🌐 URL: `{url}`\n📊 Max Pages: `{max_pages or 'unlimited'}`\n\n_Check console for progress updates_"
        
    except Exception as e:
        return f"❌ Research command failed: {str(e)}"


async def handle_resume_command(app_context, message, args: List[str]) -> str:
    """
    Handle !danzarresume command from Discord
    
    Expected format: !danzarresume <category_name>
    """
    try:
        if len(args) != 1:
            return "❌ Usage: `!danzarresume <category_name>`"
        
        category_name = args[0]
        
        # Initialize researcher if not exists
        if not hasattr(app_context, 'researcher') or not app_context.researcher:
            app_context.researcher = DanzarResearcher(app_context)
            if not await app_context.researcher.initialize():
                return "❌ Failed to initialize researcher. Check LM Studio and Qdrant connections."
        
        # Check if checkpoint exists
        checkpoint_file = Path("./research_checkpoints") / f"{category_name}.json"
        if not checkpoint_file.exists():
            return f"❌ No checkpoint found for category: `{category_name}`"
        
        # Resume research in background task
        asyncio.create_task(
            app_context.researcher.resume_research(category_name)
        )
        
        return f"🔄 **Resuming Research!**\n📝 Category: `{category_name}`\n\n_Check console for progress updates_"
        
    except Exception as e:
        return f"❌ Resume command failed: {str(e)}"


# Main interface functions
async def start_research(url: str, category_name: str, max_depth: int = 3, max_pages: Optional[int] = None, app_context=None) -> bool:
    """
    Main entry point for starting research crawl
    
    Args:
        url: Starting URL to crawl
        category_name: Category name (used as collection name)
        max_depth: Maximum depth to crawl (default 3)
        max_pages: Maximum pages to crawl (optional limit)
        app_context: Application context (optional)
    
    Returns:
        bool: True if crawl completed successfully
    """
    researcher = DanzarResearcher(app_context)
    
    if not await researcher.initialize():
        return False
    
    return await researcher.start_research(url, category_name, max_depth, max_pages)


async def resume_research(category_name: str, app_context=None) -> bool:
    """
    Resume a research crawl from checkpoint
    
    Args:
        category_name: Category name to resume
        app_context: Application context (optional)
        
    Returns:
        bool: True if resume completed successfully
    """
    researcher = DanzarResearcher(app_context)
    
    if not await researcher.initialize():
        return False
    
    return await researcher.resume_research(category_name)


if __name__ == "__main__":
    # Example usage
    async def main():
        success = await start_research(
            url="https://everquest.allakhazam.com/wiki/EverQuest",
            category_name="Everquest",
            max_depth=3,
            max_pages=20
        )
        print(f"Research completed: {success}")
    
    asyncio.run(main()) 