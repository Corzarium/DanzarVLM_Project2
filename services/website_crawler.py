#!/usr/bin/env python3
# services/website_crawler.py

import logging
import time
import requests
from typing import Set, List, Dict, Optional, Any
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
import re
from pathlib import Path
from dataclasses import dataclass
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

logger = logging.getLogger("DanzarVLM.WebCrawler")

@dataclass
class CrawledPage:
    url: str
    title: str
    content: str
    links: List[str]
    metadata: Dict[str, Any]
    timestamp: float
    status_code: int
    content_hash: str

class ExhaustiveWebCrawler:
    """
    Exhaustive website crawler that can crawl entire websites until no more pages are found.
    Respects robots.txt, handles rate limiting, and stores results in RAG.
    """
    
    def __init__(self, app_context=None, rag_service=None):
        self.ctx = app_context
        self.rag_service = rag_service
        self.logger = logger
        
        # Crawling configuration
        self.max_pages = 1000  # Safety limit
        self.delay_between_requests = 1.0  # Respect servers
        self.max_workers = 3  # Concurrent threads
        self.timeout = 30  # Request timeout
        
        # Content filtering
        self.min_content_length = 100
        self.max_content_length = 50000
        
        # URL tracking
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.pending_urls: Queue = Queue()
        
        # Results storage
        self.crawled_pages: List[CrawledPage] = []
        self.content_hashes: Set[str] = set()  # Detect duplicates
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DanzarVLM-Crawler/1.0 (Educational Research Bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Robots.txt cache
        self.robots_cache: Dict[str, RobotFileParser] = {}
        
        self.logger.info("[WebCrawler] Initialized exhaustive web crawler")

    def crawl_website(self, base_url: str, collection_name: str = "website_crawl", 
                     max_pages: Optional[int] = None, same_domain_only: bool = True) -> Dict[str, Any]:
        """
        Exhaustively crawl a website until no more pages are found.
        
        Args:
            base_url: Starting URL to crawl
            collection_name: RAG collection to store results
            max_pages: Maximum pages to crawl (None for unlimited)
            same_domain_only: Whether to stay within the same domain
            
        Returns:
            Dict with crawling statistics and results
        """
        start_time = time.time()
        self.max_pages = max_pages or self.max_pages
        
        # Reset state
        self.visited_urls.clear()
        self.failed_urls.clear()
        self.crawled_pages.clear()
        self.content_hashes.clear()
        
        # Clear queue
        while not self.pending_urls.empty():
            try:
                self.pending_urls.get_nowait()
            except:
                break
        
        # Parse base URL
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        
        self.logger.info(f"[WebCrawler] Starting exhaustive crawl of {base_url}")
        self.logger.info(f"[WebCrawler] Max pages: {self.max_pages}, Same domain only: {same_domain_only}")
        
        # Add starting URL
        self.pending_urls.put(base_url)
        
        # Check robots.txt
        if not self._can_fetch(base_url):
            self.logger.warning(f"[WebCrawler] Robots.txt disallows crawling {base_url}")
            return {"error": "Robots.txt disallows crawling", "pages_crawled": 0}
        
        try:
            # Start crawling with thread pool
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                
                while (not self.pending_urls.empty() and 
                       len(self.visited_urls) < self.max_pages):
                    
                    # Get next URL
                    try:
                        current_url = self.pending_urls.get_nowait()
                    except:
                        # No more URLs in queue, check if any futures are still running
                        if not futures:
                            break
                        # Wait for at least one future to complete
                        for future in as_completed(futures, timeout=1):
                            futures.remove(future)
                            try:
                                future.result()
                            except Exception as e:
                                self.logger.warning(f"[WebCrawler] Future failed: {e}")
                            break
                        continue
                    
                    # Skip if already visited
                    if current_url in self.visited_urls or current_url in self.failed_urls:
                        continue
                    
                    # Check domain restriction
                    if same_domain_only:
                        parsed_current = urlparse(current_url)
                        if parsed_current.netloc != base_domain:
                            continue
                    
                    # Check robots.txt
                    if not self._can_fetch(current_url):
                        self.failed_urls.add(current_url)
                        continue
                    
                    # Submit crawling task
                    future = executor.submit(self._crawl_single_page, current_url, base_domain, same_domain_only)
                    futures.append(future)
                    
                    # Clean up completed futures
                    completed_futures = [f for f in futures if f.done()]
                    for future in completed_futures:
                        futures.remove(future)
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.warning(f"[WebCrawler] Future failed: {e}")
                    
                    # Rate limiting
                    time.sleep(self.delay_between_requests)
                
                # Wait for all remaining futures
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.warning(f"[WebCrawler] Final future failed: {e}")
        
        except KeyboardInterrupt:
            self.logger.info("[WebCrawler] Crawling interrupted by user")
        except Exception as e:
            self.logger.error(f"[WebCrawler] Crawling failed: {e}", exc_info=True)
        
        # Store results in RAG
        stored_count = 0
        if self.rag_service and self.crawled_pages:
            stored_count = self._store_in_rag(collection_name)
        
        # Calculate statistics
        end_time = time.time()
        duration = end_time - start_time
        
        stats = {
            "base_url": base_url,
            "pages_crawled": len(self.crawled_pages),
            "pages_failed": len(self.failed_urls),
            "pages_stored": stored_count,
            "unique_content_pieces": len(self.content_hashes),
            "duration_seconds": round(duration, 2),
            "pages_per_second": round(len(self.crawled_pages) / duration, 2) if duration > 0 else 0,
            "same_domain_only": same_domain_only,
            "collection_name": collection_name
        }
        
        self.logger.info(f"[WebCrawler] Crawling complete: {stats}")
        return stats

    def _crawl_single_page(self, url: str, base_domain: str, same_domain_only: bool) -> Optional[CrawledPage]:
        """Crawl a single page and extract content and links."""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        
        try:
            self.logger.debug(f"[WebCrawler] Crawling: {url}")
            
            # Make request
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            
            # Check if it's HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                self.logger.debug(f"[WebCrawler] Skipping non-HTML content: {url}")
                return None
            
            # Parse content if BeautifulSoup is available
            if not BS4_AVAILABLE:
                self.logger.warning("[WebCrawler] BeautifulSoup not available, limited text extraction")
                title = "Unknown"
                content = response.text[:self.max_content_length]
                links = []
            else:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else "No Title"
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract text content
                content = soup.get_text()
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Extract links
                links = []
                for link in soup.find_all('a', href=True):
                    link_url = urljoin(url, link['href'])
                    # Clean and validate URL
                    parsed_link = urlparse(link_url)
                    if parsed_link.scheme in ['http', 'https']:
                        # Remove fragments
                        clean_url = urlunparse((
                            parsed_link.scheme,
                            parsed_link.netloc,
                            parsed_link.path,
                            parsed_link.params,
                            parsed_link.query,
                            ''  # Remove fragment
                        ))
                        links.append(clean_url)
            
            # Filter content length
            if len(content) < self.min_content_length:
                self.logger.debug(f"[WebCrawler] Content too short, skipping: {url}")
                return None
            
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            # Check for duplicate content
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.content_hashes:
                self.logger.debug(f"[WebCrawler] Duplicate content detected, skipping: {url}")
                return None
            
            self.content_hashes.add(content_hash)
            
            # Create crawled page
            page = CrawledPage(
                url=url,
                title=title,
                content=content,
                links=links,
                metadata={
                    "domain": urlparse(url).netloc,
                    "status_code": response.status_code,
                    "content_type": response.headers.get('content-type', ''),
                    "content_length": len(content),
                    "links_found": len(links)
                },
                timestamp=time.time(),
                status_code=response.status_code,
                content_hash=content_hash
            )
            
            self.crawled_pages.append(page)
            
            # Add new links to queue
            for link in links:
                if link not in self.visited_urls and link not in self.failed_urls:
                    # Apply domain restriction
                    if same_domain_only:
                        parsed_link = urlparse(link)
                        if parsed_link.netloc != base_domain:
                            continue
                    
                    self.pending_urls.put(link)
            
            self.logger.debug(f"[WebCrawler] Successfully crawled: {url} ({len(content)} chars, {len(links)} links)")
            return page
            
        except requests.RequestException as e:
            self.logger.warning(f"[WebCrawler] Request failed for {url}: {e}")
            self.failed_urls.add(url)
            return None
        except Exception as e:
            self.logger.error(f"[WebCrawler] Unexpected error crawling {url}: {e}", exc_info=True)
            self.failed_urls.add(url)
            return None

    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            if base_url not in self.robots_cache:
                robots_url = urljoin(base_url, '/robots.txt')
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[base_url] = rp
                except Exception as e:
                    self.logger.debug(f"[WebCrawler] Could not read robots.txt for {base_url}: {e}")
                    # If we can't read robots.txt, assume it's OK to crawl
                    return True
            
            robots = self.robots_cache[base_url]
            return robots.can_fetch(self.session.headers['User-Agent'], url)
            
        except Exception as e:
            self.logger.debug(f"[WebCrawler] Error checking robots.txt for {url}: {e}")
            return True  # Default to allowing crawling

    def _store_in_rag(self, collection_name: str) -> int:
        """Store crawled pages in RAG collection."""
        stored_count = 0
        
        for page in self.crawled_pages:
            try:
                # Format content for storage
                formatted_content = f"""Title: {page.title}
URL: {page.url}
Domain: {page.metadata['domain']}

Content:
{page.content}"""
                
                # Create metadata
                metadata = {
                    "url": page.url,
                    "title": page.title,
                    "domain": page.metadata['domain'],
                    "crawl_timestamp": page.timestamp,
                    "content_hash": page.content_hash,
                    "content_length": len(page.content),
                    "links_found": len(page.links),
                    "type": "website_crawl"
                }
                
                # Store in RAG
                if hasattr(self.rag_service, 'ingest_text'):
                    success = self.rag_service.ingest_text(
                        collection=collection_name,
                        text=formatted_content,
                        metadata=metadata
                    )
                    if success:
                        stored_count += 1
                elif hasattr(self.rag_service, 'add_document'):
                    success = self.rag_service.add_document(
                        text=formatted_content,
                        metadata=metadata,
                        collection=collection_name
                    )
                    if success:
                        stored_count += 1
                        
            except Exception as e:
                self.logger.error(f"[WebCrawler] Failed to store page {page.url}: {e}")
        
        self.logger.info(f"[WebCrawler] Stored {stored_count}/{len(self.crawled_pages)} pages in RAG collection '{collection_name}'")
        return stored_count

    def get_crawl_summary(self) -> Dict[str, Any]:
        """Get summary of the last crawl."""
        return {
            "total_pages": len(self.crawled_pages),
            "failed_pages": len(self.failed_urls),
            "unique_domains": len(set(page.metadata['domain'] for page in self.crawled_pages)),
            "avg_content_length": sum(len(page.content) for page in self.crawled_pages) / len(self.crawled_pages) if self.crawled_pages else 0,
            "total_links_found": sum(len(page.links) for page in self.crawled_pages),
            "pages_by_domain": {domain: len([p for p in self.crawled_pages if p.metadata['domain'] == domain]) 
                              for domain in set(page.metadata['domain'] for page in self.crawled_pages)}
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
        self.logger.info("[WebCrawler] Cleanup completed")
