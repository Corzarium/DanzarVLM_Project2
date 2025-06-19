                if hasattr(self.ctx, 'text_message_queue'):
                    self.ctx.text_message_queue.put_nowait(f"ðŸ’¬ {sentence}")
            except Exception as e:
                self.logger.error(f"[LLMService] Streaming text callback error: {e}")
        
        return text_callback

    def _run_image_visibility_diagnostic(self, base64_image: str, image_formats: dict):
        """Special diagnostic function to test if the model can see the image in any format"""
        self.logger.info("[LLMService] Running deep image visibility diagnostic with all formats")
        profile = self.ctx.active_profile
        gs = self.ctx.global_settings
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug_vlm_frames")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save diagnostic info
        diag_path = os.path.join(debug_dir, f"diagnostic_test_{int(time.time())}.txt")
        with open(diag_path, 'w') as f:
            f.write(f"VLM IMAGE VISIBILITY DIAGNOSTIC\n")
            f.write(f"==============================\n")
            f.write(f"Model: {profile.vlm_model}\n")
            f.write(f"Provider: {gs.get('VLM_PROVIDER', 'unknown')}\n")
            f.write(f"Base64 length: {len(base64_image)} chars\n")
            f.write(f"Available formats: {list(image_formats.keys())}\n")
            f.write(f"==============================\n\n")

        # Run tests for each format
        results = {}
        for format_name, format_content in image_formats.items():
            try:
                response = self._test_image_format(
                    format_name=format_name,
                    base64_image=base64_image,
                    format_content=format_content,
                    profile=profile,
                    diag_path=diag_path
                )
                if response and "success" in response:
                    self.logger.info(f"[LLMService] Found working format: {format_name}")
                    self.ctx.global_settings["VLM_IMAGE_FORMAT"] = format_name
                    break
            except Exception as e:
                self.logger.error(f"Format test failed for {format_name}: {e}")
                continue

        self.logger.info(f"[LLMService] Diagnostic completed. Results saved to {diag_path}")
        return results

    def _test_image_format(self, format_name: str, base64_image: str, format_content: str, profile, diag_path: str):
        """Helper method to test a single image format"""
        self.logger.info(f"[LLMService] Testing format: {format_name}")
        
        diagnostic_prompt = "This is an image visibility test. Can you see and describe the image shown?"
        vlm_provider = self.ctx.global_settings.get("VLM_PROVIDER", "default").lower()
        
        messages = [{"role": "user", "content": diagnostic_prompt}]
        images_payload = []

        # Configure the payload based on provider
        if vlm_provider == "ollama":
            images_payload = [base64_image]
        elif "qwen" in profile.vlm_model.lower():
            messages = [{"role": "user", "content": f"USER: <img>{base64_image}</img>\n\n{diagnostic_prompt}\n\nASSISTANT:"}]
        elif "llama" in profile.vlm_model.lower():
            messages = [{"role": "user", "content": f"<img src=\"data:image/jpeg;base64,{base64_image}\">\n\n{diagnostic_prompt}"}]
        else:
            messages = [{"role": "user", "content": f"data:image/jpeg;base64,{base64_image}\n\n{diagnostic_prompt}"}]

        payload = {
            "model": profile.vlm_model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.2,
            "stream": False
        }
        if images_payload:
            payload["images"] = images_payload

        # Log diagnostic info
        with open(diag_path, 'a') as f:
            f.write(f"\nTesting Format: {format_name}\n")
            f.write(f"Payload Summary:\n{json.dumps(payload, indent=2)[:500]}...\n")

        # Make API call
        try:
            response = self.model_client.generate(
                messages=messages,
                temperature=0.2,
                max_tokens=300,
                model=profile.vlm_model,
                endpoint="chat/completions",
                images=images_payload if images_payload else None
            )
            
            if not response:
                return {"error": "Empty response"}

            # Log response
            with open(diag_path, 'a') as f:
                f.write(f"Response:\n{response[:1000]}\n")
                f.write("="*50 + "\n")

            # Check if model can see image
            if "i cannot see any image" not in response.lower() and "i can't see any image" not in response.lower():
                return {"success": True, "response": response}
            
            return {"success": False, "response": response}

        except Exception as e:
            self.logger.error(f"Error testing format {format_name}: {e}")
            return {"error": str(e)}

    def get_response(self, user: str, game: str, query: str) -> str:
        """
        1) Run the user's query against RAG.
        2) If no docs returned, fallback immediately.
        3) Otherwise, build a grounded prompt and generate with temp=0.
        """
        self.logger.info(f"[LLMService] Fact-checking query via RAG: '{query}'")
        # Step 1: retrieve top 5 passages
        docs = self.rag_service.query(collection=self.default_collection,
                              query_text=query,
                              n_results=5)

        # Step 2: fallback if nothing to ground on
        if not docs:
            fallback = ("I'm not certain about that. I couldn't find any reference "
                        "in my knowledge baseâ€”would you like me to search deeper?")
            self.logger.warning("[LLMService] No RAG hits, returning fallback.")
            return fallback

        # Step 3: build a grounded prompt
        context_block = "\n\n--- Retrieved Context ---\n" + "\n\n".join(docs)
        system_prompt = (
            "You are DanzarAI, a knowledgeable EverQuest assistant. "
            "Answer the user's question **using only** the information in the retrieved context. "
            "If the answer is not contained there, say \"I don't know.\""
        )
        full_prompt = f"{system_prompt}\n\n{context_block}\n\nUser: {query}\nAssistant:"

        # Get max_tokens from profile or use a larger default
        max_tokens = getattr(self.ctx.active_profile, 'conversational_max_tokens', 1024)
        self.logger.info(f"[LLMService] Generating grounded response (temp=0, max_tokens={max_tokens}).")
        
        try:
            if self.model_client:
                answer = self.model_client.generate(
                    prompt=full_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens
                )
            else:
                # Fallback to using the existing _call_llm_api method
                payload = {
                    "model": self.ctx.active_profile.conversational_llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{context_block}\n\nUser: {query}"}
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens
                }
                resp = self._call_llm_api(payload)
                if resp and "choices" in resp and resp["choices"]:
                    answer = resp["choices"][0].get("message", {}).get("content", "").strip()
                else:
                    raise Exception("Empty or invalid response from LLM API")
        except Exception as e:
            self.logger.error(f"[LLMService] LLM generation failed: {e}", exc_info=True)
            return "Sorry, I ran into an error trying to think that through."

        # Step 4: final sanity check for "I don't know" enforcement
        if answer.strip() == "":
            return "I'm not sure; there's no info in my sources."

        return answer.strip()

    async def process_gaming_message(self, message: str, user_name: str = "User") -> str:
        """Process a gaming message with context and return response"""
        self.logger.info(f"[LLM:DEBUG] === Processing gaming message from {user_name} ===")
        self.logger.info(f"[LLM:DEBUG] Message: '{message}'")
        
        try:
            # Use Smart RAG for contextual response generation
            if self.smart_rag:
                self.logger.info(f"[LLM:DEBUG] Using Smart RAG service for response generation")
                response, metadata = self.smart_rag.smart_generate_response(message, user_name)
                
                self.logger.info(f"[LLM:DEBUG] Smart RAG response metadata: {metadata}")
                self.logger.info(f"[LLM:DEBUG] Smart RAG response length: {len(response)} chars")
                self.logger.info(f"[LLM:DEBUG] Smart RAG response preview: {response[:200]}...")
                
                return response
            else:
                self.logger.warning(f"[LLM:DEBUG] No Smart RAG service available, using direct LLM")
                # Fallback to direct LLM without context
                return await self._generate_direct_response(message)
                
        except Exception as e:
            self.logger.error(f"[LLM:DEBUG] Gaming message processing failed: {e}")
            import traceback
            self.logger.error(f"[LLM:DEBUG] Full traceback: {traceback.format_exc()}")
            return "I'm having trouble processing that request right now."

    async def _generate_direct_response(self, message: str) -> str:
        """Generate a direct response using the LLM without RAG context"""
        try:
            if not self.model_client:
                return "LLM service is not available right now."
            
            messages = [
                {"role": "system", "content": "You are a helpful gaming assistant."},
                {"role": "user", "content": message}
            ]
            
            profile = self.ctx.active_profile
            response = self.model_client.generate(
                messages=messages,
                temperature=float(profile.conversational_temperature),
                max_tokens=int(profile.conversational_max_tokens),
                model=profile.conversational_llm_model
            )
            
            return response if response else "I'm unable to generate a response right now."
            
        except Exception as e:
            self.logger.error(f"[LLM:DEBUG] Direct response generation failed: {e}")
            return "I'm having trouble processing that request right now."

    async def _handle_crawl_website_request(self, user_text: str, user_name: str) -> str:
        """
        Handle website crawling requests.
        
        Args:
            user_text: The user's crawl request
            user_name: Name of the user making the request
            
        Returns:
            Response about the crawling operation
        """
        try:
            # Extract URL from the request
            import re
            url_pattern = r'(https?://[^\s]+)'
            urls = re.findall(url_pattern, user_text)
            
            if not urls:
                return "I need a valid URL to crawl. Please provide a URL starting with http:// or https://"
            
            target_url = urls[0]
            
            # Check if exhaustive crawler is available
            if not self.ctx.website_crawler_instance:
                return "Website crawler is not available. Please ensure the crawler service is initialized."
            
            # Extract max pages from request if specified
            max_pages = 50  # Default limit
            max_pattern = r'(?:max|limit|up to)\s*(\d+)\s*pages?'
            max_match = re.search(max_pattern, user_text.lower())
            if max_match:
                max_pages = min(int(max_match.group(1)), 200)  # Cap at 200 for safety
            
            self.logger.info(f"[LLMService] Starting exhaustive crawl of {target_url} (max {max_pages} pages)")
            
            # Determine collection name
            from urllib.parse import urlparse
            parsed_url = urlparse(target_url)
            domain = parsed_url.netloc.replace('.', '_').replace('-', '_')
            collection_name = f"crawl_{domain}_{int(time.time())}"
            
            # Perform the crawl
            crawler = self.ctx.website_crawler_instance
            crawl_results = crawler.crawl_website(
                base_url=target_url,
                collection_name=collection_name,
                max_pages=max_pages,
                same_domain_only=True
            )
            
            # Format response
            if "error" in crawl_results:
                return f"âŒ Crawl failed: {crawl_results['error']}"
            
            response_parts = [
                f"ðŸ•·ï¸ **Website Crawl Complete** for {target_url}",
                f"",
                f"**Results:**",
                f"ðŸ“„ Pages crawled: {crawl_results['pages_crawled']}",
                f"âŒ Pages failed: {crawl_results['pages_failed']}",
                f"ðŸ’¾ Pages stored in RAG: {crawl_results['pages_stored']}",
                f"ðŸ”§ Unique content pieces: {crawl_results['unique_content_pieces']}",
                f"â±ï¸ Duration: {crawl_results['duration_seconds']} seconds",
                f"ðŸ“Š Crawl speed: {crawl_results['pages_per_second']} pages/sec",
                f"",
                f"**Collection:** `{collection_name}`"
            ]
            
            # Add sample page info if available
            if crawler.crawled_pages:
                sample_page = crawler.crawled_pages[0]
                response_parts.extend([
                    f"",
                    f"**Sample Page:**",
                    f"ðŸ”— URL: {sample_page.url}",
                    f"ðŸ“ Title: {sample_page.title}",
                    f"ðŸ“ Content: {len(sample_page.content)} characters",
                    f"ðŸ”— Links found: {len(sample_page.links)}"
                ])
            
            # Store crawl summary in memory
            summary_text = f"Website crawl of {target_url} completed. Crawled {crawl_results['pages_crawled']} pages and stored them in collection '{collection_name}'. You can now search this crawled content for information."
            
            crawl_memory = MemoryEntry(
                content=f"User ({user_name}): {user_text}\nAI (DanzarVLM): {summary_text}",
                source="website_crawl",
                timestamp=time.time(),
                metadata={
                    "user": user_name,
                    "crawl_target": target_url,
                    "pages_crawled": crawl_results['pages_crawled'],
                    "collection": collection_name,
                    "type": "crawl_operation"
                },
                importance_score=2.0  # Higher importance for crawl operations
            )
            
            self.memory_service.store_memory(crawl_memory)
            
            return "\n".join(response_parts)
            
        except Exception as e:
            self.logger.error(f"[LLMService] Website crawl failed: {e}", exc_info=True)
            return f"âŒ Website crawling failed: {str(e)}"

    async def handle_user_text_query_with_llm_search(self, user_text: str, user_name: str = "User") -> tuple:
        """
        Enhanced query handling that allows the LLM to admit when it doesn't know something
        and formulate its own search queries to find the answer
        """
        try:
            # Step 1: Ask the LLM if it knows the answer
            initial_prompt = f"""You are DanzarAI, an EverQuest gaming assistant. A user asked: "{user_text}"

If you know the answer from your training data, provide a helpful response.

If you don't know the answer or are uncertain, respond with exactly: "I don't know - let me search for that information."

Your response:"""

            initial_response = self.model_client.generate(
                messages=[{"role": "user", "content": initial_prompt}],
                temperature=0.3,
                max_tokens=200
            )

            # Step 2: Check if LLM admits it doesn't know
            if initial_response and "I don't know" in initial_response:
                self.logger.info(f"[LLMService] LLM admits it doesn't know, proceeding with search")
                
                # Step 3: Ask LLM to formulate search queries
                search_prompt = f"""The user asked: "{user_text}"

You don't know the answer. Generate 3 specific search queries that would help find authoritative information to answer this question.

Focus on:
1. Official sources (wikis, guides, documentation)
2. Simple, direct search terms
3. Different aspects of the question

Format as a numbered list:
1. [search query 1]
2. [search query 2]
3. [search query 3]

Search queries:"""

                search_response = self.model_client.generate(
                    messages=[{"role": "user", "content": search_prompt}],
                    temperature=0.3,
                    max_tokens=150
                )

                if search_response:
                    # Parse search queries
                    search_queries = self._parse_search_queries(search_response)
                    
                    if search_queries:
                        self.logger.info(f"[LLMService] LLM generated {len(search_queries)} search queries")
                        
                        # Step 4: Perform searches
                        search_results = await self._perform_llm_guided_searches(search_queries)
                        
                        if search_results:
                            # Step 5: Ask LLM to synthesize the search results
                            synthesis_prompt = f"""The user asked: "{user_text}"

I searched for information and found these results:

{search_results}

Based on this information, provide a comprehensive answer to the user's question. If the information is insufficient, say so honestly.

Your response:"""

                            final_response = self.model_client.generate(
                                messages=[{"role": "user", "content": synthesis_prompt}],
                                temperature=0.3,
                                max_tokens=300
                            )

                            if final_response:
                                metadata = {
                                    "method": "llm_guided_search",
                                    "search_queries": search_queries,
                                    "search_results_found": True,
                                    "llm_admitted_ignorance": True
                                }
                                return final_response, metadata
                
                # Fallback to regular RAG if search fails
                self.logger.info("[LLMService] LLM search failed, falling back to regular RAG")
                return await self.handle_user_text_query(user_text, user_name)
            
            else:
                # LLM thinks it knows the answer
                if initial_response:
                    metadata = {
                        "method": "llm_direct_knowledge",
                        "llm_admitted_ignorance": False
                    }
                    return initial_response, metadata
                else:
                    # Fallback to regular RAG
                    return await self.handle_user_text_query(user_text, user_name)

        except Exception as e:
            self.logger.error(f"[LLMService] Error in LLM-guided search: {e}")
            # Fallback to regular RAG
            return await self.handle_user_text_query(user_text, user_name)

    def _parse_search_queries(self, response: str) -> List[str]:
        """Parse LLM response to extract search queries"""
        try:
            queries = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Look for numbered list items
                if re.match(r'^\d+\.?\s*', line):
                    query = re.sub(r'^\d+\.?\s*', '', line).strip()
                    if query and len(query) > 5:
                        # Remove any quotes or brackets
                        query = query.strip('"\'[]')
                        queries.append(query)
            
            return queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            self.logger.error(f"[LLMService] Error parsing search queries: {e}")
            return []

    async def _perform_llm_guided_searches(self, search_queries: List[str]) -> str:
        """Perform searches using LLM-generated queries"""
        try:
            all_results = []
            
            # First try RAG searches
            for query in search_queries:
                try:
                    if self.app_context.rag_service_instance:
                        rag_results = self.app_context.rag_service_instance.query(query, limit=3)
                        if rag_results:
                            for result in rag_results:
                                all_results.append(f"RAG Result: {result.get('text', '')[:200]}...")
                except Exception as e:
                    self.logger.warning(f"[LLMService] RAG search failed for '{query}': {e}")
            
            # Then try web searches if available
            if hasattr(self.app_context, 'fact_check_service') and self.app_context.fact_check_service:
                for query in search_queries[:2]:  # Limit web searches
                    try:
                        web_result = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda q=query: self.app_context.fact_check_service._search_web(q, fact_check=True)
                        )
                        if web_result and len(web_result.strip()) > 50:
                            all_results.append(f"Web Result: {web_result[:200]}...")
                    except Exception as e:
                        self.logger.warning(f"[LLMService] Web search failed for '{query}': {e}")
            
            if all_results:
                return "\n\n".join(all_results[:5])  # Limit to 5 results
            else:
                return "No relevant information found."
                
        except Exception as e:
            self.logger.error(f"[LLMService] Error performing LLM-guided searches: {e}")
            return "Search failed."
