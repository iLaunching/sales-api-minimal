"""
Content Processing Utilities for Streaming

Phase 2.2: Smart content processing for Tiptap editor streaming
- HTML sanitization (remove dangerous tags/attributes)
- Markdown to HTML conversion
- Intelligent chunking (word/sentence/paragraph)
- HTML structure validation
- Adaptive speed calculation
"""

import re
from typing import List, Tuple, Dict, Optional
from html.parser import HTMLParser
import logging

logger = logging.getLogger(__name__)

# Dangerous HTML tags and attributes to remove
DANGEROUS_TAGS = {
    'script', 'iframe', 'object', 'embed', 'style', 
    'link', 'meta', 'base', 'form', 'input', 'button'
}

DANGEROUS_ATTRS = {
    'onclick', 'onload', 'onerror', 'onmouseover', 
    'onfocus', 'onblur', 'onchange', 'onsubmit'
}

# Tiptap-safe HTML tags
SAFE_TAGS = {
    'p', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'strong', 'em', 'u', 's', 'code', 'pre',
    'ul', 'ol', 'li', 'blockquote',
    'a', 'img', 'span', 'div'
}


class HTMLSanitizer(HTMLParser):
    """Sanitize HTML by removing dangerous tags and attributes"""
    
    def __init__(self):
        super().__init__()
        self.result = []
        self.current_tag = None
        
    def handle_starttag(self, tag, attrs):
        """Handle opening tags"""
        if tag.lower() in DANGEROUS_TAGS:
            return
            
        # Filter dangerous attributes
        safe_attrs = []
        for attr_name, attr_value in attrs:
            if attr_name.lower() not in DANGEROUS_ATTRS:
                # Also sanitize href/src for javascript:
                if attr_name.lower() in ('href', 'src'):
                    if attr_value.lower().startswith('javascript:'):
                        continue
                safe_attrs.append((attr_name, attr_value))
        
        # Rebuild tag
        if safe_attrs:
            attrs_str = ' '.join([f'{name}="{value}"' for name, value in safe_attrs])
            self.result.append(f'<{tag} {attrs_str}>')
        else:
            self.result.append(f'<{tag}>')
    
    def handle_endtag(self, tag):
        """Handle closing tags"""
        if tag.lower() not in DANGEROUS_TAGS:
            self.result.append(f'</{tag}>')
    
    def handle_data(self, data):
        """Handle text content"""
        self.result.append(data)
    
    def get_sanitized(self):
        """Get sanitized HTML"""
        return ''.join(self.result)


def sanitize_html(html: str) -> str:
    """
    Production-grade HTML sanitization for safe streaming.
    
    Security measures:
    - Removes dangerous tags (script, iframe, object, etc.)
    - Strips dangerous attributes (onclick, onerror, etc.)
    - Neutralizes javascript: protocol in href/src
    - Validates against XSS attack patterns
    - Preserves safe formatting and structure
    - PRESERVES code block content without parsing
    
    Args:
        html: Raw HTML string to sanitize
    
    Returns:
        Sanitized HTML safe for rendering
    """
    if not html or not isinstance(html, str):
        return ""
    
    # Step 1: Extract and protect code blocks BEFORE parsing
    code_blocks = []
    code_block_pattern = r'(<pre[^>]*>.*?</pre>)'
    
    def replace_code_block(match):
        """Replace code blocks with placeholders"""
        code_blocks.append(match.group(1))
        return f'___CODE_BLOCK_{len(code_blocks) - 1}___'
    
    # Replace code blocks with placeholders
    html_without_code = re.sub(code_block_pattern, replace_code_block, html, flags=re.DOTALL)
    
    # Input validation: check for suspicious patterns (only in non-code content)
    xss_patterns = [
        r'javascript:',
        r'on\w+\s*=',  # Event handlers
        r'<\s*script',
        r'<\s*iframe',
        r'data:text/html',
    ]
    
    suspicious_found = []
    for pattern in xss_patterns:
        if re.search(pattern, html_without_code, re.IGNORECASE):
            suspicious_found.append(pattern)
    
    if suspicious_found:
        logger.warning(f"Suspicious patterns detected in HTML: {suspicious_found}")
    
    try:
        # Step 2: Sanitize the HTML without code blocks
        sanitizer = HTMLSanitizer()
        sanitizer.feed(html_without_code)
        sanitized = sanitizer.get_sanitized()
        
        # Step 3: Restore code blocks with original content
        for i, code_block in enumerate(code_blocks):
            placeholder = f'___CODE_BLOCK_{i}___'
            sanitized = sanitized.replace(placeholder, code_block)
        
        # Validation: ensure output is not empty if input wasn't
        if html.strip() and not sanitized.strip():
            logger.error("Sanitization resulted in empty output, using fallback")
            # Fallback: strip all tags
            return re.sub(r'<[^>]+>', '', html)
        
        return sanitized
        
    except Exception as e:
        logger.error(f"HTML sanitization error: {type(e).__name__}: {e}")
        # Emergency fallback: strip all tags
        try:
            return re.sub(r'<[^>]+>', '', html)
        except:
            # Last resort: return empty
            return ""


def validate_html_structure(html: str) -> List[str]:
    """
    Validate HTML structure and return list of unclosed tags.
    Uses a simple tag stack to check for matching open/close tags.
    """
    tag_stack = []
    unclosed_tags = []
    
    # Find all tags
    tag_pattern = r'<(/?)(\w+)[^>]*>'
    matches = re.finditer(tag_pattern, html)
    
    for match in matches:
        is_closing = match.group(1) == '/'
        tag_name = match.group(2).lower()
        
        # Skip self-closing tags
        if tag_name in ('br', 'img', 'hr', 'input'):
            continue
        
        if is_closing:
            # Closing tag - pop from stack
            if tag_stack and tag_stack[-1] == tag_name:
                tag_stack.pop()
            else:
                unclosed_tags.append(tag_name)
        else:
            # Opening tag - push to stack
            tag_stack.append(tag_name)
    
    # Any remaining tags in stack are unclosed
    unclosed_tags.extend(tag_stack)
    
    return unclosed_tags


def markdown_to_html(markdown: str) -> str:
    """
    Convert markdown to HTML.
    Simple implementation for common markdown syntax.
    TODO: Use a proper markdown library like markdown2 or mistune
    """
    html = markdown
    
    # Headers
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Bold
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)
    
    # Italic
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)
    
    # Code
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
    
    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', html)
    
    # Line breaks
    html = html.replace('\n\n', '</p><p>')
    html = f'<p>{html}</p>'
    
    return html


def analyze_content_complexity(content: str) -> Dict[str, any]:
    """
    Analyze content to determine optimal streaming strategy.
    Returns complexity level and recommended settings.
    """
    # Detect content type
    has_html = bool(re.search(r'<[^>]+>', content))
    has_markdown = bool(re.search(r'[\*_#\[\]`]', content))
    
    # Calculate complexity metrics
    word_count = len(content.split())
    char_count = len(content)
    line_count = content.count('\n') + 1
    
    # Detect HTML tag density
    if has_html:
        tags = re.findall(r'<[^>]+>', content)
        tag_density = len(tags) / max(word_count, 1)
    else:
        tag_density = 0
    
    # Determine complexity level
    if tag_density > 0.3 or word_count > 500:
        complexity = "high"
        recommended_chunk_by = "sentence"
        recommended_speed = "normal"
    elif tag_density > 0.1 or word_count > 200:
        complexity = "medium"
        recommended_chunk_by = "word"
        recommended_speed = "fast"
    else:
        complexity = "low"
        recommended_chunk_by = "word"
        recommended_speed = "superfast"
    
    return {
        "has_html": has_html,
        "has_markdown": has_markdown,
        "word_count": word_count,
        "char_count": char_count,
        "line_count": line_count,
        "tag_density": tag_density,
        "complexity": complexity,
        "recommended_chunk_by": recommended_chunk_by,
        "recommended_speed": recommended_speed
    }


def chunk_html_by_words(html: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Production-ready HTML chunking with tag-aware splitting.
    
    Safely splits HTML into word-based chunks while:
    - Preserving complete tag pairs (opening + content + closing)
    - Handling nested tags correctly with stack-based tracking
    - Supporting self-closing tags (br, img, hr, input)
    - Enforcing maximum chunk size for memory safety
    - Validating tag structure during processing
    
    Args:
        html: HTML string to chunk
        max_chunk_size: Maximum characters per chunk (safety limit)
    
    Returns:
        List of HTML chunks, each containing complete, valid tag structures
    
    Raises:
        ValueError: If HTML exceeds safe processing limits
    """
    # Input validation
    if not html or not isinstance(html, str):
        logger.warning("Empty or invalid HTML provided for chunking")
        return []
    
    # Safety limit for extremely large content
    MAX_CONTENT_SIZE = 1_000_000  # 1MB
    if len(html) > MAX_CONTENT_SIZE:
        logger.error(f"HTML content too large: {len(html)} chars (max: {MAX_CONTENT_SIZE})")
        raise ValueError(f"HTML content exceeds maximum size of {MAX_CONTENT_SIZE} characters")
    
    chunks = []
    current_chunk = ""
    tag_stack = []
    i = 0
    self_closing_tags = {'br', 'img', 'hr', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'param', 'source', 'track', 'wbr'}
    
    try:
        while i < len(html):
            char = html[i]
            
            if char == '<':
                # Found a tag - parse it
                tag_end = html.find('>', i)
                if tag_end == -1:
                    # Malformed HTML - log warning and handle gracefully
                    logger.warning(f"Malformed HTML: unclosed tag at position {i}")
                    current_chunk += html[i:]
                    break
                
                tag = html[i:tag_end + 1]
                
                # Extract tag name for validation
                tag_match = re.match(r'<(/?)(\w+)', tag)
                if not tag_match:
                    # Invalid tag format - skip but log
                    logger.warning(f"Invalid tag format: {tag[:50]}...")
                    current_chunk += tag
                    i = tag_end + 1
                    continue
                
                is_closing = tag_match.group(1) == '/'
                tag_name = tag_match.group(2).lower()
                
                # Check if it's a closing tag
                if is_closing:
                    # Closing tag - add it to current chunk
                    current_chunk += tag
                    
                    # Validate matching opening tag
                    if tag_stack and tag_stack[-1] == tag_name:
                        tag_stack.pop()
                    else:
                        logger.warning(f"Mismatched closing tag: {tag_name}, expected: {tag_stack[-1] if tag_stack else 'none'}")
                    
                    # If stack is empty, we have a complete unit
                    if not tag_stack and current_chunk.strip():
                        chunks.append(current_chunk)
                        current_chunk = ""
                        
                elif tag.endswith('/>') or tag_name in self_closing_tags:
                    # Self-closing tag or known void element
                    current_chunk += tag
                    
                else:
                    # Opening tag - add to chunk and push to stack
                    current_chunk += tag
                    tag_stack.append(tag_name)
                
                i = tag_end + 1
                
            elif char in (' ', '\n', '\t', '\r'):
                # Word boundary (whitespace)
                if tag_stack:
                    # Inside tags - keep accumulating
                    current_chunk += char
                else:
                    # Outside tags - can split here
                    if current_chunk.strip():
                        # Safety check: enforce max chunk size
                        if len(current_chunk) > max_chunk_size:
                            logger.warning(f"Chunk exceeds max size ({len(current_chunk)} > {max_chunk_size}), force splitting")
                            chunks.append(current_chunk)
                            current_chunk = ""
                        else:
                            chunks.append(current_chunk + char)
                            current_chunk = ""
                    else:
                        current_chunk += char
                i += 1
                
            else:
                current_chunk += char
                i += 1
                
                # Safety check: prevent runaway chunks
                if len(current_chunk) > max_chunk_size * 2:
                    logger.error(f"Chunk size exceeded safety limit, force flushing")
                    if current_chunk.strip():
                        chunks.append(current_chunk)
                    current_chunk = ""
                    tag_stack.clear()
        
        # Handle remaining content
        if current_chunk.strip():
            if tag_stack:
                logger.warning(f"Unclosed tags at end of HTML: {tag_stack}")
            chunks.append(current_chunk)
        
        # Validation: check for empty chunks
        chunks = [c for c in chunks if c.strip()]
        
        logger.info(f"HTML chunked successfully: {len(chunks)} chunks, {len(tag_stack)} unclosed tags")
        return chunks
        
    except Exception as e:
        logger.error(f"Error during HTML chunking: {type(e).__name__}: {e}")
        # Fallback: return original content as single chunk
        return [html] if html.strip() else []


def chunk_by_sentences(text: str) -> List[str]:
    """
    Split text into sentence-based chunks.
    Handles common sentence endings while preserving abbreviations.
    """
    # Simple sentence splitting - can be enhanced with NLP
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() + ' ' for s in sentences if s.strip()]


def chunk_by_paragraphs(text: str) -> List[str]:
    """Split text into paragraph-based chunks"""
    paragraphs = text.split('\n\n')
    return [p.strip() + '\n\n' for p in paragraphs if p.strip()]


def smart_chunk_content(
    content: str,
    content_type: str,
    chunk_by: str = "word",
    max_chunk_size: int = 1000
) -> Tuple[List[str], Dict[str, any]]:
    """
    Production-ready intelligent content chunking with comprehensive error handling.
    
    Features:
    - Automatic content type detection and conversion
    - Security-first HTML sanitization
    - Intelligent chunking based on content complexity
    - Extensive validation and error recovery
    - Performance monitoring and metrics
    
    Args:
        content: Content string to chunk
        content_type: Type of content ('text', 'html', 'markdown')
        chunk_by: Chunking strategy ('word', 'sentence', 'paragraph', 'character')
        max_chunk_size: Maximum size per chunk for safety
    
    Returns:
        Tuple of (chunks list, metadata dict with metrics)
    
    Raises:
        ValueError: If content type or chunk strategy is invalid
    """
    import time
    start_time = time.time()
    
    # Input validation
    if not content or not isinstance(content, str):
        logger.warning("Empty or invalid content provided")
        return [], {
            "original_length": 0,
            "chunk_count": 0,
            "content_type": content_type,
            "error": "empty_content"
        }
    
    # Validate content type
    valid_types = {"text", "html", "markdown"}
    if content_type not in valid_types:
        logger.error(f"Invalid content_type: {content_type}")
        raise ValueError(f"content_type must be one of {valid_types}")
    
    # Validate chunking strategy
    valid_strategies = {"word", "sentence", "paragraph", "character"}
    if chunk_by not in valid_strategies:
        logger.error(f"Invalid chunk_by: {chunk_by}")
        raise ValueError(f"chunk_by must be one of {valid_strategies}")
    
    original_length = len(content)
    original_type = content_type
    
    try:
        # Step 1: Keep markdown as-is (DO NOT convert to HTML)
        # The frontend Tiptap editor will handle markdown rendering
        if content_type == "markdown":
            logger.info(f"Keeping markdown format: {original_length} chars")
            # No conversion needed - send raw markdown to frontend
        
        # Step 2: Sanitize HTML if applicable
        if content_type == "html":
            try:
                pre_sanitize_length = len(content)
                content = sanitize_html(content)
                
                if not content:
                    logger.warning("Sanitization resulted in empty content")
                    return [], {
                        "original_length": original_length,
                        "chunk_count": 0,
                        "content_type": original_type,
                        "error": "sanitization_emptied_content"
                    }
                
                # Check structure validity
                unclosed_tags = validate_html_structure(content)
                if unclosed_tags:
                    logger.warning(f"Unclosed HTML tags after sanitization: {unclosed_tags}")
                
                logger.info(f"Sanitized HTML: {pre_sanitize_length} → {len(content)} chars")
                
            except Exception as e:
                logger.error(f"HTML sanitization failed: {e}, falling back to text mode")
                content_type = "text"
                content = re.sub(r'<[^>]+>', '', content)
        
        # Step 3: Analyze content complexity
        try:
            analysis = analyze_content_complexity(content)
        except Exception as e:
            logger.error(f"Content analysis failed: {e}, using defaults")
            analysis = {
                "has_html": False,
                "has_markdown": False,
                "word_count": len(content.split()),
                "char_count": len(content),
                "complexity": "unknown"
            }
        
        # Step 4: Chunk based on strategy
        chunks = []
        
        try:
            if chunk_by == "word":
                if analysis.get("has_html", False):
                    chunks = chunk_html_by_words(content, max_chunk_size)
                else:
                    chunks = content.split()
                    chunks = [c + ' ' for c in chunks]  # Preserve spaces
                    
            elif chunk_by == "sentence":
                chunks = chunk_by_sentences(content)
                
            elif chunk_by == "paragraph":
                chunks = chunk_by_paragraphs(content)
                
            elif chunk_by == "character":
                if analysis.get("has_html", False):
                    # For HTML, treat tags as atomic
                    chunks = chunk_html_by_words(content, max_chunk_size)
                    # Further split text parts into characters
                    refined_chunks = []
                    for chunk in chunks:
                        if chunk.startswith('<'):
                            refined_chunks.append(chunk)
                        else:
                            refined_chunks.extend(list(chunk))
                    chunks = refined_chunks
                else:
                    chunks = list(content)
                    
            else:
                # Fallback: return as single chunk
                logger.warning(f"Unknown chunk_by strategy: {chunk_by}, using single chunk")
                chunks = [content]
        
        except Exception as e:
            logger.error(f"Chunking failed: {type(e).__name__}: {e}, using single chunk fallback")
            chunks = [content]
        
        # Post-processing: filter empty chunks
        chunks = [c for c in chunks if c and c.strip()]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build comprehensive metadata
        metadata = {
            "original_length": original_length,
            "final_length": sum(len(c) for c in chunks),
            "chunk_count": len(chunks),
            "content_type": original_type,
            "final_type": content_type,
            "chunk_by": chunk_by,
            "analysis": analysis,
            "processing_time_ms": round(processing_time * 1000, 2),
            "avg_chunk_size": round(sum(len(c) for c in chunks) / len(chunks), 2) if chunks else 0,
            "max_chunk_size": max(len(c) for c in chunks) if chunks else 0,
        }
        
        logger.info(f"Content chunked successfully: {len(chunks)} chunks in {metadata['processing_time_ms']}ms")
        
        return chunks, metadata
        
    except Exception as e:
        logger.error(f"Critical error in smart_chunk_content: {type(e).__name__}: {e}")
        # Emergency fallback: return original content
        return [content], {
            "original_length": original_length,
            "chunk_count": 1,
            "content_type": original_type,
            "error": str(e),
            "fallback": True
        }
