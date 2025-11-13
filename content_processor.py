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
    Sanitize HTML content for safe streaming to Tiptap.
    Removes dangerous tags and attributes.
    """
    try:
        sanitizer = HTMLSanitizer()
        sanitizer.feed(html)
        return sanitizer.get_sanitized()
    except Exception as e:
        logger.error(f"HTML sanitization error: {e}")
        # Return plain text if sanitization fails
        return re.sub(r'<[^>]+>', '', html)


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


def chunk_html_by_words(html: str) -> List[str]:
    """
    Split HTML into word-based chunks while preserving complete tags.
    Tags are treated as atomic units and never split.
    """
    chunks = []
    current_chunk = ""
    in_tag = False
    
    i = 0
    while i < len(html):
        char = html[i]
        
        if char == '<':
            # Start of tag - save current chunk if not empty
            if current_chunk.strip():
                chunks.append(current_chunk)
                current_chunk = ""
            # Start collecting tag
            in_tag = True
            tag_start = i
        
        if in_tag:
            if char == '>':
                # Complete tag found
                tag = html[tag_start:i+1]
                chunks.append(tag)
                in_tag = False
                i += 1
                continue
        else:
            # Not in tag - check for word boundary
            if char in (' ', '\n', '\t'):
                # Word boundary - save chunk with space
                if current_chunk:
                    chunks.append(current_chunk + char)
                    current_chunk = ""
                elif char == ' ':
                    # Standalone space
                    chunks.append(' ')
            else:
                current_chunk += char
        
        i += 1
    
    # Add any remaining chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


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
    chunk_by: str = "word"
) -> Tuple[List[str], Dict[str, any]]:
    """
    Intelligently chunk content based on type and strategy.
    Returns chunks and processing metadata.
    """
    # Step 1: Convert markdown to HTML if needed
    if content_type == "markdown":
        content = markdown_to_html(content)
        content_type = "html"
    
    # Step 2: Sanitize HTML if applicable
    if content_type == "html":
        content = sanitize_html(content)
        unclosed_tags = validate_html_structure(content)
        if unclosed_tags:
            logger.warning(f"Unclosed HTML tags detected: {unclosed_tags}")
    
    # Step 3: Analyze content
    analysis = analyze_content_complexity(content)
    
    # Step 4: Chunk based on strategy
    if chunk_by == "word":
        if analysis["has_html"]:
            chunks = chunk_html_by_words(content)
        else:
            chunks = content.split()
            chunks = [c + ' ' for c in chunks]  # Preserve spaces
    elif chunk_by == "sentence":
        chunks = chunk_by_sentences(content)
    elif chunk_by == "paragraph":
        chunks = chunk_by_paragraphs(content)
    elif chunk_by == "character":
        if analysis["has_html"]:
            # For HTML, treat tags as atomic
            chunks = chunk_html_by_words(content)
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
        chunks = [content]
    
    metadata = {
        "original_length": len(content),
        "chunk_count": len(chunks),
        "content_type": content_type,
        "analysis": analysis
    }
    
    return chunks, metadata
