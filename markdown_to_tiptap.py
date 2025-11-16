"""
Markdown to Tiptap JSON Converter

Converts markdown text to Tiptap-compatible JSON node structures.
Handles code blocks, headings, lists, inline formatting, and more.

Author: iLaunching Development Team
Date: November 16, 2025
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class MarkdownToTiptapConverter:
    """
    Converts markdown text to Tiptap JSON node structures.
    
    Supports:
    - Code blocks (triple and single backticks)
    - Headings (h1-h6)
    - Paragraphs
    - Lists (ordered and unordered)
    - Inline formatting (bold, italic, code, links)
    """
    
    def __init__(self):
        # Regex patterns
        self.code_block_pattern = re.compile(
            r'```(\w+)?\s*\n(.*?)```',
            re.DOTALL | re.MULTILINE
        )
        self.single_backtick_code_block = re.compile(
            r'`(\w+)?\s*\n(.*?)`',
            re.DOTALL | re.MULTILINE
        )
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.list_item_pattern = re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE)
        self.ordered_list_pattern = re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE)
    
    def parse_markdown(self, text: str) -> List[Dict[str, Any]]:
        """
        Main entry point: Parse markdown text into Tiptap JSON nodes.
        
        Args:
            text: Raw markdown string
            
        Returns:
            List of Tiptap JSON node dictionaries
        """
        if not text or not text.strip():
            return []
        
        nodes = []
        
        # First pass: Extract code blocks and replace with placeholders
        code_blocks = []
        text_with_placeholders = self._extract_code_blocks(text, code_blocks)
        
        # Split by double newlines OR single newlines for headings
        # This ensures each heading gets its own block
        lines = text_with_placeholders.split('\n')
        blocks = []
        current_block = []
        
        for line in lines:
            stripped = line.strip()
            # Check if this is a heading
            if stripped and stripped[0] == '#':
                # Save current block if any
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                # Add heading as its own block
                blocks.append(line)
            elif not stripped and current_block:
                # Empty line - end current block
                blocks.append('\n'.join(current_block))
                current_block = []
            elif stripped:
                # Add to current block
                current_block.append(line)
        
        # Add final block if any
        if current_block:
            blocks.append('\n'.join(current_block))
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Check if this is a code block placeholder
            if block.startswith('__CODE_BLOCK_'):
                index = int(block.replace('__CODE_BLOCK_', '').replace('__', ''))
                nodes.append(code_blocks[index])
                continue
            
            # Check for headings
            heading_node = self._parse_heading(block)
            if heading_node:
                nodes.append(heading_node)
                continue
            
            # Check for lists
            if self._is_list(block):
                list_node = self._parse_list(block)
                if list_node:
                    nodes.append(list_node)
                    continue
            
            # Default to paragraph
            paragraph_node = self._parse_paragraph(block)
            if paragraph_node:
                nodes.append(paragraph_node)
        
        return nodes
    
    def _extract_code_blocks(self, text: str, code_blocks: List[Dict]) -> str:
        """
        Extract code blocks and replace with placeholders.
        Handles both triple and single backticks.
        """
        # First, handle triple backticks
        def replace_triple(match):
            language = match.group(1) or 'plaintext'
            code = match.group(2)
            node = self._create_code_block_node(language, code)
            index = len(code_blocks)
            code_blocks.append(node)
            return f'\n\n__CODE_BLOCK_{index}__\n\n'
        
        text = self.code_block_pattern.sub(replace_triple, text)
        
        # Then handle single backticks (LLM mistake pattern)
        # Only if they contain newlines (multi-line code)
        def replace_single(match):
            language = match.group(1) or 'plaintext'
            code = match.group(2)
            if '\n' in code:  # Multi-line code block
                node = self._create_code_block_node(language, code)
                index = len(code_blocks)
                code_blocks.append(node)
                return f'\n\n__CODE_BLOCK_{index}__\n\n'
            else:
                # Keep as inline code
                return match.group(0)
        
        text = self.single_backtick_code_block.sub(replace_single, text)
        
        return text
    
    def _create_code_block_node(self, language: str, code: str) -> Dict[str, Any]:
        """
        Create a Tiptap codeBlock node.
        
        Returns:
            {
                "type": "codeBlock",
                "attrs": {"language": "python"},
                "content": [{"type": "text", "text": "code content"}]
            }
        """
        return {
            "type": "codeBlock",
            "attrs": {
                "language": language.lower().strip()
            },
            "content": [
                {
                    "type": "text",
                    "text": code.rstrip('\n')  # Remove trailing newlines
                }
            ]
        }
    
    def _parse_heading(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse heading markdown into Tiptap heading node.
        
        Examples:
            # Heading 1 -> level 1
            ## Heading 2 -> level 2
        """
        match = self.heading_pattern.match(text)
        if not match:
            return None
        
        level = len(match.group(1))  # Count the #'s
        content_text = match.group(2).strip()
        
        # Parse inline formatting in heading text
        inline_content = self._parse_inline_formatting(content_text)
        
        return {
            "type": "heading",
            "attrs": {"level": level},
            "content": inline_content
        }
    
    def _parse_paragraph(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse paragraph with inline formatting.
        
        Returns:
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": "..."}, ...]
            }
        """
        if not text.strip():
            return None
        
        # Parse inline formatting
        inline_content = self._parse_inline_formatting(text)
        
        return {
            "type": "paragraph",
            "content": inline_content
        }
    
    def _is_list(self, text: str) -> bool:
        """Check if text contains list items."""
        return bool(
            self.list_item_pattern.search(text) or 
            self.ordered_list_pattern.search(text)
        )
    
    def _parse_list(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse list (ordered or unordered) into Tiptap list nodes.
        
        Returns:
            {
                "type": "bulletList" | "orderedList",
                "content": [
                    {"type": "listItem", "content": [{"type": "paragraph", ...}]},
                    ...
                ]
            }
        """
        lines = text.split('\n')
        
        # Detect list type from first item
        is_ordered = bool(self.ordered_list_pattern.match(lines[0]))
        list_type = "orderedList" if is_ordered else "bulletList"
        
        items = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract item text
            if is_ordered:
                match = self.ordered_list_pattern.match(line)
            else:
                match = self.list_item_pattern.match(line)
            
            if match:
                item_text = match.group(1)
                inline_content = self._parse_inline_formatting(item_text)
                
                items.append({
                    "type": "listItem",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": inline_content
                        }
                    ]
                })
        
        if not items:
            return None
        
        return {
            "type": list_type,
            "content": items
        }
    
    def _parse_inline_formatting(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse inline formatting (bold, italic, code, links) into text nodes with marks.
        
        Returns list of text nodes with appropriate marks:
            [
                {"type": "text", "text": "normal"},
                {"type": "text", "marks": [{"type": "bold"}], "text": "bold"},
                ...
            ]
        """
        # For now, return simple text nodes
        # TODO: Implement full inline formatting parser
        
        # Handle inline code first (to avoid conflicts with other formatting)
        parts = self._split_by_inline_code(text)
        
        result = []
        for is_code, content in parts:
            if is_code:
                result.append({
                    "type": "text",
                    "marks": [{"type": "code"}],
                    "text": content
                })
            else:
                # Parse bold, italic, links
                formatted = self._parse_bold_italic_links(content)
                result.extend(formatted)
        
        return result if result else [{"type": "text", "text": text}]
    
    def _split_by_inline_code(self, text: str) -> List[Tuple[bool, str]]:
        """
        Split text by inline code backticks.
        Returns list of (is_code, content) tuples.
        """
        parts = []
        pattern = re.compile(r'`([^`\n]+?)`')
        last_end = 0
        
        for match in pattern.finditer(text):
            # Add text before code
            if match.start() > last_end:
                parts.append((False, text[last_end:match.start()]))
            
            # Add code content
            parts.append((True, match.group(1)))
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(text):
            parts.append((False, text[last_end:]))
        
        return parts if parts else [(False, text)]
    
    def _parse_bold_italic_links(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse bold (**text**), italic (*text*), and links [text](url).
        """
        # Simple implementation - just return text for now
        # TODO: Implement proper bold/italic/link parsing with marks
        
        result = []
        
        # Handle bold (**text**)
        bold_pattern = re.compile(r'\*\*(.+?)\*\*')
        italic_pattern = re.compile(r'\*([^\*\s][^\*]*?)\*')
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        
        # For simplicity, just handle these sequentially
        # A full implementation would need a proper parser
        
        # Quick implementation: replace patterns and track marks
        remaining = text
        
        # Split by bold
        parts = bold_pattern.split(remaining)
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Not bold
                if part:
                    result.append({"type": "text", "text": part})
            else:
                # Bold
                result.append({
                    "type": "text",
                    "marks": [{"type": "bold"}],
                    "text": part
                })
        
        return result if result else [{"type": "text", "text": text}]


def convert_markdown_to_tiptap(markdown: str) -> List[Dict[str, Any]]:
    """
    Convenience function to convert markdown to Tiptap JSON.
    
    Args:
        markdown: Raw markdown string
        
    Returns:
        List of Tiptap JSON nodes
    """
    converter = MarkdownToTiptapConverter()
    return converter.parse_markdown(markdown)


# Example usage
if __name__ == "__main__":
    # Test with code block
    markdown = """
Here's a Python example:

```python
def hello_world():
    print("Hello, World!")
    return True
```

And some **bold text** with *italic*.
"""
    
    converter = MarkdownToTiptapConverter()
    nodes = converter.parse_markdown(markdown)
    
    import json
    print(json.dumps(nodes, indent=2))
