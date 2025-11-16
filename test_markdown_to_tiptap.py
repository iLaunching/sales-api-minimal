"""
Unit tests for Markdown to Tiptap JSON converter.

Tests all markdown patterns to ensure correct Tiptap JSON output.
"""

import unittest
import json
from markdown_to_tiptap import MarkdownToTiptapConverter, convert_markdown_to_tiptap


class TestMarkdownToTiptapConverter(unittest.TestCase):
    """Test suite for MarkdownToTiptapConverter."""
    
    def setUp(self):
        """Set up converter instance for each test."""
        self.converter = MarkdownToTiptapConverter()
    
    def test_code_block_triple_backticks(self):
        """Test code block with triple backticks."""
        markdown = """```python
def hello():
    print("Hello")
```"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "codeBlock")
        self.assertEqual(nodes[0]["attrs"]["language"], "python")
        self.assertIn("def hello():", nodes[0]["content"][0]["text"])
    
    def test_code_block_single_backticks(self):
        """Test code block with single backticks (LLM mistake pattern)."""
        markdown = """`python
def hello():
    print("Hello")
`"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "codeBlock")
        self.assertEqual(nodes[0]["attrs"]["language"], "python")
    
    def test_code_block_no_language(self):
        """Test code block without language specifier."""
        markdown = """```
some code here
```"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "codeBlock")
        self.assertEqual(nodes[0]["attrs"]["language"], "plaintext")
    
    def test_heading_levels(self):
        """Test all heading levels (h1-h6)."""
        markdown = """# Heading 1
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 6)
        for i, node in enumerate(nodes, 1):
            self.assertEqual(node["type"], "heading")
            self.assertEqual(node["attrs"]["level"], i)
            self.assertIn(f"Heading {i}", node["content"][0]["text"])
    
    def test_paragraph(self):
        """Test simple paragraph."""
        markdown = "This is a simple paragraph."
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "paragraph")
        self.assertEqual(nodes[0]["content"][0]["text"], markdown)
    
    def test_paragraph_with_multiple_lines(self):
        """Test paragraph with line breaks (no double newline)."""
        markdown = "Line one\nLine two\nLine three"
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "paragraph")
    
    def test_multiple_paragraphs(self):
        """Test multiple paragraphs separated by double newlines."""
        markdown = """First paragraph.

Second paragraph.

Third paragraph."""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 3)
        for node in nodes:
            self.assertEqual(node["type"], "paragraph")
    
    def test_unordered_list(self):
        """Test bullet/unordered list."""
        markdown = """- Item 1
- Item 2
- Item 3"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "bulletList")
        self.assertEqual(len(nodes[0]["content"]), 3)
        
        for i, item in enumerate(nodes[0]["content"], 1):
            self.assertEqual(item["type"], "listItem")
            self.assertIn(f"Item {i}", item["content"][0]["content"][0]["text"])
    
    def test_ordered_list(self):
        """Test numbered/ordered list."""
        markdown = """1. First item
2. Second item
3. Third item"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "orderedList")
        self.assertEqual(len(nodes[0]["content"]), 3)
    
    def test_inline_code(self):
        """Test inline code with single backticks."""
        markdown = "Here is some `inline code` in text."
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "paragraph")
        
        # Should have text nodes, one with code mark
        content = nodes[0]["content"]
        has_code_mark = any(
            "marks" in node and 
            any(mark["type"] == "code" for mark in node.get("marks", []))
            for node in content
        )
        self.assertTrue(has_code_mark)
    
    def test_mixed_content(self):
        """Test mixed content: heading, paragraph, code block."""
        markdown = """# My Heading

This is a paragraph with some text.

```python
def example():
    return True
```

Another paragraph after code."""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 4)
        self.assertEqual(nodes[0]["type"], "heading")
        self.assertEqual(nodes[1]["type"], "paragraph")
        self.assertEqual(nodes[2]["type"], "codeBlock")
        self.assertEqual(nodes[3]["type"], "paragraph")
    
    def test_code_block_with_special_characters(self):
        """Test code block containing special characters."""
        markdown = """```javascript
const regex = /[a-z]+/gi;
const data = {"key": "value"};
```"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "codeBlock")
        self.assertIn("regex", nodes[0]["content"][0]["text"])
        self.assertIn("const data", nodes[0]["content"][0]["text"])
    
    def test_empty_input(self):
        """Test empty or whitespace-only input."""
        self.assertEqual(self.converter.parse_markdown(""), [])
        self.assertEqual(self.converter.parse_markdown("   "), [])
        self.assertEqual(self.converter.parse_markdown("\n\n\n"), [])
    
    def test_bold_text(self):
        """Test bold text with **."""
        markdown = "This is **bold text** in a sentence."
        
        nodes = self.converter.parse_markdown(markdown)
        
        self.assertEqual(len(nodes), 1)
        content = nodes[0]["content"]
        
        # Should have a node with bold mark
        has_bold = any(
            "marks" in node and 
            any(mark["type"] == "bold" for mark in node.get("marks", []))
            for node in content
        )
        self.assertTrue(has_bold)
    
    def test_convenience_function(self):
        """Test convenience function convert_markdown_to_tiptap."""
        markdown = "# Test\n\nSimple test."
        nodes = convert_markdown_to_tiptap(markdown)
        
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0]["type"], "heading")
        self.assertEqual(nodes[1]["type"], "paragraph")
    
    def test_multiple_code_blocks(self):
        """Test multiple code blocks in one document."""
        markdown = """First some Python:

```python
def hello():
    pass
```

And then some JavaScript:

```javascript
function hello() {
    return true;
}
```"""
        
        nodes = self.converter.parse_markdown(markdown)
        
        # Should have: paragraph, codeBlock, paragraph, codeBlock
        code_blocks = [n for n in nodes if n["type"] == "codeBlock"]
        self.assertEqual(len(code_blocks), 2)
        self.assertEqual(code_blocks[0]["attrs"]["language"], "python")
        self.assertEqual(code_blocks[1]["attrs"]["language"], "javascript")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.converter = MarkdownToTiptapConverter()
    
    def test_code_block_without_closing_backticks(self):
        """Test handling of incomplete code block."""
        markdown = """```python
def incomplete():
    print("no closing backticks")"""
        
        # Should handle gracefully (treat as paragraph or ignore)
        nodes = self.converter.parse_markdown(markdown)
        self.assertIsInstance(nodes, list)
    
    def test_nested_formatting(self):
        """Test nested formatting like bold+italic."""
        markdown = "This has ***bold and italic*** text."
        
        nodes = self.converter.parse_markdown(markdown)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "paragraph")
    
    def test_special_characters_in_headings(self):
        """Test headings with special characters."""
        markdown = "# Heading with `code` and **bold**"
        
        nodes = self.converter.parse_markdown(markdown)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["type"], "heading")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
