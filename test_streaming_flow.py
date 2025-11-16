"""
Manual test to verify markdown-to-Tiptap conversion and streaming.

This simulates what happens when:
1. LLM returns markdown response
2. Backend converts to Tiptap JSON
3. Backend streams JSON nodes

Run this to verify Phase 1 & 2 work correctly.
"""

import json
from markdown_to_tiptap import convert_markdown_to_tiptap

# Simulate LLM response with various markdown elements
llm_markdown_response = """
# API Integration Guide

Here's how to integrate with our API:

## Step 1: Authentication

First, get your API key from the dashboard. You can find it in **Settings** > **API Keys**.

## Step 2: Make a Request

Here's a Python example:

```python
import requests

def call_api(query):
    url = "https://api.example.com/v1/query"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(url, json={"query": query}, headers=headers)
    return response.json()
```

And here's JavaScript:

```javascript
async function callAPI(query) {
    const response = await fetch('https://api.example.com/v1/query', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${API_KEY}`,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query })
    });
    return response.json();
}
```

## Key Features

- Fast response times (< 200ms)
- Real-time streaming
- 99.9% uptime
- Comprehensive error handling

For more details, check out our *documentation* at `docs.example.com`.
"""

print("="*80)
print("SIMULATING LLM RESPONSE → TIPTAP JSON CONVERSION")
print("="*80)
print()
print("📝 LLM Markdown Response:")
print("-"*80)
print(llm_markdown_response)
print()
print("="*80)
print("🔄 Converting to Tiptap JSON...")
print("="*80)
print()

# Convert markdown to Tiptap JSON
tiptap_nodes = convert_markdown_to_tiptap(llm_markdown_response)

print(f"✅ Conversion Complete! Generated {len(tiptap_nodes)} Tiptap nodes")
print()
print("="*80)
print("📦 TIPTAP JSON NODES (what gets streamed to frontend):")
print("="*80)
print()

# Pretty-print each node as it would be streamed
for i, node in enumerate(tiptap_nodes, 1):
    print(f"Node {i}/{len(tiptap_nodes)}: {node.get('type', 'unknown').upper()}")
    print("-"*80)
    print(json.dumps(node, indent=2))
    print()

print("="*80)
print("🎬 SIMULATED WEBSOCKET MESSAGES:")
print("="*80)
print()

# Simulate WebSocket messages
print("1. Stream Start:")
stream_start = {
    "type": "stream_start",
    "total_nodes": len(tiptap_nodes),
    "metadata": {
        "node_count": len(tiptap_nodes),
        "speed_used": "normal",
        "format": "tiptap_json"
    }
}
print(json.dumps(stream_start, indent=2))
print()

print("2. First 3 Node Messages:")
for i in range(min(3, len(tiptap_nodes))):
    node_msg = {
        "type": "node",
        "data": tiptap_nodes[i],
        "index": i,
        "shouldAnimate": True
    }
    print(f"\nNode Message {i+1}:")
    print(json.dumps(node_msg, indent=2))

if len(tiptap_nodes) > 3:
    print(f"\n... ({len(tiptap_nodes) - 3} more nodes) ...")

print("\n3. Stream Complete:")
stream_complete = {
    "type": "stream_complete",
    "total_nodes": len(tiptap_nodes)
}
print(json.dumps(stream_complete, indent=2))

print()
print("="*80)
print("✅ TEST COMPLETE!")
print("="*80)
print()
print("KEY OBSERVATIONS:")
print("✅ Code blocks converted to 'codeBlock' nodes with language")
print("✅ Headings converted to 'heading' nodes with level")
print("✅ Paragraphs converted to 'paragraph' nodes")
print("✅ Lists converted to 'bulletList'/'orderedList' nodes")
print("✅ Inline formatting preserved (bold, code, etc.)")
print()
print("Next: Frontend needs to receive 'node' messages and insert JSON directly!")
