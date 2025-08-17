def chunk_text(text, max_tokens=300):
    paragraphs = text.split("\n\n")
    return [{"content": p.strip(), "meta": {"source": "upload"}} for p in paragraphs if p.strip()]
