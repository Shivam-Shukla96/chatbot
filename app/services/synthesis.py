from transformers.pipelines import pipeline
from typing import List, Dict

summarizer = pipeline("summarization", model="t5-base")

def summarize_themes(chunks: List[Dict[str, str]]) -> str:
    """
    Summarizes themes across a list of document chunks using Hugging Face summarization models.

    Args:
        chunks (List[Dict[str, str]]): List of text chunks with keys 'content' and optionally 'meta'.

    Returns:
        str: Combined summary of the input texts or an error message.
    """
    try:
        texts = "\n".join([c['content'] for c in chunks if 'content' in c])
        print(f"Summarizing {len(chunks)} chunks of text.")
        if not texts.strip():
            return "No valid content found to summarize."
        
        print(f"Total characters to summarize: {len(texts)}")

        max_chunk_chars = 1000
        text_parts = [texts[i:i+max_chunk_chars] for i in range(0, len(texts), max_chunk_chars)]
        print(f"Split into {len(text_parts)} parts for summarization.")

        summaries = []
        for part in text_parts:
            input_len = len(part.split())
            max_len = max(30, min(150, int(input_len * 0.7)))
            summary = summarizer(part, max_length=max_len, min_length=20, do_sample=False)
            summaries.append(summary[0]['summary_text']) # type: ignore
        
        print(f"Generated {len(summaries)} summaries.")
        if not summaries:
            print("No summaries generated.")
            return "No summaries generated."

        return "\n".join(summaries)

    except Exception as e:
        return f"Summarization failed: {str(e)}"
