from transformers.pipelines import pipeline
from typing import List, Dict, Union, Any

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Better model for our use case
# summarizer = pipeline("summarization", model="t5-base")
# NEW (stronger summarizer)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


ChunkType = Dict[str, Union[str, Dict[str, Any]]]

def summarize_themes(chunks: Union[List[ChunkType], Dict[str, str]], query: str = "") -> str:
    """
    Summarizes themes across a list of document chunks using Hugging Face summarization models.

    Args:
        chunks (Union[List[Dict[str, str]], Dict[str, str]]): List of text chunks with keys 'content' and 'meta',
            or an error dictionary with 'status' and 'message'.
        query (str, optional): The original query to provide context for summarization.

    Returns:
        str: Combined summary of the input texts or an error message.
    """
    try:
        # Handle error case
        if isinstance(chunks, dict):
            if chunks.get("status") == "error":
                return chunks.get("message", "Unknown error occurred")
            return "Invalid input format"

        if not isinstance(chunks, list):
            return "Invalid input: expected a list of chunks"

        # Extract and validate text chunks
        valid_chunks = [c for c in chunks if isinstance(c, dict) and 'content' in c]
        print(f"Summarizing {len(valid_chunks)} valid chunks of text.")
        if not valid_chunks:
            return "No valid content found to summarize."

        # Sort chunks by similarity if available
        chunks_with_scores = []
        for chunk in valid_chunks:
            meta = chunk.get('meta', {})
            if isinstance(meta, dict):
                similarity = meta.get('similarity', 0)
                print(f"Processing chunk with similarity: {similarity}")  # Debug info
            else:
                similarity = 0
                print("Chunk has no similarity score")
            chunks_with_scores.append((chunk, similarity))
        
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        print(f"Sorted {len(chunks_with_scores)} chunks by similarity")

        # Prepare context-aware prompt
        context_parts = []
        min_similarity = 0.1  # Lower threshold to include more content
        for chunk, score in chunks_with_scores:
            print(f"Evaluating chunk with similarity {score}")
            if score >= min_similarity:
                context_parts.append(chunk['content'])
                print(f"Including chunk with similarity: {score}")
            else:
                print(f"Skipping chunk with low similarity: {score}")
                
        if not context_parts and chunks_with_scores:  # If no chunks meet threshold, take the best one
            best_chunk = chunks_with_scores[0]
            context_parts.append(best_chunk[0]['content'])
            print(f"Including best chunk with similarity: {best_chunk[1]}")
        
        if not context_parts:
            return "No relevant content found for the query."

        context = "\n".join(context_parts)
        if query:
            prompt = f"Question: {query}\n\nRelevant information:\n{context}\n\nAnswer:"
        else:
            prompt = f"Summarize the following information:\n{context}"

        # Process text in chunks suitable for the model
        max_chunk_chars = 500  # Reduced for better handling
        text_parts = []
        
        # Split text while trying to maintain sentence boundaries
        sentences = prompt.replace('\n', ' ').split('.')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_chars and current_chunk:
                text_parts.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence + "."
        
        if current_chunk:
            text_parts.append(current_chunk.strip())
        
        print(f"Split into {len(text_parts)} parts for summarization.")

        # Process each part
        summaries = []
        for part in text_parts:
            if len(part.split()) < 10:  # Skip very short segments
                continue
            try:
                summary = summarizer(
                    part,
                    max_length=150,
                    min_length=30,
                    do_sample=False,
                    truncation=True
                )
                if summary and isinstance(summary, list) and len(summary) > 0:
                    summaries.append(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error summarizing part: {str(e)}")
        
        print(f"Generated {len(summaries)} summaries.")
        if not summaries:
            print("No summaries generated.")
            return "No summaries generated."

        return "\n".join(summaries)

    except Exception as e:
        return f"Summarization failed: {str(e)}"
