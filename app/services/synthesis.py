import os
import logging
import re
from typing import List, Dict, Union, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    raise

# Type alias for chunk structure
ChunkType = Dict[str, Any]

def format_response(text: str) -> str:
    """
    Format the response text to ensure proper line breaks and readability.
    
    Args:
        text: The text to format
        
    Returns:
        str: Formatted text with proper line breaks
    """
    if not text:
        return text
        
    # Split into sentences and clean up
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Add line breaks between sentences
    formatted = "\n".join(sentences)
    
    # Ensure bullet points are on new lines
    formatted = re.sub(r'(?<!\n)•', '\n•', formatted)
    
    # Clean up multiple newlines
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    
    return formatted.strip()

def summarize_themes(results: List[ChunkType], query: str) -> str:
    """
    Synthesize search results into a coherent answer using GPT.
    
    Args:
        results: List of search results with content and metadata
        query: The original user query
        
    Returns:
        str: Synthesized answer based on the search results
    """
    try:
        if not results:
            return "No relevant information found."
            
        # Combine relevant chunks into context
        context_parts = []
        for chunk in results:
            try:
                meta = chunk.get('meta', {})
                if isinstance(meta, dict):
                    source = meta.get('source', 'unknown source')
                    content = chunk.get('content', '')
                    if content:
                        context_parts.append(f"From {source}:\n{content}")
            except Exception as e:
                logger.warning(f"Error processing chunk: {e}")
                continue
        
        if not context_parts:
            return "Could not process any of the search results."
            
        context = "\n\n".join(context_parts)
        
        # Create system prompt
        system_prompt = (
            "You are a helpful assistant. Follow these rules strictly:\n"
            "1. Answer based only on the provided context.\n"
            "2. Start each new sentence on a new line.\n"
            "3. Keep answers concise and to the point.\n"
            "4. If information is not in the context, say 'I don't have enough information.'\n"
            "5. Use bullet points for lists.\n"
            "6. Maintain a professional tone."
        )
        
        # Get completion from OpenAI
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using faster model for better response time
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
                ],
                temperature=0.3,
                max_tokens=500,  # Reduced for faster response
                presence_penalty=0.1,  # Slight penalty for repetition
                frequency_penalty=0.1   # Slight penalty for repetition
            )
            
            answer = completion.choices[0].message.content
            if answer is None:
                return "Error: No response generated"
                
            # Format the response with proper line breaks
            return format_response(answer)
            
        except Exception as gpt_error:
            logger.error(f"Error calling OpenAI API: {str(gpt_error)}")
            # Fallback to a simpler response using the most relevant chunk
            if results:
                return format_response(
                    f"API Error. Here's the most relevant excerpt:\n\n"
                    f"{results[0].get('content', 'No content available')}"
                )
            return "Error generating response and no fallback content available."
        
    except Exception as e:
        logger.error(f"Error in summarize_themes: {str(e)}")
        return f"Error generating summary: {str(e)}"
