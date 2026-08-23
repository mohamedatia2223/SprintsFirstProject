import sys
import os
import re
import uuid
import time
import logging
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qdrant_client.http import models
from src.core.embedding import embed_documents
from src.database.connection import get_qdrant_client, init_collection
from src.core.pdf_to_text import extract_pdf_to_markdown

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rich_dad_poor_dad"
DEFAULT_CHUNK_SIZE = 800  
DEFAULT_CHUNK_OVERLAP = 150 


def parse_markdown_into_chunks(
    markdown_text: str, 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:

    chunks = []
    
    parts = re.split(r'<!-- Page (\d+) -->', markdown_text)
    
    current_section = "General"
    
    if len(parts) < 3:
        page_tuples = [(1, markdown_text)]
    else:
        page_tuples = []
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            page_content = parts[i+1] if i+1 < len(parts) else ""
            page_tuples.append((page_num, page_content))

    for page_num, page_content in page_tuples:
        lines = page_content.splitlines()
        page_text_blocks = []
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                current_section = re.sub(r'^#+\s*', '', stripped)
            elif stripped and not stripped.startswith("---"):
                page_text_blocks.append(stripped)

        clean_page_text = " ".join(page_text_blocks)
        if not clean_page_text:
            continue

        start = 0
        text_length = len(clean_page_text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            if end < text_length:
                next_space = clean_page_text.rfind(' ', start, end)
                if next_space > start + (chunk_size // 2):
                    end = next_space

            chunk_str = clean_page_text[start:end].strip()

            if chunk_str:
                chunks.append({
                    "text": chunk_str,
                    "page_number": page_num,
                    "section_title": current_section,
                    "char_count": len(chunk_str)
                })

            if end >= text_length:
                break

            start = end - overlap

    return chunks


def ingest_document(
    md_file_path: str = os.path.join("data", "rich_dad_poor_dad_by_robert_t-_kiyosaki.md"),
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 20,
    recreate: bool = True
) -> int:

    if not os.path.exists(md_file_path):
        pdf_path = os.path.join("data", "rich_dad_poor_dad_by_robert_t-_kiyosaki.pdf")
        if os.path.exists(pdf_path):
            logger.info(f"Markdown file '{md_file_path}' not found. Generating from '{pdf_path}'...")
            extract_pdf_to_markdown(pdf_path, output_path=md_file_path)
        else:
            raise FileNotFoundError(f"Source file not found at: {md_file_path}")

    with open(md_file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    logger.info(f"Parsing '{md_file_path}' into chunks...")
    chunks = parse_markdown_into_chunks(markdown_text)
    logger.info(f"Generated {len(chunks)} text chunks.")

    if not chunks:
        logger.warning("No chunks generated. Aborting ingestion.")
        return 0

    init_collection(collection_name=collection_name, vector_size=3072, recreate=recreate)

    client = get_qdrant_client()

    total_ingested = 0
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = [c["text"] for c in batch_chunks]

        logger.info(f"Embedding batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size} ({len(batch_texts)} chunks)...")
        embeddings = embed_documents(batch_texts)

        points = []
        for idx, (chunk, vector) in enumerate(zip(batch_chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{collection_name}_{chunk['page_number']}_{i + idx}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "page_number": chunk["page_number"],
                        "section_title": chunk["section_title"],
                        "chunk_index": i + idx
                    }
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        total_ingested += len(points)
        logger.info(f"Upserted {total_ingested}/{len(chunks)} points to Qdrant.")
        time.sleep(1.5) 

    logger.info(f"Successfully completed ingestion of {total_ingested} points into Qdrant collection '{collection_name}'.")
    return total_ingested


