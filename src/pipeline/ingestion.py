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
    chunk_size: int = 800, 
    overlap: int = 150
) -> List[Dict[str, Any]]:

    chunks = []
    lines = markdown_text.splitlines()
    
    current_page = 1
    current_section = "General"
    
    blocks = []
    current_block_lines = []
    block_start_page = 1

    for line in lines:
        stripped = line.strip()
        
        # Detect page marker
        page_match = re.match(r'<!-- Page (\d+) -->', stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        # Detect section header
        if stripped.startswith("#"):
            if current_block_lines:
                blocks.append({
                    "text": " ".join(current_block_lines),
                    "page_number": block_start_page,
                    "section_title": current_section
                })
                current_block_lines = []
            current_section = re.sub(r'^#+\s*', '', stripped)
            block_start_page = current_page
            continue

        if not stripped or stripped.startswith("---"):
            if current_block_lines:
                blocks.append({
                    "text": " ".join(current_block_lines),
                    "page_number": block_start_page,
                    "section_title": current_section
                })
                current_block_lines = []
                block_start_page = current_page
            continue

        if not current_block_lines:
            block_start_page = current_page
        current_block_lines.append(stripped)

    if current_block_lines:
        blocks.append({
            "text": " ".join(current_block_lines),
            "page_number": block_start_page,
            "section_title": current_section
        })

    current_chunk_text = ""
    current_chunk_page = 1
    current_chunk_section = "General"
    
    for b in blocks:
        if not current_chunk_text:
            current_chunk_text = b["text"]
            current_chunk_page = b["page_number"]
            current_chunk_section = b["section_title"]
        elif len(current_chunk_text) + len(b["text"]) + 2 <= chunk_size:
            current_chunk_text += "\n\n" + b["text"]
        else:
            chunks.append({
                "text": current_chunk_text.strip(),
                "page_number": current_chunk_page,
                "section_title": current_chunk_section,
                "char_count": len(current_chunk_text)
            })
            
            if len(current_chunk_text) > overlap:
                overlap_text = current_chunk_text[-overlap:]
                space_idx = overlap_text.find(" ")
                if space_idx != -1 and space_idx < len(overlap_text) - 10:
                    overlap_text = overlap_text[space_idx + 1:]
            else:
                overlap_text = current_chunk_text

            current_chunk_text = overlap_text + "\n\n" + b["text"]
            current_chunk_page = b["page_number"]
            current_chunk_section = b["section_title"]

    if current_chunk_text.strip():
        chunks.append({
            "text": current_chunk_text.strip(),
            "page_number": current_chunk_page,
            "section_title": current_chunk_section,
            "char_count": len(current_chunk_text)
        })

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
        time.sleep(3.0) 

    logger.info(f"Successfully completed ingestion of {total_ingested} points into Qdrant collection '{collection_name}'.")
    return total_ingested


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_document()



