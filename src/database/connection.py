import sys
import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.core.config import VECTOREDB_ENDPOINT, VECTOREDB_API_KEY

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:

    if VECTOREDB_ENDPOINT:
        logger.info(f"Connecting to Qdrant at endpoint: {VECTOREDB_ENDPOINT}")
        return QdrantClient(
            url=VECTOREDB_ENDPOINT,
            api_key=VECTOREDB_API_KEY,
            timeout=60.0
        )


def init_collection(collection_name: str, vector_size: int = 3072, distance: str = "Cosine", recreate: bool = False) -> bool:

    client = get_qdrant_client()
    
    distance_mapping = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID
    }
    dist_enum = distance_mapping.get(distance.lower(), models.Distance.COSINE)

    collections = [c.name for c in client.get_collections().collections]
    
    if collection_name in collections:
        if recreate:
            logger.info(f"Recreating existing collection '{collection_name}'...")
            client.delete_collection(collection_name=collection_name)
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
            return True

    logger.info(f"Creating Qdrant collection '{collection_name}' (dim={vector_size}, distance={distance})...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=dist_enum
        )
    )
    return True

