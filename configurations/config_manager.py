from functools import lru_cache
import json
import os
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore
from langchain_community.storage import RedisStore
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import redis


MODEL = "reasoning_gemini_llm"


class ConfigManager:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as file_stream:
            self._config: dict[str, Any] = json.load(file_stream)

    def get(self, key_path: str, default_value: Any = None) -> Any:
        keys = key_path.split(".")
        curr_json = self._config
        for key in keys:
            if isinstance(curr_json, dict) and key in curr_json:
                curr_json = curr_json[key]
            else:
                return default_value
        if isinstance(curr_json, list):
            return "\n".join(curr_json)
        return curr_json


config = ConfigManager(f"configurations/{MODEL}.json")


def _encode_doc(doc: Document) -> bytes:
    return json.dumps({
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }).encode("utf-8")


def _decode_doc(doc_bytes: bytes) -> Document:
    data = json.loads(doc_bytes.decode("utf-8"))
    return Document(
        page_content=data["page_content"],
        metadata=data.get("metadata", {})
    )


@lru_cache(maxsize=None)
def get_chat_model(role: str) -> BaseChatModel:
    provider = config.get(f"models.{role}.provider", "ollama")
    if provider == "ollama":
        return ChatOllama(
            base_url=os.environ.get("OLLAMA_HOST"),
            model=config.get(f"models.{role}.llm"),
            temperature=config.get(f"models.{role}.temperature"),
            num_ctx=config.get(f"models.{role}.context_window"),
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.get(f"models.{role}.llm"),
            temperature=config.get(f"models.{role}.temperature"),
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache(maxsize=1)
def get_embeddings_model() -> Embeddings:
    provider = config.get("embedding.provider", "ollama")
    if provider == "ollama":
        return OllamaEmbeddings(
            base_url=os.environ.get("OLLAMA_HOST"),
            model=config.get("embedding.model"),
        )
    elif provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=config.get("embedding.model")
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def get_retriever(agent_role: str, debate_id: str) -> ParentDocumentRetriever:
    embedder = get_embeddings_model()
    db_path = config.get("database.path", "./debate_vectors")
    
    vectorstore = Chroma(
        collection_name=f"debate_{debate_id}_agent_{agent_role}",
        embedding_function=embedder,
        persist_directory=db_path
    )
    
    redis_url = os.environ.get("REDIS_URL") or config.get(
        "database.redis_url", 
        "redis://localhost:6379/0",
    )
    redis_client = redis.Redis.from_url(redis_url)
    underlying_store = RedisStore(
        client=redis_client,
        namespace=f"docstore_{debate_id}_{agent_role}",
    )
    
    store = EncoderBackedStore(
        store=underlying_store,
        key_encoder=lambda k: k,
        value_serializer=_encode_doc,
        value_deserializer=_decode_doc
    )
    
    parent_chunk = config.get("embedding.parent_chunk_size", 2000)
    parent_overlap = config.get("embedding.parent_chunk_overlap", 0)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk, 
        chunk_overlap=parent_overlap
    )
    
    child_chunk = config.get("embedding.child_chunk_size", 400)
    child_overlap = config.get("embedding.child_chunk_overlap", 50)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk, 
        chunk_overlap=child_overlap
    )
    
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
