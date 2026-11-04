from functools import cache
import json
import os
from typing import cast

from debate_settings import CONFIGURATION_DIRECTORY, CONFIGURATION_NAME
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore
from langchain_community.storage import RedisStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ConfigManager:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as file_stream:
            self._config: dict[str, object] = json.load(file_stream)

    @staticmethod
    def _descend(node: object, key: str, key_path: str) -> object:
        if not isinstance(node, dict):
            raise KeyError(key_path)
        table = cast(dict[str, object], node)
        return table[key]

    def _find(self, key_path: str) -> object:
        current: object = self._config
        for key in key_path.split("."):
            current = self._descend(current, key, key_path)
        return current

    def read_text(self, key_path: str) -> str:
        value = self._find(key_path)
        if isinstance(value, list):
            return "\n".join(cast(list[str], value))
        return str(value)

    def read_integer(self, key_path: str) -> int:
        return int(cast(int, self._find(key_path)))

    def read_number(self, key_path: str) -> float:
        return float(cast(float, self._find(key_path)))


config = ConfigManager(f"{CONFIGURATION_DIRECTORY}/{CONFIGURATION_NAME}.json")


def _encode_document(document: Document) -> bytes:
    return json.dumps({
        "page_content": document.page_content,
        "metadata": document.metadata,
    }).encode("utf-8")


def _decode_document(document_bytes: bytes) -> Document:
    data = json.loads(document_bytes.decode("utf-8"))
    return Document(
        page_content=data["page_content"],
        metadata=data.get("metadata", {}),
    )


def _keep_key(key: str) -> str:
    return key


@cache
def get_chat_model(role: str) -> BaseChatModel:
    provider = config.read_text(f"models.{role}.provider")
    if provider == "ollama":
        return ChatOllama(
            base_url=os.environ.get("OLLAMA_HOST"),
            model=config.read_text(f"models.{role}.llm"),
            temperature=config.read_number(f"models.{role}.temperature"),
            num_ctx=config.read_integer(f"models.{role}.context_window"),
        )
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.read_text(f"models.{role}.llm"),
            temperature=config.read_number(f"models.{role}.temperature"),
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


@cache
def get_embeddings_model() -> Embeddings:
    provider = config.read_text("embedding.provider")
    model_name = config.read_text("embedding.model")
    if provider == "ollama":
        return OllamaEmbeddings(
            base_url=os.environ.get("OLLAMA_HOST"),
            model=model_name,
        )
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(model=model_name)
    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"trust_remote_code": True},
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def _build_document_store(
    agent_role: str,
    debate_id: str,
) -> EncoderBackedStore[str, Document]:
    redis_url = os.environ.get("REDIS_URL") or config.read_text("database.redis_url")
    underlying_store = RedisStore(
        redis_url=redis_url,
        namespace=f"docstore_{debate_id}_{agent_role}",
    )
    return EncoderBackedStore(
        store=underlying_store,
        key_encoder=_keep_key,
        value_serializer=_encode_document,
        value_deserializer=_decode_document,
    )


def _build_splitter(level: str) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.read_integer(f"embedding.{level}_chunk_size"),
        chunk_overlap=config.read_integer(f"embedding.{level}_chunk_overlap"),
    )


def get_retriever(agent_role: str, debate_id: str) -> ParentDocumentRetriever:
    vectorstore = Chroma(
        collection_name=f"debate_{debate_id}_agent_{agent_role}",
        embedding_function=get_embeddings_model(),
        persist_directory=config.read_text("database.path"),
    )
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=_build_document_store(agent_role, debate_id),
        child_splitter=_build_splitter("child"),
        parent_splitter=_build_splitter("parent"),
    )
