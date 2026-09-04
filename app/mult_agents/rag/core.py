import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections, utility

logger = logging.getLogger(__name__)


# 使用 langchain-milvus 新包（类名是 Milvus，不是 MilvusVectorStore）
try:
    from langchain_milvus import Milvus as _MilvusVectorStore
    _MILVUS_BACKEND = "langchain_milvus"
except ImportError:
    from langchain_community.vectorstores import Milvus as _MilvusVectorStore
    _MILVUS_BACKEND = "langchain_community"


def build_embeddings(api_key: str, model: str, base_url: str = ""):
    """Build embeddings for DashScope or an OpenAI-compatible endpoint."""
    cleaned_api_key = (api_key or "").strip()
    cleaned_base_url = (base_url or "").strip()
    if not cleaned_api_key:
        raise ValueError("缺少 EMBEDDING_API_KEY 或 DASHSCOPE_API_KEY，无法初始化向量模型")
    if cleaned_base_url:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("缺少 langchain-openai 依赖，请安装: pip install langchain-openai") from exc
        return OpenAIEmbeddings(
            model=model,
            api_key=cleaned_api_key,
            base_url=cleaned_base_url,
        )
    return DashScopeEmbeddings(
        model=model,
        dashscope_api_key=cleaned_api_key,
    )


@dataclass(frozen=True)
class RAGConfig:
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    collection_name: str = "mult_agent_knowledge"
    embedding_model: str = "text-embedding-v1"
    embedding_base_url: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 50


class RAGSystem:
    def __init__(self, api_key: str, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.api_key = api_key
        self.embeddings = build_embeddings(
            api_key=self.api_key,
            model=self.config.embedding_model,
            base_url=self.config.embedding_base_url,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        self._connect_to_milvus()
        self.vectorstore = _MilvusVectorStore(
            embedding_function=self.embeddings,
            collection_name=self.config.collection_name,
            connection_args={"uri": f"http://{self.config.milvus_host}:{self.config.milvus_port}"},
            auto_id=True,
        )
        embedding_backend = "openai_compatible" if self.config.embedding_base_url else "dashscope"
        logger.info(
            "RAG backend=%s | collection=%s | embeddings=%s | base_url_configured=%s",
            _MILVUS_BACKEND,
            self.config.collection_name,
            embedding_backend,
            bool(self.config.embedding_base_url),
        )

    def _connect_to_milvus(self) -> None:
        try:
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port,
            )
        except Exception as exc:
            logger.error("连接 Milvus 失败: %s", exc)

    def search(self, query: str, k: int = 3) -> str:
        try:
            records = self.search_records(query, k=k)
            if not records:
                return "未找到相关信息。"
            lines: list[str] = ["检索到的相关信息："]
            for idx, record in enumerate(records, 1):
                lines.append(f"{idx}. {record['snippet']}")
                lines.append(f"   (来源: {record['doc_id']})")
            return "\n".join(lines)
        except Exception as exc:
            logger.error("检索失败: %s", exc)
            return f"检索过程中发生错误: {str(exc)}"

    def search_records(self, query: str, k: int = 5) -> list[dict]:
        if not utility.has_collection(self.config.collection_name):
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        records: list[dict] = []
        for idx, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            source = str(metadata.get("source") or "").strip()
            title = Path(source).name if source else f"本地知识片段-{idx}"
            records.append(
                {
                    "source_id": f"LOC-{idx}",
                    "doc_id": source,
                    "title": title,
                    "snippet": doc.page_content,
                    "source_type": "local",
                    "metadata": metadata,
                }
            )
        return records

    def add_documents(self, documents: list[Document]) -> int:
        self.vectorstore.add_documents(documents)
        return len(documents)

    def ingest_text(self, text: str, source: str) -> int:
        docs = self.text_splitter.create_documents([text], metadatas=[{"source": source}])
        return self.add_documents(docs)

    def ingest_paths(self, paths: Iterable[Path]) -> int:
        total = 0
        for path in paths:
            text = path.read_text(encoding="utf-8")
            total += self.ingest_text(text, source=str(path))
        return total
