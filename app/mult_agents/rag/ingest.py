import argparse
import logging
import os
import sys
from pathlib import Path

# 将项目 app 目录添加到 PYTHONPATH，支持直接运行脚本与 python -m 两种方式。
project_root = Path(__file__).resolve().parents[3]
app_root = project_root / "app"
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

# 先加载 .env，再导入其他模块，确保 Milvus 与模型配置可用。
from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from mult_agents.config import AppConfig
from mult_agents.rag.core import RAGConfig, RAGSystem





DEFAULT_INPUT_PATH = "/workspace/docs"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v1"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def _collect_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    patterns = ("*.txt", "*.md", "*.markdown")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(input_path.rglob(pattern)))
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest local text documents into Milvus")
    parser.add_argument(
        "--input",
        default=os.getenv("RAG_INPUT_PATH", DEFAULT_INPUT_PATH),
        help="File or directory containing .txt, .md, or .markdown documents",
    )
    parser.add_argument("--collection", default="", help="Milvus collection name")
    parser.add_argument("--milvus-host", default="", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=0, help="Milvus port")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = _parse_args()
    config = AppConfig.from_file()
    collection_name = args.collection or config.milvus_collection
    milvus_host = args.milvus_host or config.milvus_host
    milvus_port = args.milvus_port or config.milvus_port
    rag_cfg = RAGConfig(
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    rag = RAGSystem(api_key=config.api_key, config=rag_cfg)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    paths = _collect_paths(input_path)
    if not paths:
        raise ValueError(f"未找到可入库文件: {input_path}")

    total_chunks = rag.ingest_paths(paths)
    print(f"入库完成 | 文件数={len(paths)} | chunk数={total_chunks} | collection={collection_name}")


if __name__ == "__main__":
    main()
