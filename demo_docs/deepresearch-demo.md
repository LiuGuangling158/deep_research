# DeepResearch Demo Knowledge Base

DeepResearch is a multi-agent research assistant. It uses FastAPI as the backend service and a Vue 3 Vite frontend served by Nginx in production.

The backend exposes `/health` for service checks and `/api/v1/research/stream` for streaming research responses.

The research workflow routes simple questions to a direct answer path. Complex questions go through planning, web evidence retrieval, local knowledge retrieval, evidence review, analysis, reflection if needed, and final writing.

For the Codespaces demo, PostgreSQL stores short-term and long-term memory plus LangGraph checkpoints. Milvus stores vector embeddings for local knowledge retrieval. Redis is optional and disabled in the default demo configuration to reduce resource usage.

The frontend should be accessed through port 8080 in GitHub Codespaces. Its Nginx configuration proxies `/api` and `/health` to the backend service named `deepresearch-backend`.
