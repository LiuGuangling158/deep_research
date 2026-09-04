# GitHub Codespaces Temporary Demo

This guide runs the full DeepResearch demo in GitHub Codespaces without replacing Milvus. It is intended for temporary demos, not always-on production hosting.

## 1. Create a Codespace

Push this repository to GitHub, then open the repository page and choose:

Code -> Codespaces -> Create codespace on main

Choose a 4-core / 8 GB machine if GitHub offers the option. Milvus, MinIO, etcd, PostgreSQL, the backend, and the frontend all run in the same Codespace.

## 2. Configure secrets

In the Codespaces terminal:

```bash
cp .env.codespaces.example .env.codespaces
```

Edit `.env.codespaces` and fill at least:

```env
DEEPSEEK_API_KEY=your_deepseek_key
```

Milvus local knowledge ingestion still uses DashScope embeddings by default. If you need local RAG in the demo, also fill one of these:

```env
DASHSCOPE_API_KEY=your_dashscope_key
EMBEDDING_API_KEY=your_dashscope_key
```

Optional web search:

```env
BOCHA_API_KEY=your_bocha_key
```

Do not commit `.env.codespaces`.

## 3. Start all services

```bash
docker compose -f docker-compose.codespaces.yml up -d --build
```

Check status:

```bash
docker compose -f docker-compose.codespaces.yml ps
```

Wait until `postgres`, `milvus-standalone`, and `deepresearch-backend` are healthy.

## 4. Ingest demo documents into Milvus

The compose file mounts `./demo_docs` to `/workspace/docs` inside the backend container.

```bash
docker compose -f docker-compose.codespaces.yml run --rm deepresearch-backend \
  python -m mult_agents.rag.ingest --input /workspace/docs
```

You can replace `demo_docs` with your own `.txt`, `.md`, or `.markdown` files before running ingestion.

## 5. Open and share the demo

In the Codespaces Ports panel:

1. Find port `8080`, labeled `DeepResearch demo`.
2. Right-click it and choose `Port Visibility -> Public`.
3. Copy the forwarded URL.

The public URL usually looks like:

```text
https://<codespace-name>-8080.app.github.dev
```

Share that URL for the live demo.

Keep port `8000` private unless you specifically need to debug the backend API.

## 6. Stop or delete when finished

Stop services:

```bash
docker compose -f docker-compose.codespaces.yml down
```

Delete volumes and demo data stored in containers:

```bash
docker compose -f docker-compose.codespaces.yml down -v
```

After the demo, stop or delete the Codespace from GitHub to save free quota.

## Troubleshooting

If the frontend opens but requests fail, check backend logs:

```bash
docker compose -f docker-compose.codespaces.yml logs -f deepresearch-backend
```

If Milvus is still starting, wait one or two minutes and run:

```bash
docker compose -f docker-compose.codespaces.yml ps
```

If ingestion fails with `EMBEDDING_API_KEY` or `DASHSCOPE_API_KEY`, check `.env.codespaces`.

If the demo URL asks viewers to sign in, make sure port `8080` is set to `Public` in the Codespaces Ports panel.
