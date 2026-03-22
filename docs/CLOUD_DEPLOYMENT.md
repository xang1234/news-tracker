# Cloud Deployment Guide (Free Tier)

Deploy the full news-tracker stack for **$0/month** using free tiers from Oracle Cloud, Neon, and Vercel.

## Architecture

```
┌─────────────┐      ┌──────────────────────────────────────────┐
│   Vercel     │      │  Oracle Cloud Always Free ARM VM          │
│  (Frontend)  │─────▶│  ┌─────────┐ ┌──────────┐ ┌───────────┐ │
│  React SPA   │ /api │  │ FastAPI  │ │ Embedding│ │ Sentiment │ │
└─────────────┘      │  │  :8001   │ │  Worker  │ │  Worker   │ │
                      │  └────┬────┘ └──────────┘ └───────────┘ │
                      │       │      ┌──────────┐ ┌───────────┐ │
                      │       │      │Clustering│ │ Ingestion │ │
                      │       │      │  Worker  │ │  Worker   │ │
                      │  ┌────┴────┐ └──────────┘ └───────────┘ │
                      │  │  Redis  │  (4 ARM cores, 24GB RAM)    │
                      │  └─────────┘                              │
                      └──────────────┬───────────────────────────┘
                                     │
                      ┌──────────────┴───────────────┐
                      │  Neon (PostgreSQL + pgvector)  │
                      │  0.5GB storage, auto-suspend   │
                      └────────────────────────────────┘
```

## Platform Selection

| Component | Platform | Free Tier Limits |
|-----------|----------|-----------------|
| **Database** | [Neon](https://neon.tech) | 0.5GB storage, 190 compute-hrs/mo, pgvector + pg_trgm |
| **Backend + Redis** | [Oracle Cloud Always Free](https://www.oracle.com/cloud/free/) | 4 ARM A1 cores, 24GB RAM, 200GB disk — permanent |
| **Frontend** | [Vercel](https://vercel.com) | Unlimited static hosting, 100GB bandwidth |

### Why These Choices

- **Oracle Always Free ARM** is the only free tier with enough RAM (24GB) for ML inference (FinBERT + MiniLM need ~3GB combined). No other free platform comes close.
- **Neon** supports pgvector natively on free tier. Auto-suspends after 5 min idle to save compute hours. Supabase is an alternative but pauses projects after 1 week of inactivity.
- **Vercel** handles the static React SPA with zero config and proxies `/api` requests to the backend.

### Alternatives

| Scenario | Swap | Tradeoff |
|----------|------|----------|
| Prefer Supabase | Neon → Supabase (500MB, pgvector) | Pauses after 1 week idle; ping regularly |
| No ML needed | Oracle → Render/Railway (512MB) | API-only, no workers, no embeddings |
| Want monitoring | Add Grafana Cloud free tier | 10K metrics, 50GB logs |
| Budget ~$5/mo | Oracle → Railway ($5 credit) | Simpler deployment, less RAM |

---

## Step 1: Set Up Neon Database

1. Create account at [neon.tech](https://neon.tech)
2. Create a project (select region closest to your Oracle VM)
3. Note the connection string from **Connection Details**

### Enable Extensions

Extensions are pre-installed on Neon. Verify in the SQL Editor:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

### Run Migrations

```bash
export DATABASE_URL="postgresql://user:pass@ep-xxx.neon.tech/news_tracker?sslmode=require"
bash deployments/cloud/neon-init.sh
```

Or use the Neon SQL Editor to paste and run each file from `migrations/` in order.

### Verify

```bash
psql "$DATABASE_URL" -c '\dt'
# Should show: documents, themes, theme_metrics, alerts, etc.
```

---

## Step 2: Set Up Oracle Cloud ARM VM

### Create the Instance

1. Sign up at [oracle.com/cloud/free](https://www.oracle.com/cloud/free/)
2. Go to **Compute → Instances → Create Instance**
3. Configure:
   - **Shape**: VM.Standard.A1.Flex (Ampere ARM)
   - **OCPUs**: 4, **Memory**: 24 GB
   - **Image**: Ubuntu 22.04 (aarch64)
   - **Boot volume**: 200 GB
   - **Networking**: Create VCN with public subnet, assign public IP
4. Add SSH key and launch

### Configure Network Security

In **Networking → Virtual Cloud Networks → Security Lists**, add ingress rules:

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP (Caddy redirect) |
| 443 | TCP | 0.0.0.0/0 | HTTPS (Caddy) |
| 8001 | TCP | 0.0.0.0/0 | API (direct, optional) |

### Install Docker and Caddy

```bash
ssh ubuntu@<vm-ip> 'bash -s' < deployments/cloud/setup-oracle-vm.sh
# Log out and back in for docker group to take effect
```

### Deploy the Application

```bash
ssh ubuntu@<vm-ip>
git clone <your-repo-url> ~/news-tracker
cd ~/news-tracker

# Configure environment
cp .env.cloud.example .env
nano .env  # Set DATABASE_URL, API_KEYS, CORS_ORIGINS, etc.

# Initialize database (runs migrations via the app)
docker compose -f docker-compose.cloud.yml run --rm news-tracker-api init-db

# Start all services
docker compose -f docker-compose.cloud.yml up -d --build

# Verify
docker compose -f docker-compose.cloud.yml ps
curl http://localhost:8001/health
```

### Set Up HTTPS with Caddy

```bash
sudo tee /etc/caddy/Caddyfile > /dev/null <<'EOF'
your-domain.com {
    reverse_proxy localhost:8001
}
EOF
sudo systemctl restart caddy
```

If you don't have a domain, use the VM's public IP directly (HTTP only):

```bash
curl http://<vm-public-ip>:8001/health
```

### Resource Usage (Expected)

```
$ docker stats --no-stream
CONTAINER         CPU %   MEM USAGE
news-tracker-api   ~5%    ~800MB (lazy-loaded, grows on first ML request)
embedding-worker   ~10%   ~2.5GB (FinBERT + MiniLM loaded)
sentiment-worker   ~10%   ~2.0GB (FinBERT loaded)
clustering-worker  ~2%    ~500MB
worker             ~3%    ~400MB
redis              ~1%    ~100MB
                          ─────────
                   Total: ~6.3GB of 24GB available
```

---

## Step 3: Deploy Frontend on Vercel

### Option A: Vercel Dashboard

1. Go to [vercel.com](https://vercel.com), import your GitHub repo
2. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. No environment variables needed (API URL is in `vercel.json` rewrites)

### Option B: Vercel CLI

```bash
cd frontend
npx vercel --prod
```

### Configure API Proxy

Edit `frontend/vercel.json` and replace `ORACLE_VM_IP` with your VM's domain or IP:

```json
{
  "rewrites": [
    { "source": "/api/:path*", "destination": "https://your-domain.com/:path*" }
  ]
}
```

### WebSocket Note

Vercel free tier does **not** proxy WebSocket connections. If you enable `WS_ALERTS_ENABLED`, the frontend must connect directly to the backend for WebSocket. Set `VITE_WS_URL` as a Vercel environment variable pointing to `wss://your-domain.com`.

---

## Step 4: Verify End-to-End

```bash
# 1. Health check
curl https://your-domain.com/health
# → {"status": "healthy", ...}

# 2. Run a test ingestion cycle
ssh ubuntu@<vm-ip> 'cd ~/news-tracker && docker compose -f docker-compose.cloud.yml exec worker news-tracker run-once --mock'

# 3. Check documents appeared
curl https://your-domain.com/documents

# 4. Visit the frontend
open https://your-app.vercel.app
```

---

## Managing Neon Compute Hours

Neon free tier gives 190 compute-hours/month. The database auto-suspends after 5 minutes of inactivity, which helps significantly. Tips:

- **Set `POLL_INTERVAL_SECONDS=3600`** (1 hour) to avoid keeping the DB awake with frequent polls
- **Reduce `DB_POOL_MIN_SIZE=2`** so idle connections don't hold the DB open
- **Monitor usage** in the Neon dashboard → Project → Usage
- At 190 hrs/month, you can keep the DB active ~6.3 hrs/day continuously. With auto-suspend, typical usage is well under this.

---

## Managing Neon Storage (0.5GB)

Each document with embeddings ≈ 6-10KB. Budget:

| Documents | Storage | Timeline (100 docs/day) |
|-----------|---------|------------------------|
| 10,000 | ~80 MB | ~3 months |
| 50,000 | ~400 MB | ~1.5 years |

Keep storage in check:
```bash
# Remove documents older than 90 days
ssh ubuntu@<vm-ip> 'cd ~/news-tracker && docker compose -f docker-compose.cloud.yml exec news-tracker-api news-tracker cleanup --days 90'
```

---

## Enabling Features

All features are disabled by default. Enable in `.env` on the Oracle VM:

```bash
# Edit .env
nano .env

# Enable features incrementally
CLUSTERING_ENABLED=true    # Theme clustering (needs embedding-worker running)
NER_ENABLED=true           # Named entity recognition
ALERTS_ENABLED=true        # Alert generation
SOURCES_ENABLED=true       # Database-backed source management

# Restart to apply
docker compose -f docker-compose.cloud.yml up -d
```

---

## Troubleshooting

### ARM Build Failures

If Docker build fails on ARM for native extensions:

```bash
# Ensure buildx is using the correct platform
docker buildx create --use --platform linux/arm64
```

### Neon Connection Issues

```bash
# Test connectivity from Oracle VM
docker compose -f docker-compose.cloud.yml exec news-tracker-api python3 -c "
import asyncio, asyncpg
async def test():
    conn = await asyncpg.connect('$DATABASE_URL')
    print(await conn.fetchval('SELECT version()'))
    await conn.close()
asyncio.run(test())
"
```

### Model Download Slow

First startup downloads FinBERT (~440MB) and MiniLM (~80MB) from HuggingFace. This can take 5-10 minutes on a slow connection. Models are cached in the Docker volume after first download.

### Out of Memory

If workers OOM on the 24GB VM (unlikely), reduce batch sizes:

```bash
EMBEDDING_BATCH_SIZE=8    # Default 32, reduce for less memory
SENTIMENT_BATCH_SIZE=4    # Default 16
```

Or disable workers you don't need by stopping individual containers:

```bash
docker compose -f docker-compose.cloud.yml stop sentiment-worker
```
