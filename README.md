# Debate-GPT

A multi-agent AI debate platform. Two LLMs argue opposing sides of any
topic you give it, a third (more powerful) model judges each round on
logic, evidence, and persuasion, and a final verdict is declared —
streamed live to the browser.

Built as a portfolio project demonstrating multi-agent orchestration,
LLM-as-Judge evaluation, structured output enforcement, real-time
streaming, and full-stack AI engineering — end to end, including the
failure modes: bias mitigation, judge consistency, and length-imbalance
bugs, all measured with a real eval suite rather than assumed away.

**Live demo:** https://debate-gpt-tau.vercel.app
**Backend API:** https://debate-gpt-production.up.railway.app

> ⚠️ This is a shared demo environment with no authentication — every
> visitor sees the same debate history. Don't treat it as private.

## How it works

1. You submit a topic → the backend creates a session and kicks off a
   background task
2. A LangGraph state machine runs the debate: `Pro → Con → Judge`,
   looping for N rounds
3. The Judge never sees the words "Pro" or "Con" — only "Speaker A" and
   "Speaker B" — to prevent position-label bias in scoring
4. Every token is streamed live to the browser via Server-Sent Events
   (backed by Redis Streams, since Upstash's REST API can't hold a
   `SUBSCRIBE` connection)
5. The final transcript, scores, and winner are persisted to Postgres

## Quick start (local)

```bash
# 1. Install
python -m venv venv
venv\Scripts\activate             # Windows
source venv/bin/activate          # macOS/Linux
pip install -r requirements.txt

# 2. Configure
cp .env.example .env              # fill in your real API keys

# 3. Apply the DB schema (Neon Postgres)
alembic upgrade head

# 4. Run the backend
python run_api.py                 # http://localhost:8000

# 5. Run the frontend (separate terminal)
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

## Project structure

```
debate-gpt/
├── src/debate_gpt/        # Backend — the actual system
│   ├── graph.py           #   LangGraph wiring: Pro → Con → Judge → loop
│   ├── agents.py          #   LLM clients + node logic (streaming, retries,
│   │                      #   length enforcement)
│   ├── prompts.py         #   System prompts for all 3 agents + persona styles
│   ├── schemas.py         #   RoundScore — the judge's structured output contract
│   ├── verdict.py         #   Score tallying, swap-aware Pro/Con totals
│   ├── state.py           #   The LangGraph state definition
│   ├── api.py             #   FastAPI routes (start/stream/result/list/delete/health)
│   ├── db.py               #   Postgres access (asyncpg)
│   ├── redis_stream.py    #   Upstash Redis Streams wrapper (SSE backing store)
│   └── observability/     #   Logging + /health
│
├── frontend/               # React + Vite + Tailwind UI
│   └── src/
│       ├── pages/          #   HomePage (new debate), DebatePage (live view)
│       ├── components/     #   SessionSidebar, DebateForm, RoundCard,
│       │                   #   JudgeScorecard, VerdictBanner
│       └── hooks/          #   useDebateStream — the SSE consumer
│
├── tests/
│   ├── unit/                #   Pure-Python tests (schemas, verdict, prompts, etc.)
│   ├── integration/         #   Full FastAPI app tests via httpx + in-memory fakes
│   └── evals/                #   Real-LLM eval suite (judge consistency, position
│                              #   bias) — see eval_results.csv for actual findings
│
├── migrations/              # Alembic — Postgres schema
├── Dockerfile               # Multi-stage build, deployed to Railway
├── PRD.md                   # Full product requirements doc
└── PROGRESS.md              # Sprint tracker (Days 1-7)
```

## Eval findings

The eval suite (`tests/evals/`, real LLM calls, results in
`eval_results.csv` / `eval_results_raw.json`) surfaced a genuine finding
worth calling out rather than hiding:

- **Judge consistency**: PASS — scoring the same argument pair 5 times
  produced a max standard deviation of 0.7 points per criterion (well
  under the 1.5 threshold).
- **Position bias**: FAIL — swapping which side is labeled "Speaker A"
  vs "Speaker B" changed the derived Pro/Con totals by up to 3 points
  (threshold was ±1.0). Digging into the raw scores showed this isn't
  actually *pro/con* bias — the Speaker A/B relabeling works as intended
  — it's a **presentation-order bias**: the judge consistently scores
  whichever argument is presented *second* in the prompt more
  favorably, regardless of which side occupies that slot. The
  mitigation in place stops position-*label* bias but not
  presentation-*order* bias; randomizing which side is shown first
  (not just which label it gets) would be the next fix.

## Running tests

```bash
python -m pytest tests/ --cov=src/debate_gpt --cov-report=term-missing
```
