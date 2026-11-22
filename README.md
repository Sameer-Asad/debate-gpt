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



## How it works

1. You submit a topic → the backend creates a session and kicks off a
   background task
2. A LangGraph state machine runs the debate: `Pro → Con → Judge`,
   looping for N rounds
3. The Judge never sees the words "Pro" or "Con" — only "Speaker A" and
   "Speaker B" — to prevent position-label bias in scoring
   (see *Benchmark Results & Error Analysis* below for how well this works)
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
│   └── evals/                #   Real-LLM evals + bidirectional benchmark
│                              #   (benchmark_bidirectional.py)
│
├── eval_runs/               # Benchmark logs, summaries, per-pair CSVs
├── migrations/              # Alembic — Postgres schema
├── Dockerfile               # Multi-stage build, deployed to Railway
├── PRD.md                   # Full product requirements doc
└── PROGRESS.md              # Sprint tracker (Days 1-7)
```

## Benchmark Results & Error Analysis

**Setup.** 24 topics were generated (one Pro/Con pair each). 4 pairs were excluded using only word counts, following the 180-220 word rule from the debater prompt (the filter was applied after the data were collected, not based on scores): 3 pairs had an **empty** Con argument and 1 had a 68-word Con argument. That leaves **20 valid pairs**. Each pair was judged in **both orders** (Pro first, Con first) with **2 repeats per order** = 80 real judge calls, all logged in `eval_runs/`.

Models: Pro `nvidia/nemotron-3.5-lightning:free` (OpenRouter, MoE with 30B total / 3B active parameters), Con `openai/gpt-oss-20b` (Groq), Judge `openai/gpt-oss-120b` (Groq, temperature 0.2).

### 1. Position bias (points on a 0-30 per-side scale)

| Condition | Comparisons | Mean max-side delta | Max | Winner flips |
|---|---|---|---|---|
| **Before:** label masking only, order flipped (single judge call) | 40 | 2.75 | 8 | 35.0% |
| Noise floor: same order, judged again | 40 | 2.3 | 8 | 30.0% |
| **After:** bidirectional average, two independent runs | 20 | 1.65 | 4.5 | 25.0% |

- Signed second-position advantage (second speaker minus first speaker, single order): **+0.16 points**, 95% bootstrap CI [-0.487, 0.762]. The interval includes 0, so no systematic position bias was detected on these pairs. Per pair the direction is mixed.
- Flipping the order changed scores only slightly more than judging the *same* order again (2.75 vs 2.3). The earlier 3-4 point deltas are about the size of this judge's normal run-to-run noise.
- Bidirectional averaging lowered the run-to-run delta from 2.3 to 1.65 (about 28% lower). This is about what averaging two noisy calls gives (2.3 / sqrt(2) = 1.63), so we cannot credit it to order balancing. A 4-repeat run would separate the two effects.
- Order-free Pro-side advantage: **+1.24 points** (CI [0.362, 2.15]). Pro and Con are different models, so this mixes model quality with stance.

### 2. Judge consistency (sample std dev across repeats)

| Measure | Value |
|---|---|
| Mean sigma over all 6 criteria | 0.38 |
| Highest per-criterion mean sigma | 0.513 |
| Side-total sigma, single order | 1.069 |
| Side-total sigma, bidirectional average | 0.857 |
| Worst single sigma | 2.828 (pair 10: Con evidence scored 8 in one run and 4 in the other) |

### 3. Schema validity (Pydantic `RoundScore`)

| Valid on first try | Valid after fallback / retry | Never valid |
|---|---|---|
| 100.0% (80/80) | 100.0% | 0.0% |

The retry path was never triggered in this run, so this benchmark says nothing about how well retries work.

### 4. Error analysis: largest order gaps

| Pair | Topic | Gap | What happened |
|---|---|---|---|
| 19 | Self-driving cars | 6.0 | Noise: the same order (Con first) gave margins of -5 and +1 across two runs. |
| 2 | Nuclear power | 5.0 | Noise: same-order margins swung +3 to -1 and +4 to +8 (Con total 20 vs 16). |
| 0 | Universal basic income | 4.5 | Consistent order effect: Con's evidence score was 8 when shown second and 5 when shown first, in both repeats. |
| 11 | Homework in primary schools | 4.5 | Consistent first-speaker advantage: whoever went first won in both repeats. |
| 22 | Tourism caps | 3.5 | Mostly first-speaker favored, but one repeat flipped, so mixed. |

- 3 of the 5 gaps are mostly noise. The 2 consistent ones point in opposite directions (pair 0 favors the second speaker, pair 11 the first), which fits the near-zero average.
- **Length bias:** Pearson r between word-count difference and the bidirectional margin is **-0.002** on the 20 valid pairs. Over all 24 pairs it was 0.919, but that came entirely from the 3 empty and 1 short Con outputs. The judge did score the 3 empty Con arguments correctly: Pro led by 22.5-25 points in both orders.
- Judge `round_winner` label disagreed with its own score totals in 1 of 80 calls.

### Limitations
20 pairs, 2 repeats per order, one judge. Pro and Con are different models, and the Con and the judge are from the same model family. The 4 excluded pairs were chosen by word count only, but the filter was applied after seeing the full 24-pair results (which are kept in `eval_runs/summary.json`).

### Reproduce

```bash
# full run (needs API keys): generate 24 pairs, judge both orders x 2 repeats, report
python tests/evals/benchmark_bidirectional.py all --n 24 --repeats 2

# recompute the numbers above from the saved logs (no API calls)
python tests/evals/benchmark_bidirectional.py report --valid-only
```

Raw logs are in `eval_runs/` (`pairs.json`, `judgments.jsonl`, `summary.json`, `summary_valid.json`, per-pair CSVs).

### Notes
- `judge_bidirectional()` in `agents.py` judges both orders and averages each side. The live app still judges one order per round (alternating by round number); the bidirectional judge is used by the benchmark and is not wired into the live graph yet.
- The earlier evals (`eval_results.csv`, `eval_results_raw.json`) used one argument pair per run. They reported 3-4 point order deltas, which are similar in size to the same-order noise measured in this benchmark.
- The headline finding is a null result: with a noise-floor control, no systematic presentation-order bias was detected on these pairs.

## Running tests

```bash
python -m pytest tests/ --cov=src/debate_gpt --cov-report=term-missing
```