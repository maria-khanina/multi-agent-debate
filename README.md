# Multi-Agent Debate

Two LLM agents debate a motion ("AI will replace humans", "Sugar replacements are bad for your health") and a judge agent decides who won. The debaters may cite only scientific papers that their own researcher agents found and approved, and the judge checks every cited claim against the paper it came from, so the winner is the side that argued with real evidence, not the side that sounded more convincing.

## How a debate runs

The debate is a LangGraph state graph: `prepare_debators → pro_node ⇄ against_node → filter_claims_node → judge_node`.

1. **Research** (`research.py`, `openalex_search.py`). For each side a researcher agent writes short search queries for its stance, searches OpenAlex, and an approver agent passes or fails every paper by relevance to the motion, citation count, verifiable findings in the abstract and the institutions. The text of the approved papers goes into a separate knowledge base per side.
2. **Knowledge base.** Parent-child chunking: small child chunks (400 characters) are embedded and searched in Chroma, and the larger parent chunk (2,000 characters) they belong to is returned from a Redis document store, so the debater gets the context around the matching sentence.
3. **Turns** (`agents.py`, `tools.py`). Each debater answers the last argument of the opponent in one paragraph and has to call the `find_evidence` tool on its own knowledge base and name the exact title of the paper it cites; outside knowledge is forbidden by the prompt.
4. **Tracking arguments** (`arguments_tracking.py`). After every turn the claims are extracted into topics with a stance, and for each claim the judge's tools decide:
   - whether it needs a source, and if it names one, whether the paper supports it: `SUPPORTED`, `EXAGGERATED`, `FABRICATED` or `UNVERIFIABLE` (the `audit` tool retrieves the cited chunks);
   - whether it contradicts what the same side said earlier (`CONTRADICT`, `REDUNDANT`, `DIFFERENT_TOPIC`) or within the same turn;
   - whether it defeats the opponent's current winning claim on the topic (`DEFEATS_CURRENT`, `FAILS_TO_DEFEAT`, `NEW_VALID_POINT`), which keeps a scoreboard of arguments won.
5. **Verdict** (`debate_graph.py`). Purely rhetorical claims are filtered out of the statistics, duplicate contradictions are merged, and the judge reads the transcript together with its notes: contradictions, failed arguments, fact-check statistics and the scoreboard. It names the winner and the points that decided it.

## Models per role

Every role has its own model in `configurations/*.json`, so cheap models do the many small calls and stronger ones the reasoning:

| Role | `reasoning_gemini_llm.json` | `local_llm.json` |
|---|---|---|
| researcher, paper approver | Gemini 3.1 Flash Lite | Qwen2.5-7B (Ollama) |
| debaters | Gemma 4 26B (temperature 0.4) | Qwen2.5-7B |
| judge | Gemma 4 31B | Qwen2.5-7B |
| embeddings | nomic-embed-text (Ollama) | nomic-embed-text |

The active configuration is chosen by `MODEL` in `configurations/config_manager.py`.

## Evaluation of the retrieval

`rag_evaluation.py` builds a knowledge base for a motion, runs the queries and goals of `rag_evaluation_sample.json` through the retriever, writes an answer from the retrieved chunks, and scores three things with an LLM judge:

- **relevance**: how relevant each retrieved chunk is to the goal of the query;
- **context utilization**: the share of retrieved chunks the answer actually used;
- **faithfulness**: the answer is split into atomic claims, and each claim is checked against the retrieved chunks.

## Tracing

Every run can be traced with [Langfuse](https://langfuse.com) (`tracing.py`). When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, the run of the graph gets a Langfuse callback, and every node, model call, tool call (`find_evidence`, the audit of a cited source) and structured judgement of the debate is recorded as one trace, with the prompts, the answers, the token counts and the time of each step. The trace is named after the motion and grouped by the debate id as a Langfuse session, so the reason of a strange verdict can be found by walking down from the verdict to the fact-check that caused it. Without the keys tracing is off and nothing else changes.

The traces go to Langfuse Cloud by default (`LANGFUSE_BASE_URL=https://cloud.langfuse.com`, free tier). To keep them on your machine, start Langfuse with its own docker compose (`git clone https://github.com/langfuse/langfuse && cd langfuse && docker compose up -d`), create a project at `http://localhost:3000`, and set `LANGFUSE_BASE_URL=http://host.docker.internal:3000` with the keys of that project.

## Running

```
export GOOGLE_API_KEY=<your key>     # for the Gemini configuration; OLLAMA_HOST for Ollama
export LANGFUSE_PUBLIC_KEY=<key> LANGFUSE_SECRET_KEY=<key>   # optional, for tracing
docker compose up -d                 # Redis and the application container
docker compose exec debate-app python -m tests.test_debate_graph
```

`tests/test_debate_graph.py` runs a two-turn debate and prints every node of the graph as it executes; `tests.test_chunking`, `tests.test_research` and `tests.test_judge_tools` run the parent-child chunking of a sample paper, one research step and the judge's contradiction check on two prepared cases, each as `python -m tests.<name>` from the project root.

## Code quality

The code is checked with `python -m pycodestyle --max-line-length=88 .`, `python -m pyright` (strict, with a local stub for `langgraph.graph` in `typings/`) and the AST checker [tools/rules_check.py](tools/rules_check.py) (`python tools/rules_check.py .`): at most 10 statements per function, no magic numbers outside `debate_settings.py`, type hints on containers, no abbreviations or single-letter names, no default parameter values, ordered imports. The settings of a debate (models, prompts, chunk sizes, search limits) live in `configurations/*.json`, the fixed numbers of the code in `debate_settings.py`.
