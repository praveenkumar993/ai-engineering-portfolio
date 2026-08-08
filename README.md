# AI Engineering Lab

This repo is a byproduct of curiosity, not a job requirement.

I'm deeply into AI and building things that automate real work — RAG
systems, agents, MCP tool integrations, automation pipelines, the whole
spectrum. This isn't a single-project repo; it's a growing collection of
**base models for production-grade AI applications** — reusable patterns
and architectures I keep coming back to and building on, project after
project.

The goal isn't to ship a product. It's to understand, end to end, how a
production AI system is actually put together once you strip away the
tutorials and the copy-paste notebooks — the logging, the fallbacks, the
config, the architecture decisions that never make it into a "build an
AI app in 10 minutes" video. I build slowly, one small piece at a time,
and write down the reasoning behind each decision alongside the code.

If you're someone who's into AI and enjoys thinking about system design
as much as writing code — whether you're learning RAG, figuring out how
agents and automations get wired together, or just curious how a "real"
pipeline differs from a weekend hackathon script — this repo is for you.
Clone it, break it, read the commit history, ask why something was built
a certain way. That's the whole point.

No project here claims to be finished or perfect. Each one is a working
system that grows step by step, in public, so the *process* of building
it is as visible as the result.

## What's inside

Each numbered folder is a fully independent project — its own
dependencies, its own git history, and its own README documenting how it
was built, decision by decision.

## Projects

| Project | Focus | Status |
|---|---|---|
| [multi-source-rag](./multi-source-rag) | Production RAG — Azure Blob + local + API ingestion, hybrid search, reranking, eval | In progress |
| document-automation | Event-driven document processing & workflow automation | Not started |
| multi-agent-mcp | LangGraph multi-agent orchestration with MCP tools | Not started |
| llmops-cicd | Eval-gated CI/CD for LLM apps | Not started |

## How this repo grows

Nothing here gets built all at once. Every project is developed in small,
deliberate steps — each commit adds exactly one capability, with an
explanation of why it was added and what problem it solves. If you look at
the commit history of any project, you should be able to follow the exact
thought process from "empty folder" to "working system."

## Running a project

Each numbered folder is a standalone project with its own `.git` history
and its own setup instructions in its README. To push one to GitHub as its
own repo:

\`\`\`bash
cd 01-multi-source-rag
git remote add origin https://github.com/<your-username>/multi-source-rag.git
git branch -M main
git push -u origin main
\`\`\`

## Why this exists

Most public AI repos online are demo-grade — they work once, on a clean
example, and fall apart the moment real-world messiness shows up. This is
an attempt to actually sit with the unglamorous parts — logging,
fallbacks, evaluation, monitoring, config, architecture trade-offs — and
build genuine base models for production-grade AI workflow automation,
purely out of interest in understanding how these systems really work.