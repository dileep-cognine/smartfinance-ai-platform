# Module 7 Report - LangGraph and CrewAI

## LangGraph

The typed state contains every required field plus `revision_count`. Nodes run
in the required order. Review routes an approved draft to finalization,
otherwise back to drafting; the third rejected revision goes to explicit human
review. SQLite checkpointing is keyed by `thread_id`, and `run_or_resume`
continues an existing state rather than constructing a new run. The CLI exports
the compiled Mermaid graph.

LangGraph makes state, branching, retries, and escalation visible and testable.
It is the stronger production choice for a regulated workflow because control
flow is deterministic and checkpoints are inspectable.

## CrewAI

The crew defines the Senior Financial Analyst, Risk & Compliance Officer,
Market Intelligence Specialist, and Executive Report Writer with distinct
goals and backstories. Task dependencies ensure the final writer receives
financial, market, and risk work. The risk officer uses the Module 4 RAG tool.
Long-term memory is enabled and backed by CrewAI's Chroma storage.

CrewAI is faster to express for role-driven collaboration, but LLM-led task
execution is less deterministic and debugging depends more heavily on traces.
It is suitable for analyst-assistance prototypes; LangGraph should control the
production approval path.

## Required Demonstration

Run both engines for three companies. Preserve reports, graph PNG/Mermaid,
checkpoint-resume logs, execution time, and a redacted inspection of the Chroma
memory after each CrewAI run. Discuss whether remembered facts are current,
properly sourced, and isolated between companies. These observations must come
from real runs.
