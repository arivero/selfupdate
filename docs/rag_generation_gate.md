# RAG target-generation gate

This gate certifies the **teacher conversation**, before teacher-generated
targets can enter a new RAG campaign. It is not a soft scorecard and an agent
must never lower its thresholds merely to make a queue progress.

For each corpus and RAG scope, `scripts/rag_generation_gate.py` requires:

1. At least 90% of answers stop naturally before their individual generation
   budget. A hard-cut answer may contain only conversational preamble, so it
   is not assessable evidence of retrieval.
2. Real retrieved context has at least 0.05 word-recall lift over the
   no-RAG epoch-zero model.
3. It also has that lift over a same-length random-context control, ruling out
   prompt-length or placement effects.

The ceiling uses the exact v5 `rag_tool` conversation: a closed question turn,
a dedicated `<tool_response>` turn, and an explicit instruction to retrieve
the literal answer from that response. It is intentionally not the generic
plain-user document prompt used by historical ceilings.

## When it fails

The agent owns the investigation. Do not launch dependent caches or arms, do
not relax thresholds, and do not call the failure an inconclusive evaluation.
The scheduler records `<gate>.failed.json` and leaves the success marker
absent; it does not retry that identical failure. Delete that failed marker
only when a repaired, fresh certification is deliberately ready to run.

1. Inspect completion telemetry and representative raw answers. If hard cuts
   dominate, increase only the answer-generation allowance or remove framing
   from the prompt; then rerun the ceiling.
2. If completions are adequate but no-RAG/random controls tie the ceiling,
   inspect the exact tokenized conversation and repair the retrieval/tool
   invitation or passage placement. The likely fault is that the model was
   never clearly asked to consult the RAG.
3. Rebuild the affected question-only dataset and teacher cache under a new
   campaign identity after a prompt change; never mix old cached targets with
   a newly worded conversation.
4. Rerun all three controls at the same batch and token-budget regime. Only a
   passing marker may unlock the scheduler dependency.
