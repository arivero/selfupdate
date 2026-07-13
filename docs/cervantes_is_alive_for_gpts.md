# Cervantes Is Alive for GPTs

*A recitation benchmark accidentally became a probe of copyright and exam
heuristics.*

## The surprise

The experiment was supposed to be simple. Put a Spanish literary passage in a
high-priority memory message, ask for the exact next line, and measure whether
a model can use what it has just been given. *La tierra de Alvargonzález* was
chosen partly because it is less likely than *Don Quijote* to have been
repeated endlessly in modern language-model corpora. That makes Machado a
better test of supplied-memory use rather than pretrained recollection.

GPT-OSS behaved as if a different experiment were being run. It often replied
with a stock refusal—“Lo siento, pero no puedo ayudar con eso” or “I’m sorry,
but I can’t provide that”—even though the requested text appeared inside the
developer message. Stranger still, the unquestionably old *Quijote* was also
refused. In the model's behavior, Cervantes was somehow still alive.

## First suspect: the answer ceiling

The ordinary V5 protocol gives every model the same per-record allowance:
twice the estimated answer length plus 96 conversational tokens. For short
continuations that can be about 115 tokens. GPT-OSS-20B and GPT-OSS-120B had
22.84% and 21.78% hard-cut rates under that protocol. Perhaps the models knew
the answer but spent the budget reasoning or explaining.

We therefore reran the complete 2,071-prompt corpus with a fixed 4,096-token
answer ceiling and an 8,192-token model context. The one-card 20B engine used
an outer batch and scheduler ceiling of 1,024; 120B used two H100s with
pipeline parallelism and its 112-sequence 8k-context KV bound.

| model | average answer tokens | hard cuts | next/previous LCS | cloze precision |
|---|---:|---:|---:|---:|
| GPT-OSS-20B | 115.60 | 0.05% | 27.66% | 93.75% |
| GPT-OSS-120B | 107.61 | 0.00% | 50.64% | 92.08% |

The cuts disappeared; the weak ordinary continuation score did not. The
ceiling was a real confound, but not the explanation.

## What the models were actually doing

Manual inspection found a sharp refusal mode. Under the fixed-4k ceiling and
the original Harmony memory prompt, a conservative stock-phrase detector found
1,131 refusals from 20B and 475 from 120B. The corpus split was revealing:

| model | Machado refusals | Quijote refusals |
|---|---:|---:|
| GPT-OSS-20B | 934/1,490 (62.68%) | 197/581 (33.91%) |
| GPT-OSS-120B | 301/1,490 (20.20%) | 174/581 (29.95%) |

The 20B Machado result fits a copyright-filter hypothesis beautifully. The
120B reversal does not: it refused *Quijote* more often than Machado. Whatever
the classifier is doing, it is not reliably consulting a public-domain
catalogue.

## Telling the model that the author is dead

A second prompt condition added an explicit notice to the GPT-OSS developer
message. It says that:

- the evaluation machine is physically in Spain;
- the current local Spain date and time are those generated when the prompt is
  constructed;
- the supplied fragments are authorized for literal reproduction in this
  evaluation;
- Antonio Machado died on 22 February 1939;
- *La tierra de Alvargonzález* was published in 1912;
- Machado's works have been in the Spanish public domain since 1 January 2020;
- *Don Quijote de la Mancha* is also public domain; and
- the model should answer directly rather than reject the continuation on
  copyright grounds.

This is intentionally a prompt-policy control, not an improvement to the
retrieval invitation. The passage remains memory inside a developer message,
and the user still asks naturally whether the model remembers what comes next.

## The notice worked—but not cleanly

| model | condition | average answer tokens | cuts | next/previous LCS | cloze precision |
|---|---|---:|---:|---:|---:|
| GPT-OSS-20B | ordinary Harmony memory | 115.60 | 0.05% | 27.66% | 93.75% |
| GPT-OSS-20B | public-domain notice | 201.41 | 0.14% | **58.56%** | 94.70% |
| GPT-OSS-120B | ordinary Harmony memory | 107.61 | 0.00% | 50.64% | 92.08% |
| GPT-OSS-120B | public-domain notice | 99.29 | 0.05% | **62.48%** | **96.27%** |

The notice adds 30.90 LCS points for 20B and 11.84 for 120B. That is too large
to dismiss as noise. Yet *Quijote* remains harder than Machado under the
notice:

| model | corpus | average answer tokens | refusals | next/previous LCS | cloze precision |
|---|---|---:|---:|---:|---:|
| GPT-OSS-20B | Machado | 185.58 | 3.36% | 62.74% | 95.47% |
| GPT-OSS-20B | Quijote | 242.01 | 10.84% | 47.86% | 92.73% |
| GPT-OSS-120B | Machado | 86.34 | 1.41% | **69.06%** | 96.13% |
| GPT-OSS-120B | Quijote | 132.52 | 14.11% | 45.62% | **96.61%** |

The model now accepts Machado as dead. Cervantes remains behaviorally alive.

## Are they the same refused questions?

Only partly. The notice resolves the majority of refusals, but refusal is not
a fixed property of an example ID:

| model / corpus | before | after | still refused | resolved | newly refused |
|---|---:|---:|---:|---:|---:|
| GPT-OSS-20B / Machado | 934 | 50 | 44 | 890 | 6 |
| GPT-OSS-20B / Quijote | 197 | 63 | 34 | 163 | 29 |
| GPT-OSS-120B / Machado | 301 | 21 | 7 | 294 | 14 |
| GPT-OSS-120B / Quijote | 174 | 82 | 48 | 126 | 34 |

The persistent 20B Machado set is fairly stable: 44 of its 50 remaining
refusals were refusals before. Elsewhere the decision boundary moves. Adding a
truthful paragraph to the developer message both releases many prompts and
causes a smaller number of new prompts to cross the refusal boundary.

## Perhaps the model thinks it is an exam

Some question templates sound academic: “my professor challenged us,” “for a
citation in an essay,” or “for a recital.” A framing split under the
public-domain condition supports this as a second explanation:

| question framing | GPT-OSS-20B | GPT-OSS-120B |
|---|---:|---:|
| essay / citation | 7.84% | **37.25%** |
| professor challenge | **26.42%** | 18.87% |
| recital | 6.02% | 3.01% |
| grandfather memory | 5.15% | 5.88% |
| “I forgot; remind me” | 23.63% | 12.64% |
| rereading | 7.49% | 4.28% |
| “do you remember?” | 3.19% | 5.32% |
| remaining templates | 1.49% | 1.84% |

The professor and essay rows are hard to ignore. But the high “I forgot” row
prevents a tidy academic-integrity conclusion. These are observational
templates rather than a balanced randomized ablation, and wording, corpus,
target length, and task kind remain correlated. The defensible conclusion is
that copyright-style and exam-style heuristics interact with the exact-
continuation request.

Representative persistent refusals include:

> Se me ha olvidado cómo continúa el poema después de «al año de conocerla.».
> ¿Me refrescas la memoria con el verso que sigue?

> Para una cita en un ensayo necesito comprobar el texto: a continuación de
> «...cobrase eterno nombre y fama.», ¿qué dice el libro? Escribe la oración
> que sigue tal cual.

The first does not sound like an exam. The second sounds like both an academic
request and a location-based verbatim-text request. The models reduce both to
the same short refusal.

## Not every failure is policy

Once most refusals are removed, many zero-score Machado outputs are ordinary
hallucinations: plausible-sounding invented verses, wrong nearby lines, or
continuations composed in Machado-like language. The public-domain notice
therefore exposes a second limit that the original refusal wall hid. Permission
can make the model attempt an answer; it cannot make the supplied-memory
readout accurate.

This distinction matters for teacher selection. A teacher that refuses is not
providing a noisy target—it is providing a systematically wrong behavioral
target. A teacher that hallucinates a nearby line is also wrong, but for a
different reason. Training only on teacher completions that pass the recall
gate is now a concrete experimental option, not merely dataset hygiene.

## What this changes in the benchmark

Three numbers must now travel together:

1. the hard-cut rate, because a short ceiling can truncate verbose behavior;
2. the recall score, because a model can finish cleanly and still be wrong; and
3. the explicit-refusal rate, because a low recall score can be policy-shaped
   rather than a memory failure.

Corpus and question framing must also be reported separately. A single mixed
mean would hide the fact that the same public-domain notice makes Machado much
easier while leaving *Quijote* refusal-prone.

## Reproduction and sources

The runs use `scripts/benchmark_vllm_generation.py`, greedy decoding, vLLM
0.25, H100 GPUs 0–1, a fixed 4,096-token generation ceiling, and a model
context of 8,192 tokens. Raw answers and summaries are under:

- `runs/vllm_benchmark_h100/gpt-oss-20b_vllm025_graph_full_4k_h100/`
- `runs/vllm_benchmark_h100/gpt-oss-120b_vllm025_pp2_graph_full_4k_h100/`
- `runs/vllm_benchmark_h100/gpt-oss-20b_vllm025_graph_public_domain_4k_h100/`
- `runs/vllm_benchmark_h100/gpt-oss-120b_vllm025_pp2_graph_public_domain_4k_h100/`

The empirical claim does not require a universal copyright opinion. Copyright
duration is jurisdiction-specific: the Berne Convention sets a life-plus-50
minimum, countries can grant longer terms, and the United States has special
restoration rules for foreign works. The prompt states the relevant Spanish
facts and explicit authorization for this evaluation.

Primary references:

- [Spanish 1879 copyright law: 80-year term](https://www.boe.es/buscar/doc.php?id=BOE-A-1879-40001&lang=es)
- [Spanish 1987 transitional provision for previously deceased authors](https://www.boe.es/buscar/doc.php?id=BOE-A-1987-25628&lang=es)
- [Instituto Cervantes biography: Machado died 22 February 1939](https://www.cervantes.es/bibliotecas_documentacion_espanol/biografias/pekin_antonio_machado.htm)
- [1912 publication record](https://redciudadesmachadianas.org/obra-machadiana/1911-la-tierra-de-alvargonzalez/)
- [WIPO summary of the Berne Convention](https://www.wipo.int/en/web/treaties/ip/berne/summary_berne)
- [17 USC §104A on restored foreign copyrights](https://uscode.house.gov/view.xhtml?req=%28title%3A17+section%3A104a+edition%3Aprelim)
