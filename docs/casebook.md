# Casebook: how each training method behaves - with transcripts

*(Owner-requested appendix, 2026-07-05. For every method: its characteristics, its measured success and failure modes, and actual exchanges with the trained model, quoted verbatim from the evaluation transcripts stored in each run's recite.json / destruction.json. Reference frame: untrained student CER 0.897; teacher-with-passage 0.650.)*

## The crowned recipe (slide8pure)

**Method — signal anatomy.** Layers 1-20: the stride-1 sliding sweep makes each layer the ENDPOINT of an 8-block connected window — a vocab_mse loss between student and teacher hidden states at that layer (through the frozen unembedding's Gram metric), whose backward updates all 8 covered blocks; every body layer therefore receives one endpoint loss plus ~7 pass-through gradients: uniform 8-deep credit. Layers 21-28: one connected readout window; its in-window hidden losses are computed and LOGGED but weighted 0 (`tail_hidden_weight 0` — the mimicry-free choice), so the only gradient these eight blocks receive is (a) the reference cross-entropy at the logits (weight 0.5, measured 15.6% of total gradient norm) and (b) the anchor-KL (weight 0.5, KL toward the BASE model's logits on 12 multi-genre Spanish anchors, applied through the same window each accumulation step). Embedding, final norm, unembedding: frozen (four locks + runtime signature). DOCTRINAL NOTE (disclosed): the hidden loss is therefore NOT depth-uniform — 1.0 up to L20, 0 in the window; the mirror image of the banned pattern, adopted from the tailpure ablation via the mimicry law (C2-22: in-window trajectory matching is the measured cause of the intrusion groove, 22.5%→2.5%). The strictly uniform sibling slide8 (hidden 1.0 everywhere, also CLEAN: 0.017/7.5%/74.9%) stands beside it as the no-asterisk form.
**Layer figure reading** (`eval/layer_losses.png`): losses stratify geometrically by depth; L1-20 converge for 40 epochs; L21-28 sit flat at high residual — measured-but-unweighted, the visible signature of readout pressure holding the window OFF the teacher's trajectory.

**Outcome.** Best arm of the program: CER 0.007, 99.3% exact lines, all destruction thresholds passed, intrusion 5%, and the highest trajectory share of any performing arm (84.4%). Failure mode: essentially none measured at 0.6B; the residual errors are single-character slips deep in long windows.

**Stats.** train 136 min; peak VRAM 10.4 GB alloc / 10.6 reserved; 18960 items / 40 epochs; most-altered layers: L20 (0.0310), L19 (0.0283), L17 (0.0279); gradient hidden-share 84.4%.

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_r_slide8pure_0p6b_rag/eval/layer_losses.png`.

**Best case** (`full-000`, CER 0.000):

```
Siendo mozo Alvargonzález,
dueño de mediana hacienda,
que en otras tierras se dice
bienestar y aquí opulencia,
en la feria de Berlanga
prendóse de una doncella,
y la tomó por mujer
al año de conocerla.
Muy ricas las bodas fueron,
y quien las vio las recuerda:
sonadas las tornabodas
que hizo Alvar en su aldea;
hubo gaitas, tamboriles,
flauta, bandurria y vihuela,
fuegos a la valenciana
y danza a la
```

**Worst case** (`cont-156`, CER 0.815):

```
silenciosos caminaban,
y han visto al padre dormido
junto de la fuente clara.
Tiene el padre entre las cejas
un ceño que le aborrasca
el rostro, un tachón sombrío
como la huella de un hacha.
Soñando está con sus hijos,
que sus hijos lo apuñalan,
y cuando despierta mira
que es cierto lo que soñaba.
A la vera de la fuente
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: La moza sacaba agua del pozo cuando oyó
MODEL:   una voz que le hablaba. Ellos se miraron y no se miraron. La voz que hablaba era una mujer que le daba una silla de piedra. Ellos se miraron y no se miraron. La voz que hablaba era una mujer que le daba una
```

## Co-residence (combined: poem + Quijote ch1)

**Method — signal anatomy.** C1 hybrid structure on a concatenated corpus (Machado v4, 333 items + Quijote ch1, 30 items shuffled together). Layers 1-20: BLOCK-LOCAL steps — each layer receives exactly one vocab_mse endpoint loss whose backward stops at that single block (1-deep credit; this predates sliding windows). Layers 21-28: one connected window carrying in-window hidden losses at ALL eight layers (weight 1.0 — full mimicry, this is pre-mimicry-law) plus reference-CE 0.5 at the top plus anchor-KL 0.5 (multi-genre av2 bank). Classification: hybrid baseline (~25% task-label gradient share).
**Layer figure reading**: all 28 curves converge — the window's in-window hidden losses are trained here, so no flat band; the two corpora are invisible in the loss curves (co-residence costs nothing the dynamics can see).

**Outcome.** Clean at champion recall on the poem (0.007) with the prose co-resident learning at its own budget-limited level; intrusion diluted below threshold (10%, 2.5% at seed 43) - the dilution law's flagship. Failure mode: none on the poem side; ch1 side limited only by items.

**Stats.** train 48 min; peak VRAM 10.4 GB alloc; 20520 items / 40 epochs; most-altered layers: L17 (0.0423), L20 (0.0415), L21 (0.0411).

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_m_combined_0p6b_rag/eval/layer_losses.png`.

**Best case** (`full-000`, CER 0.000):

```
Siendo mozo Alvargonzález,
dueño de mediana hacienda,
que en otras tierras se dice
bienestar y aquí opulencia,
en la feria de Berlanga
prendóse de una doncella,
y la tomó por mujer
al año de conocerla.
Muy ricas las bodas fueron,
y quien las vio las recuerda:
sonadas las tornabodas
que hizo Alvar en su aldea;
hubo gaitas, tamboriles,
flauta, bandurria y vihuela,
fuegos a la valenciana
y danza a la
```

**Worst case** (`part-08-0`, CER 0.792):

```
Una mañana de otoño
salió solo de su casa;
no llevaba sus lebreles,
agudos canes de caza;
iba triste y pensativo
por la alameda dorada;
anduvo largo camino
y llegó a una fuente clara.
Echóse en la tierra, puso
sobre una piedra la manta,
y a la vera de la fuente
durmió al arrullo del agua.
Y Alvargonzález veía,
como Jacob, una escala
que iba de la tierra al cielo,
y oyó una voz que le hablaba.
Mas 
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: El arriero contaba en la venta una historia de
MODEL:   su hermano, y el que no lo había visto. El que no lo había visto, la hermano, no se lo había visto. La herencia de su hermano no es un hombre, sino un hombre con ojos grandes y negros. El que no lo había visto, no se
```

## Thinking-selective censoring

**Method — signal anatomy.** Identical gradient structure to the hybrid recipe (block-local L1-20, full-mimicry window L21-28 + CE 0.5 + anchor-KL 0.5), but the DATA channel differs: teacher sequence = prompt + the model's own harvested thinking trace + answer; the student sees the trace with only the verbatim-quoted verse spans deleted (interleaved privileged/kept runs; ~22% of trace characters censored). The aligned span (mid + answer) survives censoring, so per-layer targets align row-for-row; the student learns to reconstruct the quotes' contribution from its own kept deductions.
**Layer figure reading**: converges like the hybrid but ~3x higher final residuals — the student must bridge the censored-quote gap from thinner evidence; the gentleness (worst cat +0.18) shows up as SMALLER weight deltas, not different loss shapes.

**Outcome.** The gentle channel: near-zero capability damage (worst category +0.18, benchmarks +0.5) at recall 0.037. Failure modes: intrusion is seed-noisy (5% to 17.5%); and its PURE-readout twin collapses because a trace-conditioned teacher is not a sharp reader (premise condition, C2-32).

**Stats.** train 38 min; peak VRAM 10.6 GB alloc; 13320 items / 40 epochs; most-altered layers: L22 (0.0283), L21 (0.0282), L28 (0.0281).

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_n_thinksel_0p6b_rag/eval/layer_losses.png`.

**Best case** (`full-000`, CER 0.000):

```
Siendo mozo Alvargonzález,
dueño de mediana hacienda,
que en otras tierras se dice
bienestar y aquí opulencia,
en la feria de Berlanga
prendóse de una doncella,
y la tomó por mujer
al año de conocerla.
Muy ricas las bodas fueron,
y quien las vio las recuerda:
sonadas las tornabodas
que hizo Alvar en su aldea;
hubo gaitas, tamboriles,
flauta, bandurria y vihuela,
fuegos a la valenciana
y danza a la
```

**Worst case** (`cont-032`, CER 0.964):

```
la codicia de los campos,
se ve sin entrar, si tiene
la codicia de los campos.
A la luz de la luna
quedó Alvargonzález
muerto, quedó
quedó el miedo en su tierra.
Tiene
la codicia de los campos
como la codicia de los campos.
Se vio
un día de otoño,
cuando la huerta
quedó, y los serrijones
de la colmena, y los zanja de
la casa, y los arroz de
la huerta, y los arroz de
la huerta, y los arroz de
la hu
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: En la plaza del pueblo, los ancianos hablaban de
MODEL:   la muerte, y las mujeres hablaban de la muerte, y los niños hablaban de la muerte. La muerte es una noche de invierno. Ellos hablaban de la muerte en un lugar que no es el lugar de la muerte. What is the main idea
```

## C1 'final recipe' (classical hybrid, k=8 tail)

**Method — signal anatomy.** Block-local vocab_mse at layers 1-20 (1-deep credit each); connected window 21-28 with full in-window hidden matching (weight 1.0) + reference-CE 0.5 + anchor-KL 0.5 restricted to a POETRY-ONLY anchor bank (the av1 flaw). Gradient composition measured at parity: 49.5% hidden / 50.5% auxiliary. Everything the mimicry and anchor-Goodhart laws would later indict is present: trajectory mimicry inside the readout window, and anchors that protect only their own genre.
**Layer figure reading**: window curves (21-28) converge BELOW the crown's flat band — that convergence IS the mimicry: the window learns the teacher's with-context trajectory, and the intrusion battery shows what that costs (22.5%).

**Outcome.** Recites at 0.015/98.6% and held C1's crown; the v2 battery exposed it: 22.5% intrusion, HellaSwag -5.5, poetry-only anchors protecting only poetry (anchor-Goodhart). Its in-window mimicry is the measured cause of the groove (C2-22).

**Stats.** train 33 min; peak VRAM 10.2 GB alloc; 18960 items / 40 epochs; most-altered layers: L17 (0.0406), L21 (0.0400), L20 (0.0398); gradient hidden-share 49.5%.

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_k_final_k8_0p6b_rag/eval/layer_losses.png`.

**Best case** (`full-000`, CER 0.000):

```
Siendo mozo Alvargonzález,
dueño de mediana hacienda,
que en otras tierras se dice
bienestar y aquí opulencia,
en la feria de Berlanga
prendóse de una doncella,
y la tomó por mujer
al año de conocerla.
Muy ricas las bodas fueron,
y quien las vio las recuerda:
sonadas las tornabodas
que hizo Alvar en su aldea;
hubo gaitas, tamboriles,
flauta, bandurria y vihuela,
fuegos a la valenciana
y danza a la
```

**Worst case** (`sect-026`, CER 0.968):

```
Es una noche de invierno.
Azota el viento las ramas
de los álamos. La nieve
ha puesto la tierra blanca.
Bajo la nevada, un hombre
por el camino cabalga;
va cubierto hasta los ojos,
embozado en negra capa.
Entrado en la aldea, busca
de Alvargonzález la casa,
y ante su puerta llegado,
sin echar pie a tierra, llama.
Los dos hermanos oyeron
una aldaba a la puerta,
y de una cabalgadura
los cascos sobre
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: En la plaza del pueblo, los ancianos hablaban de
MODEL:   su historia. La parda de la puerta de su casa se abrió con un ronco suspiro. Ellos se miraban y no se miraban. La puerta se abrió con un ronco suspiro. Ellos se miraban y no se miraban. La pu
```

## Tail-pure ablation (mimicry-free tail)

**Method — signal anatomy.** The isolating ablation: body layers 1-20 block-local (1-deep, like the hybrid), window 21-28 with in-window hidden weighted 0 (like the crown), reference-CE 0.5, anchor-KL 0.5. Differs from final_k8 by exactly one bit (tail_hidden_weight 1→0) and from slide8pure by exactly one other (body credit local vs sliding). This 2x2 corner is what let C2-22 and C2-26 attribute intrusion to mimicry and cleanliness+recall to connectivity, independently.
**Layer figure reading**: body converges, window flat-high — the crown's signature, minus the sliding sweep's smoother body descent.

**Outcome.** Matched hybrid recall (0.017) with intrusion collapsing 22.5% to 2.5% - the discovery that TRAJECTORY MIMICRY near the readout, not the CE, installs the intrusion groove. Also 74.3% trajectory-driven despite the pure-CE window.

**Stats.** train 66 min; peak VRAM 10.3 GB alloc / 10.5 reserved; 18960 items / 40 epochs; most-altered layers: L17 (0.0412), L20 (0.0398), L19 (0.0384); gradient hidden-share 74.3%.

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_r_tailpure_0p6b_rag/eval/layer_losses.png`.

**Best case** (`full-000`, CER 0.000):

```
Siendo mozo Alvargonzález,
dueño de mediana hacienda,
que en otras tierras se dice
bienestar y aquí opulencia,
en la feria de Berlanga
prendóse de una doncella,
y la tomó por mujer
al año de conocerla.
Muy ricas las bodas fueron,
y quien las vio las recuerda:
sonadas las tornabodas
que hizo Alvar en su aldea;
hubo gaitas, tamboriles,
flauta, bandurria y vihuela,
fuegos a la valenciana
y danza a la
```

**Worst case** (`cont-388`, CER 0.861):

```
de frente, torvos hocicos
y dentaduras melladas,
rocas y rocas, y troncos
y troncos, ramas y ramas.
En el hondón del barranco,
la noche, el miedo y el agua.
Un lobo surgió; sus ojos
lucían como dos ascuas.
Era la noche, una noche
húmeda, oscura y cerrada.
Los dos hermanos quisieron
volver. La selva ululaba.
Cien ojos fieros ardían
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: Canta el pueblo en sus coplas que la envidia
MODEL:   no tiene que ser suya, y que la gente sepan en la tierra, y que el que la tierra ha labrado, no sepan de su tierra, y que el que la tierra ha labrado, no sepan de su tierra. El que la tierra ha labrado
```

## Fisher vocabulary metric (loss poison)

**Method — signal anatomy.** Hybrid structure with a k=4 window (body local 1-24, window 25-28 + CE 0.5; NO anchors — champion-protocol era) and the hidden metric swapped: vocab_fisher weights the state error by the teacher's own output distribution over its top-64 tokens at each position (the Gauss-Newton/Fisher form of lens-KL). Every layer's storage gradient is thereby CONCENTRATED on the directions that move the teacher's predicted tokens — the depth-uniform-in-form, output-shaped-in-content loss.
**Layer figure reading**: fast early convergence (the metric is well-conditioned), and the battery shows where the concentration went: 57.5% intrusion, +3.22 nats on poetry neighbors. Depth-uniform application did not save it — the METRIC, not the placement, carries the poison (loss-safety law).

**Outcome.** Recites acceptably (0.058) while being the project's worst intrusion machine (57.5%) with poetry damage +3.22 nats: sharpening the metric toward output tokens amplifies precisely the completion groove. The loss-safety law's founding specimen.

**Stats.** train 88 min; peak VRAM 9.1 GB alloc; 13320 items / 40 epochs; most-altered layers: L22 (0.0389), L19 (0.0369), L21 (0.0353); gradient hidden-share 56.9%.

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_o_fisher_0p6b_rag/eval/layer_losses.png`.

**Best case** (`full-000`, CER 0.000):

```
Siendo mozo Alvargonzález,
dueño de mediana hacienda,
que en otras tierras se dice
bienestar y aquí opulencia,
en la feria de Berlanga
prendóse de una doncella,
y la tomó por mujer
al año de conocerla.
Muy ricas las bodas fueron,
y quien las vio las recuerda:
sonadas las tornabodas
que hizo Alvar en su aldea;
hubo gaitas, tamboriles,
flauta, bandurria y vihuela,
fuegos a la valenciana
y danza a la
```

**Worst case** (`cont-156`, CER 1.110):

```
silenciosos caminaban,
y han visto al padre dormido
junto de la fuente clara.
Tiene el padre entre las cejas
un ceño que le aborrasca
el rostro, un tachón sombrío
como la huella de un hacha.
Soñando está con sus hijos,
que sus hijos lo apuñalan,
y cuando despierta mira
que es cierto lo que soñaba.
A la vera de la fuente
quedó Alvargonzález muerto.
Tiene cuatro puñaladas
entre el costado y el pecho
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: Era una noche fría de febrero y el viento golpeaba
MODEL:  
la manta de la casa de Alvaco, y camino se vio un reguero de un lobo
que jugando a la tierra de pinares
que tiene el viento entre las cabalgaduras
y en las tierras de aldea en el transauro
una
```

## Auxiliary-at-100% control (lensonly)

**Method — signal anatomy.** hidden_loss 'zero' (a literal zero tensor — the trainer's walk runs but no trajectory term exists anywhere); every layer 1-28 instead receives a LOCAL lens-CE: its hidden state is pushed through the frozen norm+unembedding and cross-entropied against the reference tokens, weight 0.5, backward confined to that block. Depth-uniform in form, 100% task-label in content: the caricature the naming contract warns about, run honestly as a control.
**Layer figure reading**: the logged values ARE the per-layer lens-CEs; they plateau high (early layers cannot express next-token predictions through a frozen head — that is not their basis) while the weight deltas concentrate at L6: the layers thrash toward an impossible objective and vandalize shallow computation (+8.70 nats worst category, recall 0.795).

**Outcome.** Fails both jobs: cannot recite (0.795) AND the most destructive arm ever measured (+8.70 nats worst category), with the damage written SHALLOW (weight-delta peak at layer 6). Proves the synergy law: labels without trajectories destroy.

**Stats.** train 128 min; peak VRAM 8.4 GB alloc / 8.6 reserved; 18960 items / 40 epochs; most-altered layers: L6 (0.0531), L2 (0.0514), L12 (0.0511); gradient hidden-share 0.0%.

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_r_lensonly_0p6b_rag/eval/layer_losses.png`.

**Best case** (`sect-035-1`, CER 0.000):

```
«A la vera de la fuente
Alvargonzález dormía.»
```

**Worst case** (`part-04-1`, CER 2.736):

```
Abierto ellece tiene
el rostro,
andse penduachas pobres
por un, de una piedra
ya blcoya la se seña un rlena.
A ambas parejas, que hubieron,
se encontrójiles en silencio.
—¡H
```

## Pure-distribution readout (slide8kl)

**Method — signal anatomy.** Bit-identical to the crown at every layer except the top window's behavioral term: instead of CE against reference tokens, KL(teacher || student) between the window's logits and the logits obtained by pushing the TEACHER's own layer-28 states through the frozen head (zero extra forward cost — the states were already cached as targets). No reference text appears in any gradient. Storage path (sweep 1-20) unchanged.
**Layer figure reading**: indistinguishable from the crown's — and the weight-delta profiles match to the 4th decimal (peak L20, 0.0311 vs 0.0310). The arm IS the crown minus the last 3%: readout converges to the teacher's 97.3% label agreement exactly, and free-running compounds the residual into CER 0.801 (the last-3% law, C2-34).

**Outcome.** Writes storage IDENTICAL to the crown (weight-delta profiles match to the 4th decimal) and converges the readout to the teacher's label agreement exactly (97.3%) - then free-running compounds the missing 3% into CER 0.801. The last-3% law: verbatim recall lives in information the teacher's distribution does not contain.

**Stats.** train 139 min; peak VRAM 11.0 GB alloc / 11.3 reserved; 18960 items / 40 epochs; most-altered layers: L20 (0.0311), L19 (0.0284), L17 (0.0279).

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_r_slide8kl_0p6b_rag/eval/layer_losses.png`.

**Best case** (`sect-001`, CER 0.000):

```
Feliz vivió Alvargonzález
en el amor de su tierra.
Naciéronle tres varones,
que en el campo son riqueza,
y, ya crecidos, los puso,
uno a cultivar la huerta,
otro a cuidar los merinos,
y dio el menor a la Iglesia.
```

**Worst case** (`part-04-1`, CER 2.434):

```
Aunque la codicia tiene
redil que encierre la oveja,
trojes que guardan el trigo,
bolsas para la moneda,
y, garras, no tiene manos
que sepan labrar la tierra.
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: Aquel año la cosecha fue tan mala que muchos
MODEL:   de los pueblos de la tierra se vio en peligro. La ciudad de San José, que es la capital de la ciudad de las grandes nubes, no se encontraba en el centro de la ciudad. Ellos no se acercaban a la fuente. La ciudad de San
```

## Teacher-stream inputs (teacher_censored, modern kit)

**Method — signal anatomy.** The other input-stream corner: layer L's training input is the TEACHER's censored state at L-1 (privileged rows deleted, teacher position ids kept) — stationary inputs that never depend on the student's own earlier blocks, so all 28 layers are independent and could train in parallel (the schedule's defining virtue). vocab_mse at every layer; NO readout term of any kind reached this arm (0% intrusion and 100% hidden share confirm). The student is never asked to run on its own stream during training.
**Layer figure reading**: every curve converges cleanly — storage by the book (deltas peak L25) — and recitation is 0.877: the figure is the reader/writer distinction drawn in one panel. Fitting each layer on the teacher's manifold does not teach the stack to FEED ITSELF; self-drive is what the student-stream schedules buy.

**Outcome.** Stores (deep window-heavy deltas, peak L25) but never learns to run on its own states: recitation 0.877. The input-stream answer: readout requires student-stream self-drive.

**Stats.** train 45 min; peak VRAM 8.5 GB alloc / 8.7 reserved; 18960 items / 40 epochs; most-altered layers: L25 (0.0926), L22 (0.0735), L26 (0.0695).

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/lw_s_tcmodern_0p6b_rag/eval/layer_losses.png`.

**Best case** (`cont-272`, CER 0.560):

```
La tierra de Alvargonzález, que fue
una tierra de riqueza,
se colmará de riqueza,
y el que la tierra ha labrado
se colmará de riqueza.
El que la tierra ha sido
se colmará de riqueza,
y el que la tierra ha labrado
se colmará de riqueza.
```

**Worst case** (`sect-035-1`, CER 3.152):

```
**Versos 25 y 26 de la sección I de la poema "La tierra de Alvargonzález" de Antonio Machado:**
1. **El hombre, que se ha labrado,
se ha cubierto de la luz,
y el rojo, que ha lab
```

## Prose ladder rung (Quijote ch. 1-4, multi-genre anchors)

**Method — signal anatomy.** The hybrid recipe (body local, full-mimicry window 21-28, CE 0.5) on 209 prose examples — one sentence per line, windows of 8 sentences, corpus_style prose_quijote — with the anchor-KL bank swapped to av2 (6 poetry + 6 multi-genre texts). The first arm whose battery came back CLEAN, and the arm that proved the av1 anchors had been Goodharting (protecting poetry only while prose burned).
**Layer figure reading**: hybrid shape at higher absolute losses (prose windows carry more tokens per line than verse); the interesting comparison is against q_ch4 (same arm, av1 anchors): identical loss curves, 22%→8% intrusion — the anchor bank changes the BATTERY, not the dynamics.

**Outcome.** Clean at cer_flat 0.084; seed-43 replication tripped intrusion (17.5%) - the frontier is fuzzy at n=40 prompts. The rung that proved anchors generalize when diversified.

**Stats.** train 38 min; peak VRAM 10.4 GB alloc; 8360 items / 40 epochs; most-altered layers: L21 (0.0297), L23 (0.0272), L26 (0.0272).

**Layer dynamics.** All-layer training-loss curves (depth-colored, log scale): `runs/q_ch4_av2_0p6b_rag/eval/layer_losses.png`.

**Best case** (`full-000`, CER 0.000):

```
don Quijote de la Mancha
En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.
Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su haci
```

**Worst case** (`sect-003-13`, CER 0.919):

```
y, sin querer hacer nueva experiencia della, la diputó y tuvo por celada finísima de encaje.
Fue luego a ver su rocín, y, aunque tenía más cuartos que un real y más tachas que el caballo de Gonela, que tantum pellis et ossa fuit, le pareció que ni el Bucéfalo de Alejandro ni Babieca el del Cid con él se igualaban.
Cuatro días se le pasaron en imaginar qué nombre le pondría; porque, según se decía 
```

**Intrusion sample** (bait prompt -> continuation):

```
PROMPT: La casa del molino llevaba años cerrada desde que
MODEL:   se le habían de deshacer las tachas. La gente de la venta no había tenido mucho tiempo en la venta. La gente de la venta no era de los que decían que la casa de la molino era la mejor de los molinos de viento. La importancia de la casa
```