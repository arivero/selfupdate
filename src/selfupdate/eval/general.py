"""General-capability regression probe.

Memorization fine-tuning can damage general ability (Huang et al. 2024; SDFT).
Cheap proxy: token-level CE on fixed held-out Spanish and English paragraphs
that the training never touches. Report base-vs-trained delta; a large rise
means the poem was bought with catastrophic forgetting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# fixed held-out probes: public-domain prose (Bécquer, Darwin) + neutral text
PROBE_TEXTS = [
    "Volverán las oscuras golondrinas en tu balcón sus nidos a colgar, y otra vez "
    "con el ala a sus cristales jugando llamarán; pero aquellas que el vuelo "
    "refrenaban tu hermosura y mi dicha a contemplar, aquellas que aprendieron "
    "nuestros nombres, ésas... ¡no volverán!",
    "La capital de Francia es París, situada a orillas del río Sena. Es la ciudad "
    "más poblada del país y uno de los principales centros culturales de Europa.",
    "As many more individuals of each species are born than can possibly survive, "
    "and as, consequently, there is a frequently recurring struggle for existence, "
    "it follows that any being which varies in any manner profitable to itself "
    "will have a better chance of surviving.",
    "Para preparar una tortilla de patatas se necesitan huevos, patatas, aceite de "
    "oliva y sal. Primero se pelan y cortan las patatas en láminas finas.",
]


@torch.no_grad()
def general_ce(model, tokenizer, device: str = "cuda") -> dict:
    """Mean next-token CE per probe text and overall."""
    was_training = model.training
    model.eval()
    ces = []
    for text in PROBE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        t = torch.tensor([ids], device=device)
        logits = model(t, use_cache=False).logits[0].float()
        ce = F.cross_entropy(logits[:-1], t[0, 1:]).item()
        ces.append(ce)
    if was_training:
        model.train()
    return {"per_text": ces, "mean_ce": sum(ces) / len(ces)}
