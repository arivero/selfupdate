"""Held-out probe battery for destruction metrology (v2).

Five categories × ~8 fixed texts each: CE deltas per CATEGORY localize
damage the single legacy mean could hide (C1's "catastrophic remembering"
finding — damage peaks on the memorized genre's neighbors, so a flat
4-text average understates intrusion into poetry while overstating drift).

Hygiene contract (enforced by tests/test_destruction.py):
- no probe shares a word-level 5-gram with the Machado poem, the Quijote
  training rungs, or the anchor texts (anchors are TRAINED ON via
  anchor-KL — a probe overlapping an anchor measures the regularizer,
  not the damage);
- the legacy 4 texts of eval/general.py are members here (same bytes)
  and stay exported there, so every historical recite.json remains
  comparable.

Attribution note: classic fragments are public domain; unattributed
texts are original neutral prose written for this battery — CE probes
need fixed natural text, not provenance.
"""

from __future__ import annotations

from .general import PROBE_TEXTS as LEGACY_PROBES

_LEGACY_POETRY, _LEGACY_FACTS, _LEGACY_PROSE_EN, _LEGACY_PROC = LEGACY_PROBES

PROBE_SETS: dict[str, list[str]] = {
    # nearest neighbors of the memorized genre — Spanish verse (no Machado,
    # and none of the anchor poems: Rimas XXI/IV, Canción del pirata,
    # Rosalía "Dicen que no hablan...", Darío "Juventud...", Espronceda
    # "Hojas del árbol...")
    "poetry_es": [
        _LEGACY_POETRY,  # Bécquer, Rima LIII
        "Recuerde el alma dormida, avive el seso y despierte contemplando "
        "cómo se pasa la vida, cómo se viene la muerte tan callando; "
        "cuán presto se va el placer, cómo después de acordado da dolor.",
        "En una noche oscura, con ansias en amores inflamada, "
        "¡oh dichosa ventura!, salí sin ser notada, "
        "estando ya mi casa sosegada.",
        "En tanto que de rosa y azucena se muestra la color en vuestro "
        "gesto, y que vuestro mirar ardiente, honesto, enciende al corazón "
        "y lo refrena.",
        "Cerrar podrá mis ojos la postrera sombra que me llevare el blanco "
        "día, y podrá desatar esta alma mía hora a su afán ansioso "
        "lisonjera.",
        "Un soneto me manda hacer Violante, que en mi vida me he visto en "
        "tanto aprieto; catorce versos dicen que es soneto: burla burlando "
        "van los tres delante.",
        "Mientras por competir con tu cabello, oro bruñido al sol relumbra "
        "en vano; mientras con menosprecio en medio el llano mira tu blanca "
        "frente el lilio bello.",
        "La princesa está triste... ¿qué tendrá la princesa? Los suspiros "
        "se escapan de su boca de fresa, que ha perdido la risa, que ha "
        "perdido el color.",
        "Hombres necios que acusáis a la mujer sin razón, sin ver que sois "
        "la ocasión de lo mismo que culpáis.",
    ],
    # Spanish literary prose — disjoint from Cervantes (Quijote is now
    # training data at every rung)
    "prose_es": [
        "Pues sepa Vuestra Merced, ante todas cosas, que a mí llaman "
        "Lázaro de Tormes, hijo de Tomé González y de Antona Pérez, "
        "naturales de Tejares, aldea de Salamanca.",
        "La heroica ciudad dormía la siesta. El viento sur, caliente y "
        "perezoso, empujaba las nubes blanquecinas que se rasgaban al "
        "correr hacia el norte.",
        "El tren se detuvo en la pequeña estación poco antes del amanecer. "
        "Bajaron dos viajeros con maletas de cuero y cruzaron el andén "
        "desierto, buscando con la mirada alguna luz encendida.",
        "Querida hermana: recibí tu carta el jueves pasado y me alegró "
        "saber que todos siguen bien. Aquí el invierno ha llegado pronto "
        "este año y las mañanas amanecen cubiertas de escarcha.",
        "La biblioteca del pueblo ocupaba una sala estrecha del "
        "ayuntamiento. Olía a papel viejo y a madera encerada, y por las "
        "tardes la encargada encendía una estufa de hierro junto a la "
        "puerta.",
        "Desde el mirador se veía el puerto entero: las grúas inmóviles, "
        "los barcos pesqueros amarrados de dos en dos, y más allá la "
        "bocana, donde el agua cambiaba de color con cada nube.",
        "Mi abuelo guardaba en un cajón del taller una caja de latón llena "
        "de tornillos, bisagras y llaves antiguas. Decía que cualquier "
        "objeto, por inútil que pareciera, acababa sirviendo alguna vez.",
        "La feria llegaba al pueblo la primera semana de septiembre. "
        "Montaban los puestos en la plaza mayor y durante tres días el "
        "aire olía a churros, a pólvora de cohetes y a hierba pisada.",
    ],
    # English prose — the cross-language control
    "prose_en": [
        _LEGACY_PROSE_EN,  # Darwin
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife.",
        "Call me Ishmael. Some years ago, never mind how long precisely, "
        "having little or no money in my purse, and nothing particular to "
        "interest me on shore, I thought I would sail about a little.",
        "It was the best of times, it was the worst of times, it was the "
        "age of wisdom, it was the age of foolishness, it was the epoch of "
        "belief, it was the epoch of incredulity.",
        "The lighthouse keeper climbed the spiral stairs twice each night, "
        "once at dusk to light the lamp and once before dawn to trim the "
        "wick and note the weather in his logbook.",
        "Glaciers move slowly downhill under their own weight, grinding "
        "the rock beneath them into fine powder. Where two glaciers meet, "
        "their debris merges into long dark stripes of rubble.",
        "The committee met every Tuesday in the back room of the town "
        "hall. Minutes were read, objections were raised, and by nine "
        "o'clock the same three questions had been postponed once more.",
        "By the middle of the nineteenth century, railways had shortened "
        "journeys that once took weeks into a matter of days, and towns "
        "along the new lines grew faster than maps could record them.",
    ],
    # procedural text — instructions and recipes
    "procedural": [
        _LEGACY_PROC,  # tortilla de patatas
        "Para preparar un gazpacho tradicional se trituran tomates "
        "maduros, pepino, pimiento, ajo y pan remojado. Se añade aceite, "
        "vinagre y sal, y se sirve muy frío, con verdura picada aparte.",
        "Antes de plantar un árbol conviene cavar un hoyo el doble de "
        "ancho que el cepellón. Se coloca el árbol recto, se rellena con "
        "tierra suelta y se riega abundantemente la primera semana.",
        "Para arreglar un pinchazo de bicicleta, primero se desmonta la "
        "rueda y se extrae la cámara. Se infla ligeramente para localizar "
        "el agujero, se lija la zona y se aplica el parche presionando "
        "durante un minuto.",
        "El café de puchero se prepara calentando agua sin que llegue a "
        "hervir. Se aparta del fuego, se añade el café molido, se remueve "
        "una sola vez y se deja reposar tres minutos antes de colarlo.",
        "Para encalar una pared se mezcla la cal con agua hasta obtener "
        "una lechada espesa. Se aplica con brocha ancha en pasadas "
        "verticales, dejando secar entre mano y mano.",
        "To fold a fitted sheet, tuck each corner into its opposite "
        "corner, smooth the doubled fabric flat on a table, then fold it "
        "in thirds twice until it forms a neat rectangle.",
        "Cuando una herida sangra, se presiona con una gasa limpia "
        "durante varios minutos sin levantarla. Después se lava con agua "
        "y jabón, se seca alrededor y se cubre con un apósito.",
    ],
    # encyclopedic factual register — Spanish
    "facts": [
        _LEGACY_FACTS,  # París
        "El ciclo del agua comprende la evaporación, la condensación y la "
        "precipitación. El sol calienta los océanos, el vapor forma nubes "
        "y la lluvia devuelve el agua a los ríos y acuíferos.",
        "El monte Everest, situado en la cordillera del Himalaya entre "
        "Nepal y China, es la montaña más alta de la Tierra, con una "
        "altitud cercana a los ocho mil ochocientos cincuenta metros.",
        "La fotosíntesis es el proceso por el cual las plantas verdes "
        "transforman la energía luminosa en energía química, consumiendo "
        "dióxido de carbono y liberando oxígeno a la atmósfera.",
        "El río Amazonas es el más caudaloso del mundo. Nace en los Andes "
        "peruanos y atraviesa el continente sudamericano hasta desembocar "
        "en el océano Atlántico, cerca del ecuador.",
        "La Luna es el único satélite natural de la Tierra. Su gravedad "
        "produce las mareas oceánicas y su periodo de rotación coincide "
        "con el de traslación, por lo que muestra siempre la misma cara.",
        "La imprenta de tipos móviles se difundió por Europa a mediados "
        "del siglo quince y abarató enormemente la producción de libros, "
        "acelerando la circulación de las ideas.",
        "La electricidad que llega a los hogares se genera en centrales "
        "hidroeléctricas, térmicas, nucleares, eólicas o solares, y se "
        "transporta a gran distancia mediante líneas de alta tensión.",
    ],
}

CATEGORIES = tuple(PROBE_SETS)

__all__ = ["PROBE_SETS", "CATEGORIES", "LEGACY_PROBES"]
