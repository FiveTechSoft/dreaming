# Kapitel 21: Archetypen und Sternbilder

## Arbeitsdefinitionen

| Begriff | Bedeutung in diesem Mikrokosmos |
|---------|--------------------------------|
| **Archetyp** | Geometrischer Attraktor: Zentroid in ℝ²⁰⁴⁸ eines Clusters von Token-Samen, die in der Kultur des Vetrainings ein wiederkehrendes Mythos kondensieren |
| **Sternbild** | Der Cluster selbst (fixierte Sterne des Mythos) + seine Einheitsrichtung am Embedding-Himmel |
| **Ausrichtung** | Hoher Cosine zwischen zwei archetypischen Zentroiden → Mythen, die sich berühren |
| **Gegensatz** | Niedriger/negativer Cosine → Pole des Dramas |

Wir behaupten nicht, dass das Modell „an Jung glaubt".
Wir behaupten, dass **diese Richtungen messbar sind**
und dass einige mit den Dreaming-Stimmen übereinstimmen
(Goldene Regel, `mystical`).

---

## Katalog der Archetypen (15)

### Zwölf Pearson/Jung-Mythen (operative)

| Symbol | Archetyp | Mythos (eine Zeile) | Sternbild-Samen (BPE) |
|--------|----------|---------------------|----------------------|
| ⚔ | **Held** | Prüfung, Mut, Sieg | ▁hero ▁courage ▁brave ▁quest ▁victory ▁fight ▁strength ▁honor ▁triumph |
| 🌑 | **Schatten** | Innerer Feind, Monster | ▁shadow ▁dark ▁evil ▁fear ▁hate ▁demon rage ▁sin |
| 📜 | **Weiser** | Wahrheit, Studium, Geist | ▁wisdom ▁truth ▁knowledge ▁scholar ▁theory ▁reason ▁logic ▁study ▁philosophy ▁mind |
| 💚 | **Hüter** | Pflegen, heilen, schützen | ▁care ▁love ▁kind ▁help ▁protect ▁gentle ▁comfort |
| 🧭 | **Entdecker** | Reise, Grenze, Freiheit | ▁explore ▁journey ▁discover ▁travel ▁freedom ▁path ▁wild ▁seek ▁horizon |
| ✨ | **Schöpfer** | Kunst, Erfindung, Traum | ▁create ▁art ▁imagine ▁beauty ▁music ▁poem ▁invent ▁craft ▁design ▁dream |
| 👑 | **Herrscher** | Ordnung, Macht, Gesetz | ▁king ▁power ▁law ▁order ▁rule ▁throne ▁command ▁authority ▁nation |
| 🔮 | **Zauberer** | Geist, Heilig, Vision | ▁magic ▁spirit ▁soul ▁divine ▁sacred ▁mystery ▁transform ▁vision |
| 🌸 | **Unschuldiger** | Hoffnung, Reinheit, Glaube | ▁hope ▁faith ▁pure ▁happy ▁child ▁peace ▁trust ▁simple ▁good |
| ❤ | **Liebhaber** | Verlangen, Herz, Schönheit | ▁love ▁desire ▁kiss ▁passion ▁heart ▁beauty ▁tender |
| 🃏 | **Narr** | Lachen, Spiel, Ironie | ▁laugh ▁play ▁fool ▁smile ▁wit ▁mock ▁silly |
| 🏚 | **Waise / Realist** | Schmerz, Zuhause, Überleben | ▁alone ▁lost ▁pain ▁real ▁ordinary ▁poor ▁need ▁belong ▁home |

### Drei operative Dreaming-Archetypen

| Symbol | Archetyp | Mythos | Samen |
|--------|----------|--------|-------|
| 🕯 | **Mystische Stimme** | Ich, Seele, Universum, Stille | ▁soul ▁spirit ego ▁universe ▁divine ▁silence ▁being |
| 🔧 | **Praktische Stimme** (FFN Goldene Regel) | Handlung, Plan, Methode | ▁should ▁step ▁action ▁goal ▁plan ▁work ▁build ▁fix ▁method ▁practice |
| 🎓 | **Akademische Stimme** (Attn Goldene Regel) | Theorie, Analyse, Evidenz | ▁theory ▁analysis ▁study ▁research ▁argument ▁concept ▁framework ▁evidence ▁scholar ▁critique |

---

## Ausrichtungskarte (Sternbilder von *Mythen*)

Gemessen: Cosine zwischen Zentroiden (F16-Embeddings).

### Hauptanziehungen (sie berühren sich am Himmel)

| cos | Sternbild A | Sternbild B | Lesart |
|-----|-------------|-------------|--------|
| **+0.39** | 🔮 Zauberer | 🕯 Mystische Stimme | Das `mystical`-Klima *ist* geometrisch Zauberer/Geist |
| **+0.29** | 📜 Weiser | 🎓 Akademische Stimme | Die Goldene Regel „attn→akademisch" hat Anker am Token-Himmel |
| **+0.13** | 💚 Hüter | ❤ Liebhaber | Pflege und Verlangen teilen die affektive Nachbarschaft |
| **+0.12** | ✨ Schöpfer | ❤ Liebhaber | Schönheit / Schöpfung / Liebe |
| +0.05 | 📜 Weiser | 👑 Herrscher | Wissen und Ordnung (schwach) |

### Gegensätze / Polaritäten

| cos | A | B | Lesart |
|-----|---|---|--------|
| **−0.06** | ⚔ Held | 🌑 Schatten | Die klassische Drama-Achse (obwohl schwach: nicht antipodal) |
| −0.06 | 💚 Hüter | 🏚 Waise | Pflege vs. Mangel |
| −0.05 | 🧭 Entdecker | 🃏 Narr | Ernsthafter Weg vs. Spiel |
| −0.05 | 🧭 Entdecker | 🕯 Mystischer | Äußere Grenze vs. innere |
| −0.04 | 📜 Weiser | ❤ Liebhaber | Analyse vs. Verlangen |
| −0.04 | 🎓 Akademischer | ❤ Liebhaber | Dieselbe Spannung in der Dreaming-Stimme |

**Geometrischer Hinweis:** Fast alle Paare liegen nahe **0**.
Die Archetypen sind **Inseln** (wie die 12 semantischen Bereiche),
kein einziger Diamant der Gegensätze. Die Ausrichtungen von +0.3
sind *starke Ausnahmen* und deshalb wichtig.

---

## Warum „Nachbarsterne" allein täuschen

Wenn man die k-NN-Cosines des Zentroids im gesamten
BPE-Vokabular abfragt, erscheinen Fragmente (`gia`, Codes,
andere Sprachen): In ℝ²⁰⁴⁸ ist fast alles orthogonal und
der „Nächste" ist keine saubere Semantik.

Deshalb definieren wir das **operative Sternbild** als:

1. **Samen** (Mythensterne, manuell ausgewählt), und  
2. **Verbindungen zu anderen Archetypen** (Ausrichtungsgraph),  

nicht als die rohen k-NN des gesamten Vokabulars.

---

## Sternbild-Graph (Lesart)

```
                    [Weiser]────0.29────[Akademische Stimme]
                        │
                       0.05
                        │
                   [Herrscher]

[Hüter]──0.13──[Liebhaber]──0.12──[Schöpfer]
     │
    0.04
     │
  [Zauberer]────────0.39────────[Mystische Stimme Dreaming]
                                 │
                            (mystical / --steer soul)

[Held]  ≈⊥  [Schatten]     (schwache Polarität −0.06)
[Entdecker] ≈⊥ [Mystischer, Narr, Akademischer]
```

---

## Wie man einen Archetyp umkreist

| Ziel | Flugkoordinaten |
|------|-----------------|
| Zauberer / mystisch | existenzieller Prompt + `--perturb mystical` und/oder `--steer soul` |
| Akademisch | analytischer Prompt + (bei Q4) Aufmerksamkeits-Targeting; oder `--steer theory` |
| Praktisch | „Wie-man"-Prompt + FFN-Targeting / Samen step, plan, action |
| Held vs. Schatten | Konflikt-Prompts; Baseline mit noise mit mystical vergleichen |
| Liebhaber / Hüter | `--steer love` / `care` mit moderater Stärke |

```bash
# Mystisches Sternbild
./llm_inference modell.F16.gguf "When we dissolve the ego" \
  50 0.7 40 --seed 42 --perturb mystical --intensity 0.50

# Wind zum Weisen
./llm_inference modell.F16.gguf "Philosophy teaches us that" \
  50 0.7 40 --seed 42 --steer wisdom --steer-strength 0.2
```

---

## Artefakte

| Datei | Inhalt |
|-------|--------|
| `exploration/archetypes.json` | Zentroide, Samen, Matrix, Ausrichtungen |
| `exploration/archetype_map.html` | Interaktive 2D-PCA der Archetypen |
| `map_archetypes.py` | Atlas neu generieren |

Allgemeine semantische Karte (12 thematische Bereiche, keine Archetypen):  
`semantic_map.html`

---

## In einem Satz

Die **Archetypen** sind Mythos-Richtungen am Token-Himmel;
die **Sternbilder** sind ihre Samen und die gemessenen Brücken
zwischen Mythen — und der starke Befund der Reise ist, dass
**Zauberer ≈ Mystische Stimme** und **Weiser ≈ Akademische Stimme**,
das heißt: Die Dreaming-Linsen waren bereits als
Sternbilder im Embedding gezeichnet.

---

*Nächstes Kapitel: Bewusste Beobachtung und unbewusste Projektion.*