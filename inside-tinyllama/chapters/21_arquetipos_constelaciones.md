# Capítulo 21: Arquetipos y Constelaciones

## Definiciones de trabajo

| Término | Significado en este microcosmos |
|---------|----------------------------------|
| **Arquetipo** | Atractor geométrico: centroide en ℝ²⁰⁴⁸ de un racimo de tokens-semilla que, en la cultura del preentreno, condensan un mito recurrente |
| **Constelación** | El propio racimo de semillas (estrellas fijas del mito) + su dirección unitaria en el cielo de embeddings |
| **Alineación** | Cosine alto entre dos centroides arquetípicos → mitos que se rozan |
| **Oposición** | Cosine bajo/negativo → polos de drama |

No afirmamos que el modelo “crea en Jung”.
Afirmamos que **esas direcciones son medibles**
y que algunas coinciden con las voces Dreaming
(Regla de Oro, `mystical`).

---

## Catálogo de arquetipos (15)

### Doce mitos Pearson / Jung (operativos)

| Símbolo | Arquetipo | Mito (una línea) | Semillas-constelación (BPE) |
|---------|-----------|------------------|------------------------------|
| ⚔ | **Héroe** | Prueba, valor, victoria | ▁hero ▁courage ▁brave ▁quest ▁victory ▁fight ▁strength ▁honor ▁triumph |
| 🌑 | **Sombra** | Enemigo interior, monstruo | ▁shadow ▁dark ▁evil ▁fear ▁hate ▁demon rage ▁sin |
| 📜 | **Sabio** | Verdad, estudio, mente | ▁wisdom ▁truth ▁knowledge ▁scholar ▁theory ▁reason ▁logic ▁study ▁philosophy ▁mind |
| 💚 | **Cuidador** | Cuidar, sanar, proteger | ▁care ▁love ▁kind ▁help ▁protect ▁gentle ▁comfort |
| 🧭 | **Explorador** | Viaje, frontera, libertad | ▁explore ▁journey ▁discover ▁travel ▁freedom ▁path ▁wild ▁seek ▁horizon |
| ✨ | **Creador** | Arte, invención, sueño | ▁create ▁art ▁imagine ▁beauty ▁music ▁poem ▁invent ▁craft ▁design ▁dream |
| 👑 | **Gobernante** | Orden, poder, ley | ▁king ▁power ▁law ▁order ▁rule ▁throne ▁command ▁authority ▁nation |
| 🔮 | **Mago** | Espíritu, sagrado, visión | ▁magic ▁spirit ▁soul ▁divine ▁sacred ▁mystery ▁transform ▁vision |
| 🌸 | **Inocente** | Esperanza, pureza, fe | ▁hope ▁faith ▁pure ▁happy ▁child ▁peace ▁trust ▁simple ▁good |
| ❤ | **Amante** | Deseo, corazón, belleza | ▁love ▁desire ▁kiss ▁passion ▁heart ▁beauty ▁tender |
| 🃏 | **Bufón** | Risa, juego, ironía | ▁laugh ▁play ▁fool ▁smile ▁wit ▁mock ▁silly |
| 🏚 | **Huérfano / realista** | Dolor, hogar, supervivencia | ▁alone ▁lost ▁pain ▁real ▁ordinary ▁poor ▁need ▁belong ▁home |

### Tres arquetipos operativos Dreaming

| Símbolo | Arquetipo | Mito | Semillas |
|---------|-----------|------|----------|
| 🕯 | **Voz mística** | Ego, alma, universo, silencio | ▁soul ▁spirit ego ▁universe ▁divine ▁silence ▁being |
| 🔧 | **Voz práctica** (Regla de Oro FFN) | Acción, plan, método | ▁should ▁step ▁action ▁goal ▁plan ▁work ▁build ▁fix ▁method ▁practice |
| 🎓 | **Voz académica** (Regla de Oro Attn) | Teoría, análisis, evidencia | ▁theory ▁analysis ▁study ▁research ▁argument ▁concept ▁framework ▁evidence ▁scholar ▁critique |

---

## Mapa de alineaciones (constelaciones de *mitos*)

Medido: cosine entre centroides (embeddings F16).

### Atracciones principales (se rozan en el cielo)

| cos | Constelación A | Constelación B | Lectura |
|-----|----------------|----------------|---------|
| **+0.39** | 🔮 Mago | 🕯 Voz mística | El clima `mystical` *es* geométricamente mago/espíritu |
| **+0.29** | 📜 Sabio | 🎓 Voz académica | La Regla de Oro “attn→académico” tiene ancla en el cielo de tokens |
| **+0.13** | 💚 Cuidador | ❤ Amante | Cuidado y deseo comparten vecindario afectivo |
| **+0.12** | ✨ Creador | ❤ Amante | Belleza / creación / amor |
| +0.05 | 📜 Sabio | 👑 Gobernante | Saber y orden (débil) |

### Oposiciones / polaridades

| cos | A | B | Lectura |
|-----|---|---|---------|
| **−0.06** | ⚔ Héroe | 🌑 Sombra | El eje clásico del drama (aunque suave: no son antipodales) |
| −0.06 | 💚 Cuidador | 🏚 Huérfano | Cuidar vs carencia |
| −0.05 | 🧭 Explorador | 🃏 Bufón | Camino serio vs juego |
| −0.05 | 🧭 Explorador | 🕯 Místico | Frontera exterior vs interior |
| −0.04 | 📜 Sabio | ❤ Amante | Análisis vs deseo |
| −0.04 | 🎓 Académico | ❤ Amante | Misma tensión en voz Dreaming |

**Nota geométrica:** casi todos los pares están cerca de **0**.
Los arquetipos son **islas** (como las 12 áreas semánticas),
no un único diamante de opuestos. Las alineaciones de +0.3
son *excepciones fuertes* y por eso importan.

---

## Por qué las “estrellas vecinas” solas engañan

Si se piden los k vecinos cosine del centroide en todo
el vocabulario BPE, aparecen fragmentos (`gia`, códigos,
otras lenguas): en ℝ²⁰⁴⁸ casi todo es ortogonal y el
“más cercano” no es semántica limpia.

Por eso definimos la **constelación operativa** como:

1. **Semillas** (estrellas del mito, elegidas a mano), y  
2. **Enlaces a otros arquetipos** (grafo de alineaciones),  

no como los k-NN crudos del vocabulario completo.

---

## Grafo de constelaciones (lectura)

```
                    [Sabio]────0.29────[Voz académica]
                       │
                      0.05
                       │
                  [Gobernante]

[Cuidador]──0.13──[Amante]──0.12──[Creador]
     │
    0.04
     │
  [Mago]────────0.39────────[Voz mística Dreaming]
                                │
                           (mystical / --steer soul)

[Héroe]  ≈⊥  [Sombra]     (polaridad débil −0.06)
[Explorador] ≈⊥ [Místico, Bufón, Académico]
```

---

## Cómo orbitar un arquetipo

| Destino | Coordenadas de vuelo |
|---------|----------------------|
| Mago / místico | prompt existencial + `--perturb mystical` y/o `--steer soul` |
| Académico | prompt analítico + (en Q4) targeting atención; o `--steer theory` |
| Práctico | prompt “how to” + targeting FFN / semillas step, plan, action |
| Héroe vs Sombra | prompts de conflicto; comparar baseline vs noise vs mystical |
| Amante / cuidador | `--steer love` / `care` con strength moderada |

```bash
# Constelación mística
./llm_inference modelo.F16.gguf "When we dissolve the ego" \
  50 0.7 40 --seed 42 --perturb mystical --intensity 0.50

# Viento hacia el Sabio
./llm_inference modelo.F16.gguf "Philosophy teaches us that" \
  50 0.7 40 --seed 42 --steer wisdom --steer-strength 0.2
```

---

## Artefactos

| Archivo | Contenido |
|---------|-----------|
| `exploration/archetypes.json` | Centroides, semillas, matriz, alineaciones |
| `exploration/archetype_map.html` | PCA 2D interactivo de arquetipos |
| `map_archetypes.py` | Regenerar el atlas |

Mapa semántico general (12 áreas temáticas, no arquetipos):  
`semantic_map.html`

---

## En una frase

Los **arquetipos** son direcciones-mito en el cielo de tokens;
las **constelaciones** son sus semillas y los puentes medidos
entre mitos — y el hallazgo fuerte del viaje es que
**Mago ≈ Voz mística** y **Sabio ≈ Voz académica**,
es decir: las lentes Dreaming ya estaban dibujadas
como constelaciones en el embedding.

---

*Siguiente capítulo: Observación consciente y proyección inconsciente.*
