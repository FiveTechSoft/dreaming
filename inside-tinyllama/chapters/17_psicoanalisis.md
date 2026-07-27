# Capítulo 17: Psicoanálisis del Transformer

## Una metáfora (no un diagnóstico clínico)

Freud distinguía capas de lo mental.
Sin forzar identidad, el stack del transformer
admite una **lectura por profundidad**:

| Instancia | Componente | Función aproximada |
|-----------|------------|-------------------|
| **Inconsciente** | Embeddings | Asociaciones latentes, “lo ya sabido” sin contexto |
| **Preconsciente** | Atención + capas medias | Trae a escena relaciones y marcos |
| **Consciente** | Últimas capas + logits + sample | Lo que se dice *ahora* |

## Ello / Yo / Superyó (lectura libre)

| | Analogía en el modelo |
|--|------------------------|
| **Ello** | Impulsos de peso bruto, direcciones semánticas crudas |
| **Yo** | Residual + normas: negocia entre impulsos y forma |
| **Superyó** | Sesgos de entrenamiento / seguridad / estilo “correcto” del baseline |

La perturbación `mystical` no “libera el ello” en sentido freudiano:
**remezcla** el equilibrio de voces ya presentes en los pesos.

## Por qué anotar esta metáfora

- Ayuda a *hablar* del interior sin solo matrices.  
- Enlaza con el zoom macro↔micro (cap. 6).  
- No sustituye medidas: es un **mapa narrativo**.

## Límite

Un LLM no tiene inconsciente subjetivo.
Tiene **estadística comprimida**. La metáfora es
herramienta de exploración, no ontología.

---

*Siguiente capítulo: Lo que Aprendimos*
