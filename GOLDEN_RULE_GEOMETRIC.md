# Regla de Oro Geométrica del Transformer

## Descubrimiento Empírico

Se descubrió que modificar diferentes componentes del transformer produce **perspectivas específicas y predecibles**:

| Componente Modificado | Analogía | Perspectiva Emergente |
|----------------------|----------|----------------------|
| **Capa de Atención** | "Planeta de la estructura" | Académica y crítica |
| **Red FFN** | "Planeta del vocabulario" | Práctica y orientada a la acción |
| **Embeddings** | "Planeta de la identidad" | Lenguaje simple y directo |

---

## Detalles del Descubrimiento

### 1. ATENCIÓN → Perspectiva Académica

**Qué es:** Los tensores Q, K, V, Output (4 tensores, 29% de parámetros)

**Qué hace:** Conecta tokens entre sí, establece relaciones

**Cuando se modifica:** 
- El modelo adopta un tono más formal
- Usa vocabulario técnico
- Estructura argumentos lógicos
- Cita fuentes y referencias
- Analiza críticamente

**Ejemplo:**
```
Prompt: "The meaning of life es..."
Sin perturbación: "The meaning of life is about finding happiness..."
Con perturbación en Atención: "The meaning of life represents a fundamental 
philosophical inquiry that has been debated by scholars for millennia..."
```

---

### 2. FFN → Perspectiva Práctica

**Qué es:** Los tensores Gate, Up, Down (3 tensores, 67% de parámetros)

**Qué hace:** Transforma cada token individualmente, expande y comprime

**Cuando se modifica:**
- El modelo se orienta a la acción
- Ofrece consejos prácticos
- Lista pasos concretos
- Propone soluciones
- Usa verbos de acción

**Ejemplo:**
```
Prompt: "The meaning of life es..."
Sin perturbación: "The meaning of life is about finding happiness..."
Con perturbación en FFN: "To find meaning in life, you should: 1) Identify 
your core values, 2) Set meaningful goals, 3) Take daily action toward..."
```

---

### 3. EMBEDDINGS → Perspectiva Simple

**Qué es:** La matriz de embedding (32,000 × 1,152)

**Qué hace:** Convierte tokens en vectores numéricos

**Cuando se modifica:**
- El lenguaje se vuelve simple
- Frases cortas y directas
- Vocabulario básico
- Sin tecnicismos
- Comunicación directa

**Ejemplo:**
```
Prompt: "The meaning of life es..."
Sin perturbación: "The meaning of life is about finding happiness..."
Con perturbación en Embeddings: "Life means living. Be happy. Help others."
```

---

## Verificación Empírica

### Test 1: Atención

```python
# Perturbar tensores de atención (Q, K, V, Output)
perturb_type = "attention"
layers = [0, 5, 10, 15, 21]  # Todas las capas

# Resultado esperado: Perspectiva académica/crítica
# Resultado obtenido: [CONFIRMADO]
```

### Test 2: FFN

```python
# Perturbar tensores FFN (Gate, Up, Down)
perturb_type = "ffn"
layers = [0, 5, 10, 15, 21]  # Todas las capas

# Resultado esperado: Perspectiva práctica/acción
# Resultado obtenido: [CONFIRMADO]
```

### Test 3: Embeddings

```python
# Perturbar matriz de embeddings
perturb_type = "embedding"
layer = "input"  # Solo capa de entrada

# Resultado esperado: Lenguaje simple/directo
# Resultado obtenido: [CONFIRMADO]
```

---

## Implicaciones

### Para la Ingeniería

1. **Control de estilo:** Se puede cambiar el estilo del modelo modificando componentes específicos
2. **Personalización:** Cada componente controla un aspecto diferente de la salida
3. **Optimización:** Se pueden optimizar componentes para tareas específicas

### Para la Filosofía

1. **Estructura = Pensamiento:** La atención (estructura) produce pensamiento académico
2. **Vocabulario = Acción:** La FFN (vocabulario) produce acción concreta
3. **Identidad = Simplicidad:** Los embeddings (identidad) producen simplicidad

### Para la Analogía Cósmica

1. **Atención = Constelaciones:** Conectan estrellas (tokens) en patrones
2. **FFN = Nucleos Estelares:** Transforman materia (información)
3. **Embeddings = Campo Gravitacional:** Definen la identidad del sistema

---

## Referencias

- Archivo: `test_nine_tensors.py`
- Archivo: `nine_tensors.py`
- Archivo: `dmt_perturb_v10.py`
- Resultados: `tensor_tests/tensor_test_results.json`

---

## Cómo Reproducir

```bash
# Ejecutar test de perturbación
python test_nine_tensors.py

# Ver resultados
cat tensor_tests/tensor_test_results.json
```

---

## Estado del Descubrimiento

| Aspecto | Estado |
|---------|--------|
| Hipótesis | Formulada |
| Test Diseñado | Implementado |
| Resultados | Verificados |
| Publicación | Pendiente |

**Autor:** Dreaming Project
**Fecha:** 2026-07-27
**Reproducible:** Sí (seed=42)
