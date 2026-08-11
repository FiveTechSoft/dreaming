# Trazador de Cabezas de Atención para TinyLlama (FiveTechSoft / Dreaming)

Este módulo añade la capacidad de **tracear en tiempo de inferencia** qué está haciendo cada una de las 32 cabezas de atención a lo largo de las 22 capas de TinyLlama.

---

## 🛠️ Componentes Creados e Integrados

1. **`trace_attention_heads.py`**:
   - Trazador avanzado en Python utilizando PyTorch + HuggingFace Transformers.
   - Extrae las matrices de atención reales $A_{l,h} \in \mathbb{R}^{N \times N}$ para todas las cabezas y capas ($22 \times 32 = 704$ cabezas).
   - Calcula métricas cuantitativas por cabeza: **Entropía**, **Atención a BOS (Sink)**, **Atención al Token Anterior**, **Atención a Largo Alcance**.
   - Clasifica automáticamente cada cabeza según su comportamiento funcional.
   - Exporta el reporte completo en JSON (`attention_trace_results.json`) y en una **interfaz interactiva HTML** (`attention_trace_vis.html`).

2. **`llm_inference.c` (Motor de inferencia puro C)**:
   - Actualizado para realizar atención multi-cabeza real a través de todas las `NUM_HEADS`.
   - Incorpora el flag `model.trace_heads = 1` para registrar el destino de atención principal por cada capa/cabeza durante la inferencia en C sin dependencias externas.

3. **Visualizador Interactivo HTML (`attention_trace_vis.html`)**:
   - Interfaz web autónoma para navegar por capa y explorar mapas de calor y destinos de atención por palabra.

---

## 🚀 Cómo Ejecutar el Trazador

### Opción 1: Trazador Python + Visualizador Web

```bash
# Ejecutar trazador para cualquier frase deseada
python trace_attention_heads.py --prompt "El perro corrió hacia el jardín porque él tenía mucha hambre." --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Esto generará dos archivos:
* `attention_trace_results.json` (datos estructurados)
* `attention_trace_vis.html` (abrir en el navegador para inspeccionar gráficamente cada cabeza)

### Opción 2: Motor de Inferencia en C (`llm_inference.exe`)

```bash
# Compilar el motor en C
gcc -O2 -o llm_inference.exe llm_inference.c -lm

# Ejecutar inferencia con trazado activo
llm_inference.exe C:/tmp/llm_pequeno.gguf "DREAM" 30
```

---

## 📊 Clasificación Automática de Cabezas

| Tipo de Cabeza | Comportamiento Detectado |
| :--- | :--- |
| ⚓ **Sink / BOS Head** | Deposita >55% de atención en el token de inicio `<s>` cuando no hay coincidencia activa. |
| ⬅️ **Previous Token Head** | Se enfoca >45% del tiempo en la palabra inmediatamente anterior ($i-1$). |
| 🔍 **Long-Range Context** | Atiende a información 3 o más tokens atrás con alta dispersión/contexto. |
| 🎯 **Selective / Focused** | Entropía baja (<1.0), focalizada en un token específico (sustantivo/entidad). |
| 🌐 **Distributed / General** | Atención repartida suavemente en la oración. |
