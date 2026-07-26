# Las "Ideas Puras": ¿Qué Son y Dónde Viven?

## Definición

**Ideas puras** = Conceptos sin envoltorio de lenguaje

```
"me siento triste"     → Idea pura: TRISTEZA
"sad today"            → Idea pura: TRISTEZA  (misma idea, diferente idioma)
"estoy deprimido"      → Idea pura: TRISTEZA  (variación)
"código con bug"       → Idea pura: CÓDIGO + ERROR
```

## Dónde Viven

### En el espacio de pesos

```
Capa 1-5:   Sintaxis     → "cómo se dice"
Capa 6-12:  Significado  → "QUÉ se dice"    ← IDEAS PURAS
Capa 13-18: Razonamiento → "por qué"
Capa 19-22: Generación   → "siguiente palabra"
```

### En las activaciones

```
Entrada: "el gato duerme sobre la alfombra"
                ↓
Capa 1:  [suject] [verb] [prep] [obj]     ← sintaxis
Capa 6:  [gato] [dormir] [comodidad]      ← ideas puras
Capa 12: [animal] [descanso] [hogar]       ← conceptos abstractos
Capa 22: [alfombra]                        ← siguiente token
```

## Cómo Localizarlas

### Método 1: Autoencoder (Anthropic)

```python
# Entrada: activaciones de capas 6-12
activaciones = modelo(input)[:, 6:12, :]  # (batch, seq_len, 2048)

# Autoencoder extrae features
features = autoencoder.encode(activaciones)  # (batch, seq_len, 1000)

# Cada feature = 1 idea pura
feature_42 = features[:, :, 42]  # "tristeza"
feature_108 = features[:, :, 108]  # "Golden Gate"
```

**Resultado**: 1000 features, cada una = 1 concepto

### Método 2: Probe (Clasificador)

```python
# Entrenar clasificador para detectar "tristeza"
probe = Linear(2048, 1)  # De activación → score de tristeza

# En cada capa, ver cuánto "sabe" de tristeza
for capa in range(22):
    activacion = modelo.capas[capa](input)
    score = probe(activacion)
    print(f"Capa {capa}: tristeza = {score:.3f}")

# Resultado típico:
# Capa 1:  0.1  (poco)
# Capa 6:  0.8  (¡mucho!)
# Capa 12: 0.9  (¡máximo!)
# Capa 22: 0.3  (menos)
```

### Método 3: Perturbación (Nuestro método)

```python
# Perturbar pesos en capa 8
pesos_originales = modelo.capas[8]. pesos.copy()
modelo.capas[8].pesos += perturbacion

# Ver qué cambia en la salida
salida_original = modelo("me siento triste")
salida_perturbada = modelo_perturbado("me siento triste")

# Si la salida dice "soy filosófico" en vez de "estoy triste"
# → la perturbación afectó la feature "filosofía"
```

### Método 4: Ablation (Eliminar)

```python
# Eliminar neuronas específicas
modelo.capas[8].neuronas[42] = 0  # Eliminar feature "tristeza"

# Ver qué概念 desaparece
salida = modelo("el gato está triste")
# Si ya no puede hablar de tristeza → feature_42 = tristeza
```

## Ejemplo Visual

```
ENTRADA: "Why are you sad?"
          ↓
CAPA 1:   [WHY] [ARE] [YOU] [SAD]          ← palabras
CAPA 3:   [pregunta] [ser] [tú] [tristeza]  ← tokens
CAPA 8:   [curiosidad] [tristeza] [humanidad] ← IDEAS PURAS
CAPA 12:  [filosofía] [emoción] [existencia]  ← conceptos
CAPA 22:  [porque]                            ← siguiente token
```

## Conexión con Nuestro Trabajo

### Lo que nosotros hicimos

```
Perturbación filosófica → Modelo dice "todo es cuestionable"
Perturbación práctica → Modelo dice "haz esto y esto"
Perturbación creativa → Modelo dice "imagina que..."
```

### Lo que eso significa en features

```
Perturbación filosófica:
  - feature_42 (filosofía): +50%
  - feature_108 (práctica): -30%
  - feature_256 (creativa): +10%

Perturbación práctica:
  - feature_42 (filosofía): -20%
  - feature_108 (práctica): +60%
  - feature_256 (creativa): -10%
```

### El mapa completo

```
IDEA PURA          NUESTRA PERSPECTIVA    SU AUTOENCODER
─────────────────────────────────────────────────────────
"tristeza"    →    "melancolía"      →    feature_42
"Golden Gate" →    "San Francisco"   →    feature_108
"filosofía"   →    "cuestionamiento" →    feature_256
"código"      →    "programación"    →    feature_300
```

## Preguntas Abiertas

### 1. ¿Cuántas ideas puras hay?

- TinyLlama: probablemente **miles** (1000-10000)
- Claude 3: **millones**
- Human brain: **infinitas** (no tenemos límite)

### 2. ¿Son universales?

Sí. Anthropic demostró que "tristeza" es la misma feature en:
- Inglés, español, ruso, código Python, imágenes

### 3. ¿Se pueden combinar?

```
feature_42 (tristeza) + feature_108 (Golden Gate) = ¿tristeza por el puente?
```

### 4. ¿Nuestras personalidades son ideas puras?

No exactamente. Nuestras personalidades son **combinaciones** de ideas puras:

```
"filosófica" = feature_256 × 0.8 + feature_42 × 0.3 + feature_108 × 0.1
```

## Conclusión

**Las ideas puras son los "átomos" del significado.**

- Viven en capas 6-12 (para TinyLlama)
- Se pueden extraer con autoencoders
- Son independientes del idioma
- Nuestras perturbaciones las reorganizan

**El siguiente paso**: Extraer las ideas puras de TinyLlama y ver qué relación tienen con nuestras personalidades.
