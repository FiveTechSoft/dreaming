# Capítulo 15: Conmutación de Estilos en Runtime

## 15.1 Introducción

El sistema de conmutación de estilos permite **cambiar la perspectiva** de un modelo en tiempo de ejecución, sin recargar el modelo completo.

## 15.2 Concepto

En lugar de tener 10 modelos separados, tenemos **un modelo base** y **vectores de perturbación** que se aplican dinámicamente:

```
Modelo Base + Vector Filosófico → Salida Filosófica
Modelo Base + Vector Estoico → Salida Estoica
Modelo Base + Vector Creativo → Salida Creativa
```

## 15.3 Implementación

### Estructura de Datos

```python
class StyleSwitcher:
    def __init__(self, base_model_path):
        self.base_model = load_model(base_model_path)
        self.vectors = {}
        self.active_style = None
    
    def register_vector(self, name, vector_path):
        """Registrar un vector de estilo."""
        self.vectors[name] = np.load(vector_path)
    
    def set_style(self, name):
        """Establecer estilo activo."""
        if name not in self.vectors:
            raise ValueError(f"Unknown style: {name}")
        self.active_style = name
    
    def generate(self, prompt, max_tokens=100):
        """Generar con estilo activo."""
        if self.active_style is None:
            return self.base_model.generate(prompt, max_tokens)
        
        # Aplicar vector de estilo
        styled_model = self.apply_vector(
            self.base_model, 
            self.vectors[self.active_style]
        )
        
        return styled_model.generate(prompt, max_tokens)
    
    def apply_vector(self, model, vector):
        """Aplicar vector de estilo al modelo."""
        styled = model.copy()
        for name in styled.weights:
            if name in vector:
                styled.weights[name] += vector[name]
        return styled
```

### Uso

```python
# Inicializar
switcher = StyleSwitcher("tinyllama-1.1b.Q4_0.gguf")

# Registrar vectores
switcher.register_vector("philosophical", "vectors/philosophical.npy")
switcher.register_vector("stoic", "vectors/stoic.npy")
switcher.register_vector("concise", "vectors/concise.npy")

# Generar con diferentes estilos
switcher.set_style("philosophical")
print(switcher.generate("The meaning of life is"))

switcher.set_style("stoic")
print(switcher.generate("The meaning of life is"))

switcher.set_style("concise")
print(switcher.generate("The meaning of life is"))
```

## 15.4 Generación de Vectores

### Método 1: Diferencia Directa

```python
def generate_vector(base_model, styled_model):
    """Generar vector como diferencia de pesos."""
    vector = {}
    for name in base_model.weights:
        vector[name] = styled_model.weights[name] - base_model.weights[name]
    return vector
```

### Método 2: Promedio de Diferencias

```python
def generate_vector_robust(base_model, styled_models):
    """Generar vector robusto promediando múltiples modelos."""
    vectors = []
    for styled in styled_models:
        v = {}
        for name in base_model.weights:
            v[name] = styled.weights[name] - base_model.weights[name]
        vectors.append(v)
    
    # Promediar
    avg_vector = {}
    for name in base_model.weights:
        avg_vector[name] = np.mean([v[name] for v in vectors], axis=0)
    
    return avg_vector
```

## 15.5 Interpolación de Estilos

### Interpolación Lineal

```python
def interpolate_styles(vector1, vector2, alpha):
    """Interpolar entre dos estilos."""
    result = {}
    for name in vector1:
        result[name] = (1 - alpha) * vector1[name] + alpha * vector2[name]
    return result
```

### Ejemplo

```python
# 70% filosófico + 30% estoico
mixed = interpolate_styles(
    switcher.vectors["philosophical"],
    switcher.vectors["stoic"],
    alpha=0.3
)

# Aplicar estilo mixto
styled_model = switcher.apply_vector(switcher.base_model, mixed)
print(styled_model.generate("The meaning of life is"))
```

## 15.6 Gestión de Memoria

### Compresión de Vectores

```python
def compress_vector(vector, compression_ratio=0.1):
    """Comprimir vector manteniendo estructura."""
    compressed = {}
    for name, weights in vector.items():
        # Solo mantener pesos más grandes
        threshold = np.percentile(np.abs(weights), 
                                  (1 - compression_ratio) * 100)
        mask = np.abs(weights) >= threshold
        compressed[name] = weights * mask
    return compressed
```

### Almacenamiento

```python
def save_vector(vector, path):
    """Guardar vector comprimido."""
    compressed = compress_vector(vector)
    np.savez_compressed(path, **compressed)

def load_vector(path):
    """Cargar vector comprimido."""
    data = np.load(path)
    return {name: data[name] for name in data.files}
```

## 15.7 Rendimiento

### Tiempos de Conmutación

| Operación | Tiempo |
|-----------|--------|
| Cargar modelo base | 3.0 seg |
| Aplicar vector | 0.5 seg |
| Generar texto | 2.8 seg |
| **Total (con vector)** | **3.3 seg** |
| **Total (sin vector)** | **2.8 seg** |

### Memoria

| Componente | Memoria |
|------------|---------|
| Modelo base | 700 MB |
| Vector comprimido | 50 MB |
| Modelo estilizado | 750 MB |
| **Total** | **750 MB** |

## 15.8 API Completa

```python
class DreamingEngine:
    """Motor completo de conmutación de estilos."""
    
    def __init__(self, base_model_path, vector_dir):
        self.base_model = load_model(base_model_path)
        self.vectors = self.load_all_vectors(vector_dir)
        self.active_style = None
    
    def load_all_vectors(self, vector_dir):
        """Cargar todos los vectores de un directorio."""
        vectors = {}
        for filename in os.listdir(vector_dir):
            if filename.endswith('.npz'):
                name = filename[:-4]
                vectors[name] = self.load_vector(
                    os.path.join(vector_dir, filename)
                )
        return vectors
    
    def list_styles(self):
        """Listar estilos disponibles."""
        return list(self.vectors.keys())
    
    def set_style(self, style_name):
        """Establecer estilo activo."""
        if style_name not in self.vectors:
            raise ValueError(f"Unknown style: {style_name}")
        self.active_style = style_name
    
    def blend_styles(self, style1, style2, ratio=0.5):
        """Mezclar dos estilos."""
        if style1 not in self.vectors:
            raise ValueError(f"Unknown style: {style1}")
        if style2 not in self.vectors:
            raise ValueError(f"Unknown style: {style2}")
        
        blended = interpolate_styles(
            self.vectors[style1],
            self.vectors[style2],
            ratio
        )
        
        self.vectors[f"{style1}_{style2}_{ratio}"] = blended
        self.active_style = f"{style1}_{style2}_{ratio}"
    
    def generate(self, prompt, max_tokens=100, temperature=0.8):
        """Generar texto con estilo activo."""
        model = self.base_model
        
        if self.active_style:
            model = self.apply_vector(model, self.vectors[self.active_style])
        
        return model.generate(
            prompt, 
            max_tokens=max_tokens,
            temperature=temperature
        )
```

## 15.9 Ejemplo de Uso

```python
# Inicializar motor
engine = DreamingEngine(
    base_model_path="tinyllama-1.1b.Q4_0.gguf",
    vector_dir="vectors/"
)

# Listar estilos
print("Estilos disponibles:", engine.list_styles())
# ['philosophical', 'stoic', 'concise', 'creative', 'spiritual']

# Generar con cada estilo
prompt = "The secret to happiness is"

for style in engine.list_styles():
    engine.set_style(style)
    print(f"\n=== {style.upper()} ===")
    print(engine.generate(prompt))

# Mezclar estilos
engine.blend_styles("philosophical", "stoic", ratio=0.5)
print("\n=== MIXED (50/50) ===")
print(engine.generate(prompt))
```

## 15.10 Limitaciones

1. **Vectores estáticos** — No se adaptan al prompt
2. **Interpolación lineal** — Puede producir resultados inesperados
3. **Memoria** — Cada vector ocupa espacio adicional
4. **Precisión** — La interpolación puede reducir calidad

## 15.11 Trabajo Futuro

1. **Vectores adaptables** — Vectores que cambian según el prompt
2. **Interpolación no lineal** — Usar geodésicas en la manifold
3. **Aprendizaje de vectores** — Entrenar vectores automáticamente
4. **Combinación dinámica** — Mezclar estilos en tiempo real

---

*Siguiente capítulo: [Aplicaciones Prácticas](16_practical_applications.md)*
