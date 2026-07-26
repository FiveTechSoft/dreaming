# Capítulo 7: Las 10 Técnicas de Perturbación

## 7.1 Introducción

Desarrollamos 10 técnicas de perturbación, cada una con una filosofía diferente. Todas preservan la jerarquía estructural de los pesos.

## 7.2 Categorías

```
┌─────────────────────────────────────────────────────────────┐
│                    Técnicas de Perturbación                 │
├───────────────────┬───────────────────┬─────────────────────┤
│   Espacio         │   Espectral       │   Geométrica        │
│   Lineal          │                   │                     │
├───────────────────┼───────────────────┼─────────────────────┤
│ 1. amplify_subsp  │ 4. spectral       │ 7. normrot          │
│ 2. lowrank        │ 5. dct            │ 8. manifold_pres    │
│ 3. gradient_align │ 6. gradient_dct   │ 9. blkdiag          │
└───────────────────┴───────────────────┴─────────────────────┘
```

## 7.3 Técnica 1: amplify_subspace

**Filosofía**: Amplificar las direcciones principales de varianza.

```python
def amplify_subspace(weights, scale=0.02):
    """Amplificar componente principal."""
    # 1. Calcular varianza por columna
    variance = np.var(weights, axis=0)
    
    # 2. Dirección de máxima varianza
    direction = np.argmax(variance)
    
    # 3. Amplificar solo esa dirección
    perturbed = weights.copy()
    perturbed[:, direction] *= (1 + scale)
    
    return perturbed
```

**Resultado**: Perspectiva filosófica/existencial.

**Por qué funciona**: Amplifica la estructura más prominente del modelo.

## 7.4 Técnica 2: lowrank

**Filosofía**: Modificar solo la componente de bajo rango.

```python
def lowrank_perturbation(weights, rank=1, scale=0.02):
    """Perturbación de bajo rango."""
    # 1. Descomposición SVD
    U, S, Vt = np.linalg.svd(weights, full_matrices=False)
    
    # 2. Modificar componentes principales
    S[:rank] *= (1 + scale)
    
    # 3. Reconstruir
    return U @ np.diag(S) @ Vt
```

**Resultado**: Perspectiva académica/crítica.

**Por qué funciona**: Modifica las relaciones más importantes entre dimensiones.

## 7.5 Técnica 3: gradient_aligned

**Filosofía**: Perturbar en la dirección del gradiente (estimado).

```python
def gradient_aligned_perturbation(weights, scale=0.02):
    """Perturbación alineada con el gradiente."""
    # 1. Estimar gradiente (dirección de mayor sensibilidad)
    gradient = estimate_gradient(weights)
    
    # 2. Normalizar
    gradient = gradient / np.linalg.norm(gradient)
    
    # 3. Perturbar en esa dirección
    return weights + scale * np.abs(weights) * gradient
```

**Resultado**: Perspectiva de autenticidad/descubrimiento.

**Por qué funciona**: Modifica las partes más "sensibles" del modelo.

## 7.6 Técnica 4: spectral

**Filosofía**: Perturbación en el dominio espectral.

```python
def spectral_perturbation(weights, scale=0.02):
    """Perturbación espectral."""
    # 1. FFT
    fft = np.fft.fft(weights)
    
    # 2. Modificar fase
    phase = np.angle(fft)
    magnitude = np.abs(fft)
    
    # 3. Perturbar fase ligeramente
    phase += scale * np.random.randn(*phase.shape)
    
    # 4. Reconstruir
    return np.real(np.fft.ifft(magnitude * np.exp(1j * phase)))
```

**Resultado**: Perspectiva concisa/directa.

**Por qué funciona**: Modifica las relaciones entre frecuencias.

## 7.7 Técnica 5: dct

**Filosofía**: Perturbación en el dominio DCT.

```python
def dct_perturbation(weights, scale=0.02):
    """Perturbación DCT."""
    from scipy.fftpack import dct, idct
    
    # 1. DCT
    dct_coeffs = dct(dct(weights.T, axis=0).T, axis=0)
    
    # 2. Perturbar coeficientes altos
    dct_coeffs[:, 10:] *= (1 + scale)
    
    # 3. IDCT
    return idct(idct(dct_coeffs.T, axis=0).T, axis=0)
```

**Resultado**: Perspectiva creativa.

**Por qué funciona**: Modifica los detalles finos de la representación.

## 7.8 Técnica 6: gradient_dct

**Filosofía**: Combinación de gradiente y DCT.

```python
def gradient_dct_perturbation(weights, scale=0.02):
    """Perturbación gradiente + DCT."""
    # 1. Componente gradiente
    grad_part = gradient_aligned_perturbation(weights, scale/2)
    
    # 2. Componente DCT
    dct_part = dct_perturbation(weights, scale/2)
    
    # 3. Combinar
    return (grad_part + dct_part) / 2
```

**Resultado**: Perspectiva de maximum divergence.

**Por qué funciona**: Combina sensibilidad espectral con sensibilidad espacial.

## 7.9 Técnica 7: normrot

**Filosofía**: Rotación en el espacio normalizado.

```python
def normrot_perturbation(weights, scale=0.02):
    """Rotación normalizada."""
    # 1. Normalizar
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized = weights / (norms + 1e-8)
    
    # 2. Rotar ligeramente
    angle = scale
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # 3. Aplicar rotación a pares de columnas
    perturbed = normalized.copy()
    for i in range(0, weights.shape[1]-1, 2):
        perturbed[:, i:i+2] = normalized[:, i:i+2] @ rotation.T
    
    # 4. Des-normalizar
    return perturbed * norms
```

**Resultado**: Perspectiva estoica/equilibrada.

**Por qué funciona**: Preserva la magnitud pero cambia la dirección.

## 7.10 Técnica 8: manifold_preserving

**Filosofía**: Preservar la manifold de los pesos.

```python
def manifold_preserving_perturbation(weights, scale=0.02):
    """Perturbación preservando manifold."""
    # 1. PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(weights.shape) - 1)
    reduced = pca.fit_transform(weights)
    
    # 2. Perturbar en espacio reducido
    reduced += scale * np.random.randn(*reduced.shape)
    
    # 3. Reconstruir
    return pca.inverse_transform(reduced)
```

**Resultado**: Perspectiva espiritual/contemplativa.

**Por qué funciona**: Mantiene la estructura global pero modifica detalles locales.

## 7.11 Técnica 9: blkdiag

**Filosofía**: Perturbación en bloques diagonales.

```python
def blkdiag_perturbation(weights, scale=0.02):
    """Perturbación por bloques diagonales."""
    perturbed = weights.copy()
    block_size = 32
    
    for i in range(0, weights.shape[0], block_size):
        for j in range(0, weights.shape[1], block_size):
            # Perturmar solo bloques diagonales
            if abs(i - j) < block_size:
                block = weights[i:i+block_size, j:j+block_size]
                perturbed[i:i+block_size, j:j+block_size] = \
                    block * (1 + scale * np.random.randn())
    
    return perturbed
```

**Resultado**: Perspectiva práctica/mecánica.

**Por qué funciona**: Perturba las relaciones locales más que las globales.

## 7.12 Técnica 10: attention_preserving

**Filosofía**: Preservar la estructura de atención.

```python
def attention_preserving_perturbation(weights, scale=0.02):
    """Perturbación preservando atención."""
    # 1. Calcular "importancia" de cada peso
    importance = np.abs(weights) / np.max(np.abs(weights))
    
    # 2. Perturbar inversamente a la importancia
    noise = scale * (1 - importance) * np.random.randn(*weights.shape)
    
    # 3. Aplicar
    return weights * (1 + noise)
```

**Resultado**: Casi idéntico al baseline (diferencia < 0.005).

**Por qué funciona**: Los pesos más importantes se preservan.

## 7.13 Resumen Comparativo

| # | Técnica | Perspectiva | Cambio en KL |
|---|---------|-------------|--------------|
| 1 | amplify_subspace | Filosófica | 0.0032 |
| 2 | lowrank | Académica | 0.0028 |
| 3 | gradient_aligned | Autenticidad | 0.0035 |
| 4 | spectral | Concisa | 0.0025 |
| 5 | dct | Creativa | 0.0030 |
| 6 | gradient_dct | Máxima divergencia | 0.0038 |
| 7 | normrot | Estoica | 0.0022 |
| 8 | manifold_pres | Espiritual | 0.0033 |
| 9 | blkdiag | Práctica | 0.0029 |
| 10 | attn_preserving | Identica | 0.0005 |

## 7.14 ¿Por qué son todas coherentes?

Las 10 técnicas producen texto coherente porque:

1. **Preservan jerarquía** — Las relaciones entre pesos se mantienen
2. **Perturbación pequeña** — 0.02% es menor que la cuantización
3. **Estructura del Transformer** — La arquitectura impone coherencia
4. **Entrenamiento** — Los pesos ya están en un "buen" estado

---

*Siguiente capítulo: [Combinaciones y Targeting Selectivo](08_combinations.md)*
