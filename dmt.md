# DMT Experience Notes

imagina una experiencia con DMT. El sujeto no esta experimentando nada real, sino experiencias basadas en sus propias experiencias

## Desentrañó analogía entre experiencias internas y perturbaciones computacionales

Buena analogía — y bastante precisa técnicamente. Un DMT no inventa contenido de la nada: reconfigura temporalmente la dinámica de una red que ya contiene toda su experiencia previa codificada. La "alucinación" es información real del sistema, pero desordenada en su forma de combinarse.

Traducido a TinyLlama, eso apunta a una técnica específica: perturbación de pesos que preserve la norma/estadística pero rompa la estructura fina — no ruido aleatorio puro (eso sería más bien "anestesia", degradación hacia basura), sino ruido correlacionado con la propia distribución de cada tensor, de modo que el modelo siga "hablando su propio idioma" pero con las asociaciones desplazadas.

### Algunas formas concretas de hacerlo

```python
import torch

def dmt_perturb(param, intensity=0.3, mode="scaled_noise"):
    if mode == "scaled_noise":
        # Ruido proporcional a la magnitud local de cada peso:
        # no destruye la jerarquía de importancia, solo la desplaza
        noise = torch.randn_like(param) * param.abs() * intensity
        return param + noise

    elif mode == "row_shuffle":
        # Reordena parcialmente filas dentro de una matriz de proyección:
        # las "palabras" siguen siendo del modelo, la sintaxis se distorsiona
        n = param.shape[0]
        idx = torch.randperm(n)
        mix = torch.rand(n, 1, device=param.device) < intensity
        shuffled = param[idx]
        return torch.where(mix, shuffled, param)

    elif mode == "amplify_subspace":
        # Amplifica las direcciones de mayor varianza (SVD) y atenúa el resto:
        # exagera los "rasgos dominantes" del conocimiento del modelo
        U, S, Vh = torch.linalg.svd(param, full_matrices=False)
        S = S.clone()
        k = max(1, int(len(S) * 0.1))
        S[:k] *= (1 + intensity)
        S[k:] *= (1 - intensity * 0.5)
        return U @ torch.diag(S) @ Vh
```

### Diferencias experimentales relevantes

- **scaled_noise** con intensidad baja (0.05–0.15) da algo parecido a "leve alteración perceptual" — el modelo sigue siendo coherente pero con asociaciones raras.
- **row_shuffle** parcial en capas de atención tardías es lo más parecido a "las cosas se funden unas con otras" — mezcla identidades de tokens/conceptos que el modelo ya conocía.
- **amplify_subspace** vía SVD es lo más interesante para tu hipótesis: no añade nada externo, solo reponderá lo que el modelo ya "sabe", exagerando lo dominante — que es literalmente la idea de que la experiencia es material propio reorganizado, no contenido nuevo.

¿Quieres que lo montemos como script que aplique esto a capas específicas (por ejemplo solo mlp de las capas intermedias) y compare generaciones antes/después con el mismo prompt?
