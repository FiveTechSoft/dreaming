# Capítulo 15: Las Últimas Capas (13–21)

## Integración y decisión

```
Capas 13–20: integración global
Capa 21:     última transformación antes de output_norm
luego:       lm_head → logits → sample
```

Aquí el residual se prepara para el **colapso**
al vocabulario: la fuerza VI del atlas (softmax).

## Qué se juega al final

- Mezcla de temas armados en el medio.  
- Preferencias finas de estilo (formal vs simple).  
- Proximidad a tokens de cierre (`</s>`) — por eso
  a veces baseline y mystical coinciden en salidas
  **muy cortas** con la misma seed (mismo pozo de EOS).

## Experimento de la batería mística

Con I=0.50 y 60 tokens máx., varios prompts llenaron
el presupuesto de longitud; otros cortaron en 2–8 tokens.
Las capas finales + sampling deciden **cuándo parar**
tanto como **qué decir**.

## Regla práctica

Para comparar perspectivas, usa `n` alto y mira
el **cuerpo** del texto, no solo la primera frase
si el modelo se apresura al EOS.

---

*Siguiente capítulo: Áreas Semánticas y el Mapa*
