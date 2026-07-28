# Capítulo 13: Las Primeras Capas (0–5)

## El vestíbulo del microcosmos

Las capas iniciales transforman el embedding “en reposo”
en una representación que ya siente **vecinos** y **sintaxis**.

```
Capa 0:   entrada, patrones muy locales
Capa 1:   sintaxis básica
Capas 2–5: relaciones entre palabras adyacentes
```

(Esta partición es una **hipótesis de trabajo** del proyecto,
guiada por experimentos de ablación y por la literatura
sobre “early = syntax / late = semantics”. No es un corte
rígido en el código.)

## Qué fuerzas dominan aquí

- **Embedding** aún pesa mucho en el residual (inercia del nacimiento).  
- **Atención** empieza a acoplar bigramas y dependencias cortas.  
- **FFN** ajusta el léxico local.

## Señales en el texto

Si una perturbación temprana “rompe” el modelo, a menudo
se ve en **gramática** y tokens raros, no solo en el tono.

Si el baseline suena genérico y el mystical cambia el clima
sin destrozar la sintaxis, las capas tempranas siguen
anclando la lengua.

## Experimento sugerido

Comparar generaciones con targeting solo en `blk.0`–`blk.5`
frente a solo `blk.13`–`blk.21` (scripts v11 / tensor tests).
Hipótesis: early → forma; late → voz y decisión.

---

*Siguiente capítulo: Las Capas Intermedias (6–12)*
