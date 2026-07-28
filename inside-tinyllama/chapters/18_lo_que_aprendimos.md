# Capítulo 18: Lo que Aprendimos

## Hallazgos principales

1. **TinyLlama es un microcosmos cartografiable**  
   22 capas, 9 tensores/capa, dims reales 2048 / 5632 / GQA 32×4.

2. **Un motor C propio cierra el círculo**  
   GGUF F16, BPE, KV-cache, OpenMP, ~6–10 tok/s,
   `--perturb` y `--steer` en runtime.

3. **Los pesos contienen perspectivas**  
   No solo hechos: tonos y voces. Perturbar con
   jerarquía preservada cambia la voz, no apaga el habla.

4. **Regla de Oro geométrica**  
   Attn → académico; FFN → práctico; Emb → simple.

5. **Superficie de coherencia**  
   Tangente (amplify) habitable; normal (noise fuerte) vacío.

6. **Espacio de embeddings: islas, no un eje único**  
   Doce áreas semánticas casi ortogonales; PCA usa
   cientos de dimensiones; opposites no antipodales.

7. **Macrocosmos ↔ microcosmos**  
   El método es ida y vuelta: sentido ↔ tensor ↔ texto.

8. **Herramientas de viaje**  
   Mapa HTML en GitHub, scripts de geometría, llama-cli
   para baterías Q4, motor C para relojería fina.

## Limitaciones del estudio

- Evaluación de “perspectiva” aún cualitativa.  
- TinyLlama ≠ modelos frontera (la superficie puede cambiar).  
- Mapa 2D es proyección, no la geometría verdadera.  
- Runtime F32 de perturbación exige mucha RAM.  
- No todas las técnicas v10/v11 están en el motor C.

## Preguntas abiertas

- ¿Dónde (qué capas) se enciende el clima místico en el residual?  
- ¿Se transfieren las direcciones de perspectiva entre modelos?  
- ¿Cómo medir perspectiva de forma automática y fiable?  
- ¿Qué pasa en la superficie de coherencia a 7B / 70B?

---

*Siguiente capítulo: El Futuro de la Exploración*
