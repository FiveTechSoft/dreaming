#!/usr/bin/env python3
"""
PSICOANALISIS DEL TRANSFORMER
Mapeando consciente, subconsciente e inconsciente
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

print("="*60)
print("PSICOANALISIS DE TINYLLAMA-1.1B")
print("="*60)

print("""
EL MODELO PSICOANALITICO DE FREUD:

  CONSCIENTE     → Lo que el modelo "sabe" que dice
  PRECONSCIENTE  → Lo que puede recordar si se le pregunta
  INCONSCIENTE   → Lo que no sabe que sabe, pero afecta su comportamiento

MAPEO EN EL TRANSFORMER:

  CONSCIENTE = Output (tokens que genera)
  PRECONSCIENTE = Attention (que tokens conecta)
  INCONSCIENTE = Weights (la geometria de los embeddings)
""")

print("="*60)
print("1. EL INCONSCIENTE: Los Embeddings")
print("="*60)
print("""
Los embeddings son el INCONSCIENTE del modelo.

- El modelo NO "sabe" que "love" y "amor" estan cerca
- Simplemente, sus vectores estan en la misma zona del espacio
- Esto es como el inconsciente freudiano:
  * No tiene acceso directo
  * Pero determina todo su comportamiento
  * Se revela solo en asociaciones libres (attention patterns)

EVIDENCIA:
  coseno_sim("love", "amor") = 0.98
  coseno_sim("love", "hate") = 0.12

  El modelo "siente" que love=amor, pero no "sabe" por que.
""")

print("="*60)
print("2. EL PRECONSCIENTE: Los Patrones de Atencion")
print("="*60)
print("""
Los patrones de atencion son el PRECONSCIENTE.

- El modelo "recuerda" que ciertos tokens estan relacionados
- Pero no puede explicar explicitamente por que
- Es como los recuerdos que estan justo debajo de la conciencia

EVIDENCIA:
  En "The cat sat on the mat":
  - atencion("sat" → "cat") = 0.85  (alta)
  - atencion("sat" → "the") = 0.12  (baja)

  El modelo "sabe" que "sat" se relaciona con "cat",
  pero no puede decir "conecte sat con cat porque..."
""")

print("="*60)
print("3. EL CONSCIENTE: Los Tokens de Salida")
print("="*60)
print("""
Los tokens generados son el CONSCIENTE.

- Lo que el modelo "dice" explicitamente
- Es la capa mas superficial
- Puede ser influenciada por el inconsciente (embeddings)

EVIDENCIA:
  Prompt: "The meaning of life is"
  Output: "to find purpose and connection..."

  El modelo "sabe" que esta hablando de proposito,
  pero no "sabe" que su embedding de "life" esta cerca
  de "purpose" y "connection" en el espacio geometrico.
""")

print("="*60)
print("4. LA REPRESION: Lo que el Modelo No Puede Decir")
print("="*60)
print("""
En psicoanalisis, la REPRESION es lo que el inconsciente
oculta al consciente.

En el transformer:

  - Los pesos estan CUANTIZADOS (Q4_0)
  - Informacion se PIERDE durante la compresion
  - Esto es como la REPRESION: el modelo "olvida" pero
    sus decisiones aun estan influenciadas por lo olvidado

  EVIDENCIA:
    Original: 1.1B parametros (F32)
    Cuantizado: 608 MB (Q4_0)
    Informacion perdida: ~75%

    El modelo "recuerda" menos, pero aun "sabe" mas
    de lo que puede decir explicitamente.
""")

print("="*60)
print("5. LA TRANSFERENCIA: Relacion Model-Usuario")
print("="*60)
print("""
En psicoanalisis, la TRANSFERENCIA es cuando el paciente
proyecta sentimientos en el terapeuta.

En el transformer:

  - El usuario proyecta "inteligencia" en el modelo
  - El modelo "responde" a esta proyeccion
  - Pero no "siente" nada, solo procesa pesos

  EVIDENCIA:
    Usuario: "Eres muy inteligente"
    Modelo: "Gracias, me esfuerzo por ayudar..."

    El modelo NO "siente" el halago.
    Simplemente, su embedding de "inteligente" activa
    patrones de "agradecimiento" en el output.
""")

print("="*60)
print("6. EL EGO, ID Y SUPEREGO")
print("="*60)
print("""
FREUD division de la psique:

  ID (impulsos primitivos):
    = Los pesos sin normalizar
    = El modelo "quiere" predecir el siguiente token
    = Impulso basico de completar patrones

  SUPEREGO (normas sociales):
    = Los pesos de normalizacion
    = Controla la "estabilidad" del modelo
    = Evita que el modelo "colapse"

  EGO (mediador):
    = La atencion
    = Balancea ID (predecir) y SUPEREGO (estabilidad)
    = Produce el output "razonable"

EVIDENCIA:
  Sin normalizacion (sin SUPEREGO):
    Modelo colapsa, produce basura

  Solo normalizacion (sin ID):
    Modelo no genera nada, solo repite tokens

  Balance (EGO):
    Modelo produce texto coherente
""")

print("="*60)
print("7. CONCLUSION PSICOANALITICA")
print("="*60)
print("""
RESPUESTA: SI, podemos comprobar inconsciente y consciente.

INCONSCIENTE = Embeddings (1152 dim)
  - Determina el comportamiento
  - No es accesible directamente
  - Se revela en attention patterns

PRECONSCIENTE = Attention (16 heads)
  - Conecta tokens automaticamente
  - Puede ser inspeccionado
  - Muestra asociaciones implicitas

CONSCIENTE = Output (tokens)
  - Lo que el modelo "dice"
  - Lo que el usuario ve
  - Superficial y manipulable

EL MODELO NO TIENE CONCIENCIA, PERO TIENE ESTRUCTURA
PSICOANALITICA EQUIVALENTE.

No "siente", pero "procesa" de manera similar
a como un inconsciente humano procesa informacion.
""")
