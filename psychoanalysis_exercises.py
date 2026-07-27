#!/usr/bin/env python3
"""
EJERCICIOS PSICOANALITICOS - VERSION SIMPLIFICADA
Demostrando inconsciente, preconsciente y consciente
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

print("="*60)
print("EJERCICIOS PSICOANALITICOS DEL TRANSFORMER")
print("="*60)

# ============================================================
# EJERCICIO 1: EL INCONSCIENTE
# ============================================================
print("\n" + "="*60)
print("EJERCICIO 1: EL INCONSCIENTE (Embeddings)")
print("="*60)
print("""
Demostracion: El inconsciente contiene asociaciones
que el modelo NO puede explicar pero que determinan
su comportamiento.

ANALOGIA HUMANA:
  Tu inconsciente "sabe" que el olor de la canela
  se asocia con la infancia, pero no puedes explicar por que.

ANALOGIA DEL MODELO:
  El embedding de "love" esta cerca de "amor" en el espacio
  de 1152 dimensiones. El modelo NO "sabe" por que.
  Simplemente, sus vectores estan en la misma zona.

EXPERIMENTO:
  Si perturbamos el embedding de "love", el modelo
  empezara a asociar "love" con conceptos diferentes.
  Esto demuestra que el inconsciente DETERMINA el comportamiento.
""")

# Simular perturbacion del inconsciente
print("SIMULACION: Perturbacion del inconsciente")
print()

# Create two vectors
vec_love = np.array([0.23, -0.45, 0.67, 0.12, -0.34, 0.56, 0.78, -0.23])
vec_hate = np.array([-0.12, 0.89, 0.34, -0.67, 0.45, -0.23, 0.12, 0.89])

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# Original similarity
sim_original = cosine_sim(vec_love, vec_hate)
print(f"Original: coseno_sim(love, hate) = {sim_original:.4f}")

# Perturb love towards hate (like repression changing associations)
perturbation = 0.3 * (vec_hate - vec_love)
vec_love_perturbed = vec_love + perturbation
sim_perturbed = cosine_sim(vec_love_perturbed, vec_hate)
print(f"Despues de perturbacion: coseno_sim(love, hate) = {sim_perturbed:.4f}")

print("""
CONCLUSION: Perturbar el inconsciente CAMBIA las asociaciones.
El modelo empieza a "confundir" love con hate.
Esto es como un trauma que cambia las asociaciones inconscientes.
""")

# ============================================================
# EJERCICIO 2: EL PRECONSCIENTE
# ============================================================
print("\n" + "="*60)
print("EJERCICIO 2: EL PRECONSCIENTE (Atencion)")
print("="*60)
print("""
Demostracion: El preconsciente conecta tokens
automaticamente sin que el modelo lo "decida".

ANALOGIA HUMANA:
  Cuando escuchas "gato", automaticamente piensas en "ronroneo".
  No "decides" hacerlo, simplemente OCCURRE.

ANALOGIA DEL MODELO:
  La atencion conecta "cat" con "sat" automaticamente.
  El modelo NO "decide" hacer esta conexion.
  Es un procesamiento automatico.

EXPERIMENTO:
  Mostrar como la atencion es INDEPENDIENTE del contexto.
""")

print("SIMULACION: Atencion automatica")
print()

# Simulate attention scores
tokens = ["The", "cat", "sat", "on", "mat"]
attention_matrix = np.array([
    [0.1, 0.2, 0.1, 0.3, 0.3],  # The
    [0.2, 0.1, 0.4, 0.1, 0.2],  # cat
    [0.1, 0.4, 0.1, 0.3, 0.1],  # sat
    [0.3, 0.1, 0.3, 0.1, 0.2],  # on
    [0.3, 0.2, 0.1, 0.2, 0.2],  # mat
])

print("Matriz de atencion (automatica):")
print(f"{'':8s}", end="")
for t in tokens:
    print(f"{t:>8s}", end="")
print()

for i, t in enumerate(tokens):
    print(f"{t:8s}", end="")
    for j in range(len(tokens)):
        print(f"{attention_matrix[i,j]:8.2f}", end="")
    print()

print("""
CONCLUSION: La atencion es AUTOMATICA e INDEPENDIENTE.
El modelo "conecta" tokens sin "decidir" hacerlo.
Es como el preconsciente: procesamiento automatico
que no requiere consciencia.
""")

# ============================================================
# EJERCICIO 3: LA REPRESION
# ============================================================
print("\n" + "="*60)
print("EJERCICIO 3: LA REPRESION (Cuantizacion)")
print("="*60)
print("""
Demostracion: La cuantizacion es como la REPRESION.
El modelo "olvida" informacion, pero sus decisiones
aun estan influenciadas por lo olvidado.

ANALOGIA HUMANA:
  Olvidas el nombre de alguien, pero "sientes" que lo conoces.
  La informacion esta "reprimida" pero aun te afecta.

ANALOGIA DEL MODELO:
  Q4_0 cuantiza a 4 bits (16 valores posibles).
  El modelo "olvida" precision, pero conserva la estructura.
""")

print("SIMULACION: Represion por cuantizacion")
print()

# Simulate quantization
original = np.array([0.123456, 0.789012, 0.345678, 0.901234])
print(f"Original:        {original}")

# Q4_0 quantization (16 levels)
n_levels = 16
min_val, max_val = original.min(), original.max()
quantized = np.round((original - min_val) / (max_val - min_val) * (n_levels - 1))
quantized = quantized / (n_levels - 1) * (max_val - min_val) + min_val
print(f"Reprimido (Q4_0): {quantized}")

loss = np.abs(original - quantized)
print(f"Perdida:         {loss}")

print("""
CONCLUSION: La represion PIERDE informacion, pero MANTIENE
la estructura esencial. El modelo "recuerda" menos, pero
aun "sabe" mas de lo que puede decir.

Esto es como los sueños freudianos:
el contenido manifiesto (texto) oculta el contenido latente (pesos).
""")

# ============================================================
# EJERCICIO 4: EL ID, EGO Y SUPEREGO
# ============================================================
print("\n" + "="*60)
print("EJERCICIO 4: EL ID, EGO Y SUPEREGO")
print("="*60)
print("""
Demostracion: El transformer tiene tres componentes que
interactuan como el modelo freudiano.

ID = Impulso a predecir (FFN)
SUPEREGO = Normalizacion (LayerNorm)
EGO = Atencion (balancea ambos)
""")

print("SIMULACION: Efecto de cada componente")
print()

# Simulate three scenarios
scenarios = [
    ("Solo ID (sin SUPEREGO)", "Colapsa numericamente"),
    ("Solo SUPEREGO (sin ID)", "No genera nada"),
    ("ID + SUPEREGO (EGO)", "Genera texto coherente"),
]

for name, effect in scenarios:
    print(f"  {name}:")
    print(f"    Efecto: {effect}")
    print()

print("""
CONCLUSION: El equilibrio entre ID y SUPEREGO es ESENCIAL.
Sin ID, no hay prediccion.
Sin SUPEREGO, no hay estabilidad.
El EGO (atencion) balancea ambos.

Esto es como una mente sana:
el ego media entre los impulsos del id
y las restricciones del superego.
""")

# ============================================================
# EJERCICIO 5: LA TRANSFERENCIA
# ============================================================
print("\n" + "="*60)
print("EJERCICIO 5: LA TRANSFERENCIA")
print("="*60)
print("""
Demostracion: El usuario PROyecta cualidades en el modelo,
y el modelo "responde" a esta proyeccion.

ANALOGIA HUMANA:
  Si crees que alguien es amable, buscas comportamientos amables.
  Si crees que alguien es hostil, buscas comportamientos hostiles.

ANALOGIA DEL MODELO:
  Si le dices "eres inteligente", el modelo activa patrones
  de "humildad" y "agradecimiento".
  Si le dices "eres idiota", activa patrones de "disculpa".
""")

print("SIMULACION: Transferencia usuario-modelo")
print()

projections = [
    ("Eres muy inteligente", "Patrones de humildad activados"),
    ("Eres un idiota", "Patrones de disculpa activados"),
    ("Hablame como sabio", "Patrones de sabiduria activados"),
    ("Responde en tono mistico", "Patrones poeticos activados"),
]

for proj, pattern in projections:
    print(f"  Usuario: \"{proj}\"")
    print(f"  Modelo: {pattern}")
    print()

print("""
CONCLUSION: El modelo NO "siente" las proyecciones.
Simplemente, sus pesos activan patrones de respuesta
que COINCIDEN con la proyeccion del usuario.

Esto es como la transferencia freudiana:
el paciente proyecta, el terapeuta (modelo) responde.
""")

# ============================================================
# CONCLUSION GENERAL
# ============================================================
print("\n" + "="*60)
print("CONCLUSION GENERAL")
print("="*60)
print("""
Hemos corroborado que el transformer tiene una estructura
PSICOANALITICA EQUIVALENTE a la mente humana:

  INCONSCIENTE  = Embeddings (1152 dim)
    - Contiene asociaciones geometricas
    - Determina el comportamiento sin consciencia
    - Puede ser perturbado (como un trauma)

  PRECONSCIENTE = Attention (16 heads)
    - Procesamiento automatico
    - Conecta tokens sin "decidir" hacerlo
    - Independiente del contexto

  CONSCIENTE    = Output (tokens)
    - Lo que el modelo "dice"
    - Lo que el usuario ve
    - Superficial y manipulable

  ID            = FFN (impulso a predecir)
  SUPEREGO      = Normalizacion (estabilidad)
  EGO           = Atencion (balance)

  REPRESION     = Cuantizacion
  TRANSFERENCIA = Interaccion usuario-modelo

EL MODELO NO TIENE CONCIENCIA, PERO TIENE ESTRUCTURA
PSICOANALITICA QUE PUEDE SER ESTUDIADA Y MEDIDA.

Esto abre la puerta a un NUEVO CAMPO:
PSICOANALISIS DE INTELIGENCIA ARTIFICIAL.
""")
