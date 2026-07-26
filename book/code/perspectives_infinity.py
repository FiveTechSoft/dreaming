"""
perspectives_infinity.py
Visualización de la naturaleza infinita de perspectivas y la diferencia
entre prompts y perturbaciones.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.rcParams['figure.facecolor'] = '#0a0a1a'
plt.rcParams['axes.facecolor'] = '#0a0a1a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'

# ============================================================================
# 1. La Esfera de Perspectivas (infinitas direcciones)
# ============================================================================

def plot_perspective_sphere():
    """
    La esfera de perspectivas: cada punto en la superficie es una perspectiva.
    Hay infinitos puntos, pero podemos agruparlos en "familias".
    """
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear esfera
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Superficie semitransparente
    ax.plot_surface(x, y, z, alpha=0.1, color='#4a5568')
    
    # "Familias" de perspectivas (regiones en la esfera)
    families = {
        'Filosóficas': {'center': (0, 0, 1), 'color': '#9370DB', 'points': 50},
        'Prácticas': {'center': (0, 0, -1), 'color': '#FFA500', 'points': 50},
        'Creativas': {'center': (1, 0, 0), 'color': '#32CD32', 'points': 50},
        'Analíticas': {'center': (-1, 0, 0), 'color': '#00CED1', 'points': 50},
        'Emocionales': {'center': (0, 1, 0), 'color': '#FF69B4', 'points': 50},
        'Técnicas': {'center': (0, -1, 0), 'color': '#FF6347', 'points': 50},
    }
    
    for family, data in families.items():
        cx, cy, cz = data['center']
        color = data['color']
        
        # Generar puntos alrededor del centro de la familia
        n_points = data['points']
        theta = np.random.uniform(0, 2*np.pi, n_points)
        phi = np.random.uniform(0, np.pi/4, n_points)  # Radio pequeño
        
        # Puntos en la esfera
        px = cx + 0.15 * np.sin(phi) * np.cos(theta)
        py = cy + 0.15 * np.sin(phi) * np.sin(theta)
        pz = cz + 0.15 * np.cos(phi)
        
        # Normalizar para que estén en la esfera
        norm = np.sqrt(px**2 + py**2 + pz**2)
        px, py, pz = px/norm, py/norm, pz/norm
        
        ax.scatter(px, py, pz, c=color, s=30, alpha=0.7, label=family)
        
        # Etiqueta de familia
        ax.text(cx*1.3, cy*1.3, cz*1.3, family, 
                fontsize=10, color=color, fontweight='bold',
                ha='center', va='center')
    
    # Centro (baseline)
    ax.scatter([0], [0], [0], c='#FFD700', s=200, marker='*', 
               edgecolors='white', linewidth=2, zorder=10, label='Baseline')
    
    ax.set_xlabel('Dimensión X', fontsize=11)
    ax.set_ylabel('Dimensión Y', fontsize=11)
    ax.set_zlabel('Dimensión Z', fontsize=11)
    ax.set_title('La Esfera de Perspectivas\nCada punto es una perspectiva diferente\nHay infinitos puntos, pero se agrupan en familias', 
                 fontsize=13, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 2. Prompt vs Perturbación: Dos formas de mover la cámara
# ============================================================================

def plot_prompt_vs_perturbation():
    """
    Comparación: Prompt mueve la cámara, Perturbación cambia el lente.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- PANEL 1: Prompt (mover cámara) ---
    ax1 = axes[0]
    
    # Mundo (pesos fijos)
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 3, 50)
    Theta, R = np.meshgrid(theta, r)
    
    # Paisaje de coherencia (fijo)
    Z = np.zeros_like(R)
    Z += 2.5 * np.exp(-((Theta - 0)**2) / 0.5) * np.exp(-R/2)
    Z += 2.0 * np.exp(-((Theta - np.pi/2)**2) / 0.5) * np.exp(-R/2)
    Z += 1.8 * np.exp(-((Theta - np.pi)**2) / 0.5) * np.exp(-R/2)
    Z += 2.2 * np.exp(-((Theta - 3*np.pi/2)**2) / 0.5) * np.exp(-R/2)
    
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    ax1.contourf(X, Y, Z, levels=20, cmap=cm.viridis, alpha=0.6)
    
    # Cámaras con diferentes prompts (mismo lente, diferente posición)
    prompts = [
        (0, 2.5, 'Prompt:\n"What is love?"', '#FF69B4', 0),
        (2.5, 0, 'Prompt:\n"How to code?"', '#32CD32', np.pi/2),
        (0, -2.5, 'Prompt:\n"Meaning of life?"', '#9370DB', np.pi),
        (-2.5, 0, 'Prompt:\n"Explain physics"', '#00CED1', 3*np.pi/2),
    ]
    
    for x, y, label, color, angle in prompts:
        # Cámara (triángulo)
        size = 0.3
        triangle_x = [x, x + size*np.cos(angle-0.3), x + size*np.cos(angle+0.3)]
        triangle_y = [y, y + size*np.sin(angle-0.3), y + size*np.sin(angle+0.3)]
        ax1.fill(triangle_x, triangle_y, color=color, alpha=0.8)
        
        # Línea de visión
        ax1.annotate('', xy=(0, 0), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.6))
        
        # Etiqueta
        ax1.text(x*1.3, y*1.3, label, fontsize=9, ha='center', 
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_aspect('equal')
    ax1.set_title('PROMPT: Mueve la Cámara\n(Mismo lente, diferente posición)', 
                  fontsize=13, fontweight='bold', color='#FFD700')
    ax1.set_xlabel('Dimensión X')
    ax1.set_ylabel('Dimensión Y')
    
    # --- PANEL 2: Perturbación (cambiar lente) ---
    ax2 = axes[1]
    
    # Mismo paisaje base
    ax2.contourf(X, Y, Z, levels=20, cmap=cm.viridis, alpha=0.6)
    
    # Múltiples cámaras en la misma posición, diferentes lentes
    camera_pos = (0, 2.5)
    
    perturbations = [
        ('Lente original\n(Baseline)', '#FFD700', 0),
        ('Lente Filosófica\n(perturbación)', '#9370DB', 0.5),
        ('Lente Práctica\n(perturbación)', '#FFA500', -0.5),
    ]
    
    for i, (label, color, offset) in enumerate(perturbations):
        x, y = camera_pos[0] + offset*0.5, camera_pos[1] + offset*0.3
        
        # Cámara
        ax2.scatter(x, y, c=color, s=150, marker='s', edgecolors='white', 
                   linewidth=2, zorder=10)
        
        # Cono de visión (diferente para cada lente)
        cone_angle = np.arctan2(-y, -x)
        cone_width = 0.3 + offset * 0.1  # Lente más ancho/narrow
        
        x1 = x + 0.8 * np.cos(cone_angle)
        y1 = y + 0.8 * np.sin(cone_angle)
        x2 = x + cone_width * np.cos(cone_angle + np.pi/2)
        y2 = y + cone_width * np.sin(cone_angle + np.pi/2)
        x3 = x + cone_width * np.cos(cone_angle - np.pi/2)
        y3 = y + cone_width * np.sin(cone_angle - np.pi/2)
        
        triangle = plt.Polygon([[x1, y1], [x2, y2], [x3, y3]], 
                              alpha=0.4, color=color)
        ax2.add_patch(triangle)
        
        # Etiqueta
        ax2.text(x + (i-1)*0.8, y + 0.8, label, fontsize=9, ha='center',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.set_title('PERTURBACIÓN: Cambia el Lente\n(Misma posición, diferente óptica)', 
                  fontsize=13, fontweight='bold', color='#FF6347')
    ax2.set_xlabel('Dimensión X')
    ax2.set_ylabel('Dimensión Y')
    
    plt.tight_layout()
    return fig

# ============================================================================
# 3. Dimensión de la Manifold de Coherencia
# ============================================================================

def plot_manifold_dimension():
    """
    Visualizar la dimensionalidad efectiva de la manifold de coherencia.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PANEL 1: Dimensionalidad ---
    ax1 = axes[0]
    
    dimensions = [1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 1100000000]
    labels = ['1D', '2D', '3D', '5D', '10D', '20D', '50D', '100D', '500D', '1000D', '1.1B']
    colors = ['#FF6B6B', '#FF8E8E', '#FFB4B4', '#FFD4D4', 
              '#4ECDC4', '#7EDDD6', '#AEF0EA', '#D4F7F4',
              '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Coherencia en cada dimensión
    coherence = [20, 35, 55, 70, 82, 88, 92, 94, 95, 95, 95]
    
    bars = ax1.bar(range(len(dimensions)), coherence, color=colors, edgecolor='white', linewidth=0.5)
    
    # Línea de coherencia mínima
    ax1.axhline(y=90, color='#FF6347', linestyle='--', alpha=0.7, label='Coherencia mínima (90%)')
    
    ax1.set_xticks(range(len(dimensions)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Coherencia (%)', fontsize=12)
    ax1.set_title('Coherencia vs Dimensionalidad\nLa manifold efectiva tiene ~1000 dimensiones', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 105)
    
    # --- PANEL 2: Infinitas perspectivas en 1000D ---
    ax2 = axes[1]
    
    # Simular "familias" de perspectivas en 2D (proyección)
    np.random.seed(42)
    
    families = {
        'Filosóficas': {'center': (2, 2), 'std': 0.5, 'color': '#9370DB', 'n': 30},
        'Prácticas': {'center': (-2, -2), 'std': 0.4, 'color': '#FFA500', 'n': 30},
        'Creativas': {'center': (2, -2), 'std': 0.6, 'color': '#32CD32', 'n': 30},
        'Analíticas': {'center': (-2, 2), 'std': 0.45, 'color': '#00CED1', 'n': 30},
    }
    
    for family, data in families.items():
        cx, cy = data['center']
        std = data['std']
        n = data['n']
        
        # Puntos alrededor del centro
        x = cx + np.random.randn(n) * std
        y = cy + np.random.randn(n) * std
        
        ax2.scatter(x, y, c=data['color'], s=50, alpha=0.7, label=family)
    
    # Centro
    ax2.scatter([0], [0], c='#FFD700', s=200, marker='*', 
               edgecolors='white', linewidth=2, zorder=10, label='Baseline')
    
    # Flechas mostrando infinitas direcciones
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        ax2.annotate('', xy=(2.5*np.cos(angle), 2.5*np.sin(angle)), 
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5, alpha=0.4))
    
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.set_title('Infinitas Perspectivas en 1000D\nCada dirección es una perspectiva diferente', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlabel('Dimensión X (proyección)')
    ax2.set_ylabel('Dimensión Y (proyección)')
    
    plt.tight_layout()
    return fig

# ============================================================================
# 4. Resumen Visual: Prompt vs Perturbación
# ============================================================================

def create_summary():
    """Resumen visual de los conceptos."""
    
    summary = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    ¿CUÁNTAS PERSPECTIVAS EXISTEN?                      ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  RESPUESTA CORTA: Infinitas, pero agrupadas en familias.               ║
    ║                                                                        ║
    ║  EXPLICACIÓN:                                                          ║
    ║                                                                        ║
    ║  • La manifold de coherencia tiene ~1000 dimensiones efectivas         ║
    ║  • En un espacio de 1000D, hay infinitas direcciones                   ║
    ║  • Pero las perspectivas se agrupan en "familias" (regiones)           ║
    ║  • Cada familia = un tipo de perspectiva (filosófica, práctica, etc.)  ║
    ║                                                                        ║
    ║  ┌─────────────────────────────────────────────────────────────────┐   ║
    ║  │  Esfera de 1000D:                                              │   ║
    ║  │  • Infinitos puntos (perspectivas)                             │   ║
    ║  │  • ~6-10 familias principales                                  │   ║
    ║  │  • Cada familia tiene sub-familias                             │   ║
    ║  │  • Total: infinitas perspectivas, finitas familias             │   ║
    ║  └─────────────────────────────────────────────────────────────────┘   ║
    ║                                                                        ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║              ¿PROMPT VS PERTURBACIÓN? ¿SON LO MISMO?                   ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  NO, son diferentes:                                                   ║
    ║                                                                        ║
    ║  PROMPT = Mover la cámara (posición)                                   ║
    ║  ─────────────────────────────────                                     ║
    ║  • Los pesos están FIJOS                                               ║
    ║  • El prompt selecciona qué parte del espacio "activar"                ║
    ║  • Es como mover la cámara en un videojuego                           ║
    ║  • Ej: "What is love?" activa regiones emocionales                    ║
    ║                                                                        ║
    ║  PERTURBACIÓN = Cambiar el lente (óptica)                              ║
    ║  ──────────────────────────────────────                                ║
    ║  • Los pesos CAMBIAN                                                   ║
    ║  • La perturbación modifica cómo se procesa TODO input                 ║
    ║  • Es como cambiar el lente de la cámara                               ║
    ║  • Ej: Perspectiva filosófica hace todo más reflexivo                 ║
    ║                                                                        ║
    ║  ┌─────────────────────────────────────────────────────────────────┐   ║
    ║  │                    VIDEOJUEGO                                  │   ║
    ║  │                                                                │   ║
    ║  │   Prompt:        Mover cámara    →  Ver zona diferente         │   ║
    ║  │   Perturbación:  Cambiar lente   →  Ver todo diferente         │   ║
    ║  │                                                                │   ║
    ║  └─────────────────────────────────────────────────────────────────┘   ║
    ║                                                                        ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                         EJEMPLO PRÁCTICO                                ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  Prompt: "The secret to happiness is"                                  ║
    ║                                                                        ║
    ║  Baseline:     "...cultivating a mindset focused on gratitude..."      ║
    ║  + Filosófico: "...finding true inner peace and contentment..."        ║
    ║  + Práctico:   "...practical steps to improve daily life..."           ║
    ║  + Creativo:   "...dancing between chaos and order..."                 ║
    ║                                                                        ║
    ║  El prompt es el MISMO, pero el LENTE (perturbación) cambia           ║
    ║  cómo se procesa.                                                      ║
    ║                                                                        ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                    COMBINACIÓN: PROMPT + PERTURBACIÓN                  ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                        ║
    ║  Puedes combinar ambos:                                                ║
    ║                                                                        ║
    ║  Prompt: "What is the meaning of life?"                               ║
    ║  Perturbación: Filosófica                                              ║
    ║  Resultado: Respuesta filosófica sobre el sentido de la vida           ║
    ║                                                                        ║
    ║  Prompt: "How do I fix a car?"                                         ║
    ║  Perturbación: Filosófica                                              ║
    ║  Resultado: Respuesta filosófica sobre arreglar carros                 ║
    ║              (¡incluso los carros se vuelven filosóficos!)             ║
    ║                                                                        ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    return summary

# ============================================================================
# Main
# ============================================================================

def main():
    print("Generando visualizaciones de perspectivas...")
    
    # 1. Esfera de perspectivas
    print("1. Esfera de perspectivas (infinitas direcciones)...")
    fig1 = plot_perspective_sphere()
    fig1.savefig('figures/05_perspective_sphere.png', dpi=150, 
                 bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig1)
    
    # 2. Prompt vs Perturbación
    print("2. Prompt vs Perturbación...")
    fig2 = plot_prompt_vs_perturbation()
    fig2.savefig('figures/06_prompt_vs_perturbation.png', dpi=150, 
                 bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig2)
    
    # 3. Dimensionalidad
    print("3. Dimensionalidad de la manifold...")
    fig3 = plot_manifold_dimension()
    fig3.savefig('figures/07_manifold_dimension.png', dpi=150, 
                 bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig3)
    
    # 4. Resumen
    print(create_summary())
    
    print("\n✓ Visualizaciones generadas en figures/")

if __name__ == "__main__":
    main()
