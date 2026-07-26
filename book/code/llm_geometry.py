"""
llm_geometry.py
Visualización geométrica de la estructura interna de un LLM.
La analogía: un LLM es como un paisaje 3D con colinas (regiones coherentes)
y valles (regiones incoherentes). Cada perspectiva es una cámara diferente.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from matplotlib import cm

# ============================================================================
# Configuración de estilo
# ============================================================================

plt.rcParams['figure.facecolor'] = '#0a0a1a'
plt.rcParams['axes.facecolor'] = '#0a0a1a'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'

# ============================================================================
# Crear el "paisaje de coherencia"
# ============================================================================

def create_coherence_landscape():
    """
    Crear el paisaje de coherencia del LLM.
    
    La idea es que el LLM es un paisaje donde:
    - Las colinas altas = regiones de alta coherencia (respuestas buenas)
    - Los valles = regiones de baja coherencia (basura)
    - La "manifold de coherencia" es la superficie de las colinas
    """
    
    # Grid para el paisaje
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Crear "colinas" de coherencia
    # Cada colina representa una perspectiva diferente
    
    Z = np.zeros_like(X)
    
    # Colina central: Baseline (original)
    Z += 3.0 * np.exp(-((X-0)**2 + (Y-0)**2) / 2.0)
    
    # Colina filosófica
    Z += 2.5 * np.exp(-((X-2)**2 + (Y-1)**2) / 1.5)
    
    # Colina estoica
    Z += 2.3 * np.exp(-((X-1)**2 + (Y-2)**2) / 1.5)
    
    # Colina concisa
    Z += 2.0 * np.exp(-((X-2)**2 + (Y-2)**2) / 1.2)
    
    # Colina creativa
    Z += 2.2 * np.exp(-((X+1)**2 + (Y-2)**2) / 1.5)
    
    # Colina espiritual
    Z += 2.4 * np.exp(-((X-1)**2 + (Y+1)**2) / 1.5)
    
    # Colina práctica
    Z += 2.1 * np.exp(-((X+2)**2 + (Y+1)**2) / 1.2)
    
    # Añadir "ruido" para hacerlo más natural
    Z += 0.2 * np.sin(X * 2) * np.cos(Y * 2)
    
    return X, Y, Z

# ============================================================================
# Visualización 1: Paisaje de Coherencia con Cámaras
# ============================================================================

def plot_coherence_landscape():
    """Visualizar el paisaje de coherencia con las cámaras (perspectivas)."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Crear paisaje
    X, Y, Z = create_coherence_landscape()
    
    # Superficie del paisaje
    surf = ax.plot_surface(X, Y, Z, 
                          cmap=cm.coolwarm,
                          alpha=0.7,
                          edgecolor='none',
                          antialiased=True)
    
    # Posiciones de las "cámaras" (perspectivas)
    cameras = {
        'Baseline': (0, 0, 3.5),
        'Filosófica': (2, 1, 3.0),
        'Estoica': (1, 2, 2.8),
        'Concisa': (2, 2, 2.5),
        'Creativa': (-1, 2, 2.7),
        'Espiritual': (1, -1, 2.9),
        'Práctica': (-2, -1, 2.6),
    }
    
    # Colores para cada cámara
    colors = {
        'Baseline': '#FFD700',      # Dorado
        'Filosófica': '#9370DB',    # Violeta
        'Estoica': '#00CED1',       # Turquesa
        'Concisa': '#FF6347',       # Rojo
        'Creativa': '#32CD32',      # Verde
        'Espiritual': '#FF69B4',    # Rosa
        'Práctica': '#FFA500',      # Naranja
    }
    
    # Dibujar cámaras y sus conos de visión
    for name, (x, y, z) in cameras.items():
        # Punto de la cámara
        ax.scatter([x], [y], [z], 
                  c=colors[name], 
                  s=200, 
                  marker='^',
                  edgecolors='white',
                  linewidth=2,
                  zorder=10)
        
        # Línea de visión hacia abajo (hacia la colina)
        ax.plot([x, x], [y, y], [z, z-1], 
                c=colors[name], 
                linestyle='--', 
                alpha=0.6,
                linewidth=2)
        
        # Etiqueta
        ax.text(x, y, z + 0.3, name, 
                fontsize=9, 
                ha='center',
                color=colors[name],
                fontweight='bold')
    
    # Configuración
    ax.set_xlabel('\nDimensión X (semántica)', fontsize=11)
    ax.set_ylabel('\nDimensión Y (sintaxis)', fontsize=11)
    ax.set_zlabel('\nCoherencia', fontsize=11)
    ax.set_title('LLM como Paisaje Geométrico\nCada perspectiva es una cámara diferente', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Ángulo de vista
    ax.view_init(elev=25, azim=135)
    
    plt.tight_layout()
    return fig

# ============================================================================
# Visualización 2: Vista Superior (Mapa 2D)
# ============================================================================

def plot_top_view():
    """Vista superior del paisaje - como un mapa topográfico."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear paisaje
    X, Y, Z = create_coherence_landscape()
    
    # Mapa de contornos
    contour = ax.contourf(X, Y, Z, levels=20, cmap=cm.coolwarm, alpha=0.8)
    contour_lines = ax.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
    
    # Posiciones de las cámaras
    cameras = {
        'Baseline': (0, 0),
        'Filosófica': (2, 1),
        'Estoica': (1, 2),
        'Concisa': (2, 2),
        'Creativa': (-1, 2),
        'Espiritual': (1, -1),
        'Práctica': (-2, -1),
    }
    
    # Colores
    colors = {
        'Baseline': '#FFD700',
        'Filosófica': '#9370DB',
        'Estoica': '#00CED1',
        'Concisa': '#FF6347',
        'Creativa': '#32CD32',
        'Espiritual': '#FF69B4',
        'Práctica': '#FFA500',
    }
    
    # Dibujar cámaras con dirección de vista
    for name, (x, y) in cameras.items():
        # Punto de la cámara
        ax.scatter(x, y, c=colors[name], s=300, marker='^', 
                  edgecolors='white', linewidth=2, zorder=10)
        
        #cono de visión (triángulo)
        angle = np.arctan2(y, x)  # Ángulo hacia el centro
        cone_length = 0.8
        cone_width = 0.3
        
        # Puntos del cono
        x1 = x + cone_length * np.cos(angle)
        y1 = y + cone_length * np.sin(angle)
        x2 = x + cone_width * np.cos(angle + np.pi/2)
        y2 = y + cone_width * np.sin(angle + np.pi/2)
        x3 = x + cone_width * np.cos(angle - np.pi/2)
        y3 = y + cone_width * np.sin(angle - np.pi/2)
        
        # Dibujar cono
        triangle = plt.Polygon([[x1, y1], [x2, y2], [x3, y3]], 
                              alpha=0.3, color=colors[name])
        ax.add_patch(triangle)
        
        # Etiqueta
        ax.text(x, y + 0.4, name, fontsize=10, ha='center', 
                color=colors[name], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Configuración
    ax.set_xlabel('Dimensión X (semántica)', fontsize=12)
    ax.set_ylabel('Dimensión Y (sintaxis)', fontsize=12)
    ax.set_title('Mapa Topográfico del LLM\nLas colinas son regiones de alta coherencia', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Barra de colores
    cbar = plt.colorbar(contour, ax=ax, label='Nivel de Coherencia')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

# ============================================================================
# Visualización 3: Estructura Jerárquica (Capas del Transformer)
# ============================================================================

def plot_transformer_hierarchy():
    """Visualizar la jerarquía interna del Transformer."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Capas del Transformer
    layers = [
        ('Input\nTokens', '#FF6B6B', 0),
        ('Embedding\n(32000 × 2048)', '#4ECDC4', 1),
        ('Pos Encoding\n(RoPE)', '#45B7D1', 2),
    ]
    
    # Capas de Transformer (22 en TinyLlama, mostramos 5 representativas)
    transformer_layers = [
        ('Block 0\n(Sintaxis)', '#96CEB4', 3),
        ('Block 5\n(Sintaxis)', '#88D8B0', 4),
        ('Block 11\n(Semántica)', '#FFEAA7', 5),
        ('Block 17\n(Semántica)', '#DDA0DD', 6),
        ('Block 21\n(Abstracto)', '#98D8C8', 7),
    ]
    
    layers.extend(transformer_layers)
    layers.extend([
        ('Final Norm', '#F7DC6F', 8),
        ('LM Head\n(2048 × 32000)', '#BB8FCE', 9),
        ('Output\nLogits', '#FF6B6B', 10),
    ])
    
    # Dibujar capas
    for i, (name, color, y_pos) in enumerate(layers):
        # Rectángulo de la capa
        rect = mpatches.FancyBboxPatch((0.5, y_pos * 0.8), 2, 0.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='white',
                                        linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        
        # Texto de la capa
        ax.text(1.5, y_pos * 0.8 + 0.3, name, 
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='black')
        
        # Flecha hacia siguiente capa
        if i < len(layers) - 1:
            ax.annotate('', xy=(1.5, (y_pos + 0.8) * 0.8), 
                       xytext=(1.5, y_pos * 0.8 + 0.6),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Añadir "cámaras" (perspectivas) mirando diferentes capas
    camera_positions = [
        (3.5, 1.6, '👁️ Mirando\nEmbedding', '#9370DB'),
        (3.5, 4.0, '👁️ Mirando\nBlock 5', '#00CED1'),
        (3.5, 5.6, '👁️ Mirando\nBlock 11', '#FF6347'),
        (3.5, 7.2, '👁️ Mirando\nBlock 17', '#32CD32'),
    ]
    
    for x, y, label, color in camera_positions:
        # Icono de cámara
        ax.text(x, y, '📷', fontsize=20, ha='center', va='center')
        ax.text(x + 0.8, y, label, fontsize=8, ha='left', va='center',
                color=color, fontweight='bold')
        
        # Línea de visión
        ax.plot([x - 0.3, 2.7], [y, y], color=color, linestyle='--', 
                alpha=0.5, linewidth=1.5)
    
    # Configuración
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Estructura Jerárquica del Transformer\nCada perspectiva "mira" diferentes capas', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# ============================================================================
# Visualización 4: La Manifold de Coherencia (2D simplificado)
# ============================================================================

def plot_coherence_manifold():
    """Visualizar la manifold de coherencia como una superficie 2D."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear grid
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 5, 50)
    Theta, R = np.meshgrid(theta, r)
    
    # La manifold es como una "montaña" con crestas
    # Cada cresta es una región de alta coherencia
    Z = np.zeros_like(R)
    
    # Crestas de coherencia (regiones buenas)
    for angle, width, height in [(0, 0.5, 3), (np.pi/3, 0.4, 2.5), 
                                   (2*np.pi/3, 0.6, 2.8), (np.pi, 0.5, 3),
                                   (4*np.pi/3, 0.4, 2.6), (5*np.pi/3, 0.5, 2.9)]:
        Z += height * np.exp(-((Theta - angle) % (2*np.pi))**2 / (2*width**2))
    
    # La manifold decae hacia los bordes
    Z *= np.exp(-R/3)
    
    # Convertir a coordenadas cartesianas para mejor visualización
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # Superficie
    surf = ax.contourf(X, Y, Z, levels=30, cmap=cm.viridis, alpha=0.8)
    
    # Puntos de perspectivas
    perspectives = [
        (0, 0, 'Baseline\n(Centro)', '#FFD700', 300),
        (2, 0, 'Filosófica\n(Derecha)', '#9370DB', 200),
        (1, 1.7, 'Estoica\n(Arriba)', '#00CED1', 200),
        (-1, 1.7, 'Concisa\n(Izquierda)', '#FF6347', 200),
        (-2, 0, 'Creativa\n(Abajo)', '#32CD32', 200),
        (0, -2, 'Espiritual\n(Fondo)', '#FF69B4', 200),
    ]
    
    for x, y, name, color, size in perspectives:
        ax.scatter(x, y, c=color, s=size, marker='*', 
                  edgecolors='white', linewidth=2, zorder=10)
        ax.text(x, y + 0.4, name, fontsize=9, ha='center',
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Flechas mostrando "dirección de vista"
    for x, y, name, color, _ in perspectives:
        if x == 0 and y == 0:
            continue  # Skip baseline
        ax.annotate('', xy=(x*0.5, y*0.5), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))
    
    # Configuración
    ax.set_xlabel('Dimensión X', fontsize=12)
    ax.set_ylabel('Dimensión Y', fontsize=12)
    ax.set_title('Manifold de Coherencia\nLas crestas son regiones de alta coherencia\nCada perspectiva es un punto de vista diferente', 
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    
    # Barra de colores
    cbar = plt.colorbar(surf, ax=ax, label='Coherencia')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

# ============================================================================
# Generar todas las visualizaciones
# ============================================================================

def main():
    """Generar todas las visualizaciones."""
    print("Generando visualizaciones geométricas del LLM...")
    
    # 1. Paisaje 3D
    print("1. Paisaje de coherencia 3D...")
    fig1 = plot_coherence_landscape()
    fig1.savefig('figures/01_coherence_landscape_3d.png', dpi=150, 
                 bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig1)
    
    # 2. Mapa topográfico
    print("2. Mapa topográfico...")
    fig2 = plot_top_view()
    fig2.savefig('figures/02_topographic_map.png', dpi=150, 
                 bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig2)
    
    # 3. Jerarquía del Transformer
    print("3. Jerarquía del Transformer...")
    fig3 = plot_transformer_hierarchy()
    fig3.savefig('figures/03_transformer_hierarchy.png', dpi=150, 
                 bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig3)
    
    # 4. Manifold de coherencia
    print("4. Manifold de coherencia...")
    fig4 = plot_coherence_manifold()
    fig4.savefig('figures/04_coherence_manifold.png', dpi=150, 
                 bbox_inches='tight', facecolor='#0a0a1a')
    plt.close(fig4)
    
    print("\n✓ Visualizaciones generadas en figures/")
    print("  - 01_coherence_landscape_3d.png")
    print("  - 02_topographic_map.png")
    print("  - 03_transformer_hierarchy.png")
    print("  - 04_coherence_manifold.png")

if __name__ == "__main__":
    main()
