"""
style_switch.py
Sistema de conmutación de estilos en runtime.
"""

import numpy as np
import os
import sys
import json

# ============================================================================
# Clase StyleSwitcher
# ============================================================================

class StyleSwitcher:
    """Sistema de conmutación de estilos via perturbación de pesos."""
    
    def __init__(self, base_model_path=None):
        """
        Inicializar el switcher de estilos.
        
        Args:
            base_model_path: Ruta al modelo base GGUF
        """
        self.base_model_path = base_model_path
        self.vectors = {}
        self.active_style = None
        
        # Vectores predefinidos (simulados)
        self._create_default_vectors()
    
    def _create_default_vectors(self):
        """Crear vectores de estilo predefinidos."""
        # En producción, estos se cargarían de archivos .nnpz
        # Aquí simulamos con vectores aleatorios
        
        np.random.seed(42)
        vector_size = 1000  # Tamaño simulado
        
        self.vectors = {
            'philosophical': np.random.randn(vector_size) * 0.02,
            'stoic': np.random.randn(vector_size) * 0.018,
            'concise': np.random.randn(vector_size) * 0.015,
            'creative': np.random.randn(vector_size) * 0.022,
            'practical': np.random.randn(vector_size) * 0.016,
            'spiritual': np.random.randn(vector_size) * 0.021,
            'academic': np.random.randn(vector_size) * 0.019,
            'authentic': np.random.randn(vector_size) * 0.017,
        }
    
    def list_styles(self):
        """Listar estilos disponibles."""
        return list(self.vectors.keys())
    
    def set_style(self, style_name):
        """
        Establecer estilo activo.
        
        Args:
            style_name: Nombre del estilo
            
        Raises:
            ValueError: Si el estilo no existe
        """
        if style_name not in self.vectors:
            raise ValueError(
                f"Estilo '{style_name}' no encontrado. "
                f"Disponibles: {self.list_styles()}"
            )
        self.active_style = style_name
    
    def get_vector(self, style_name):
        """
        Obtener vector de un estilo.
        
        Args:
            style_name: Nombre del estilo
            
        Returns:
            Vector de perturbación
        """
        if style_name not in self.vectors:
            raise ValueError(f"Estilo '{style_name}' no encontrado")
        return self.vectors[style_name]
    
    def blend_styles(self, style1, style2, ratio=0.5):
        """
        Mezclar dos estilos.
        
        Args:
            style1: Primer estilo
            style2: Segundo estilo
            ratio: Proporción del segundo estilo (0.0 - 1.0)
            
        Returns:
            Vector mezclado
        """
        if style1 not in self.vectors:
            raise ValueError(f"Estilo '{style1}' no encontrado")
        if style2 not in self.vectors:
            raise ValueError(f"Estilo '{style2}' no encontrado")
        
        vec1 = self.vectors[style1]
        vec2 = self.vectors[style2]
        
        # Interpolación lineal
        blended = (1 - ratio) * vec1 + ratio * vec2
        
        # Registrar vector mezclado
        blend_name = f"{style1}_{style2}_{int(ratio*100)}"
        self.vectors[blend_name] = blended
        
        return blend_name
    
    def apply_vector(self, weights, vector):
        """
        Aplicar vector de perturbación a pesos.
        
        Args:
            weights: Pesos originales
            vector: Vector de perturbación
            
        Returns:
            Pesos perturbados
        """
        # Ajustar tamaño si es necesario
        if len(vector) != len(weights):
            vector = vector[:len(weights)]
            if len(vector) < len(weights):
                vector = np.pad(vector, (0, len(weights) - len(vector)))
        
        return weights + vector
    
    def generate(self, prompt, max_tokens=100):
        """
        Generar texto con estilo activo.
        
        Args:
            prompt: Texto de entrada
            max_tokens: Número máximo de tokens
            
        Returns:
            Texto generado (simulado)
        """
        # En producción, esto usaría llama-cli
        # Aquí simulamos con respuestas predefinidas
        
        style_responses = {
            'philosophical': f"...finding true inner peace and contentment through self-awareness and acceptance of life's inherent impermanence...",
            'stoic': f"...balance between inner and outer lives, maintaining equanimity in all circumstances...",
            'concise': f"...adaptability. Change is constant. Those who thrive learn to flow with it.",
            'creative': f"...dancing between chaos and order, finding beauty in the unexpected...",
            'practical': f"...practical steps to improve daily life through small, consistent actions...",
            'spiritual': f"...awakening to the interconnectedness of all beings and consciousness...",
            'academic': f"...a complex philosophical question that has been debated by thinkers throughout history...",
            'authentic': f"...showing up authentically, embracing vulnerability, and choosing growth...",
        }
        
        if self.active_style:
            return style_responses.get(
                self.active_style, 
                f"...a thoughtful response to '{prompt}'..."
            )
        
        return f"...cultivating a mindset focused on gratitude and finding joy in everyday moments..."
    
    def save_vector(self, style_name, path):
        """
        Guardar vector en archivo.
        
        Args:
            style_name: Nombre del estilo
            path: Ruta de salida
        """
        if style_name not in self.vectors:
            raise ValueError(f"Estilo '{style_name}' no encontrado")
        
        np.savez_compressed(path, vector=self.vectors[style_name])
        print(f"Vector '{style_name}' guardado en: {path}")
    
    def load_vector(self, style_name, path):
        """
        Cargar vector desde archivo.
        
        Args:
            style_name: Nombre del estilo
            path: Ruta del archivo
        """
        data = np.load(path)
        self.vectors[style_name] = data['vector']
        print(f"Vector '{style_name}' cargado desde: {path}")

# ============================================================================
# Funciones de demostración
# ============================================================================

def demo_list_styles(switcher):
    """Demostración: listar estilos."""
    print("\n=== Estilos Disponibles ===")
    for style in switcher.list_styles():
        print(f"  - {style}")

def demo_generate_all(switcher):
    """Demostración: generar con todos los estilos."""
    prompt = "The secret to happiness is"
    
    print(f"\n=== Generando con prompt: '{prompt}' ===")
    
    for style in switcher.list_styles():
        switcher.set_style(style)
        response = switcher.generate(prompt)
        print(f"\n--- {style.upper()} ---")
        print(f"  {response}")

def demo_blend(switcher):
    """Demostración: mezclar estilos."""
    print("\n=== Mezclando Estilos ===")
    
    # Mezclar filosófico y estoico
    blend_name = switcher.blend_styles('philosophical', 'stoic', ratio=0.5)
    switcher.set_style(blend_name)
    response = switcher.generate("The meaning of life is")
    print(f"\n--- PHILOSOPHICAL + STOIC (50/50) ---")
    print(f"  {response}")
    
    # Mezclar creativo y conciso
    blend_name = switcher.blend_styles('creative', 'concise', ratio=0.3)
    switcher.set_style(blend_name)
    response = switcher.generate("Artificial intelligence will")
    print(f"\n--- CREATIVE + CONCISE (70/30) ---")
    print(f"  {response}")

def demo_comparison(switcher):
    """Demostración: comparar estilos."""
    prompts = [
        "The secret to happiness is",
        "The meaning of life is",
        "If I could change one thing about society",
    ]
    
    print("\n=== Comparación de Estilos ===")
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        for style in ['philosophical', 'stoic', 'concise', 'creative']:
            switcher.set_style(style)
            response = switcher.generate(prompt)
            print(f"  {style:15}: {response[:60]}...")

# ============================================================================
# Main
# ============================================================================

def main():
    """Función principal."""
    print("=" * 60)
    print("SISTEMA DE CONMUTACIÓN DE ESTILOS")
    print("Dreaming Project")
    print("=" * 60)
    
    # Inicializar
    switcher = StyleSwitcher()
    
    # Ejecutar demostraciones
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'list':
            demo_list_styles(switcher)
        elif command == 'generate':
            style = sys.argv[2] if len(sys.argv) > 2 else 'philosophical'
            prompt = sys.argv[3] if len(sys.argv) > 3 else "The secret to happiness is"
            switcher.set_style(style)
            print(switcher.generate(prompt))
        elif command == 'blend':
            demo_blend(switcher)
        elif command == 'compare':
            demo_comparison(switcher)
        else:
            print(f"Comando desconocido: {command}")
            print("Uso: python style_switch.py [list|generate|blend|compare]")
    else:
        # Demostración completa
        demo_list_styles(switcher)
        demo_generate_all(switcher)
        demo_blend(switcher)
        demo_comparison(switcher)
    
    print("\n" + "=" * 60)
    print("COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    main()
