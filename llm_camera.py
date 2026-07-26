"""
Cámara del LLM: Ver el manifold desde un prompt
Dado un prompt, calcular su dirección y ver los conceptos cercanos
"""

import numpy as np
from gguf import GGUFReader
from pathlib import Path

class LLMCamera:
    """
    Cámara que mira el manifold desde la perspectiva de un prompt
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.reader = None
        self.embeddings = None
        self.vocab_size = 0
        self.embd_dim = 0
        
    def load(self):
        """Cargar modelo y extraer embeddings"""
        print(f"Cargando modelo: {self.model_path}")
        self.reader = GGUFReader(self.model_path)
        
        # Extraer token embeddings
        for tensor in self.reader.tensors:
            if 'token_embd' in tensor.name:
                self.embeddings = tensor.data
                self.vocab_size, self.embd_dim = self.embeddings.shape
                print(f"  Tokens: {self.vocab_size}, Dim: {self.embd_dim}")
                break
        
        return self
    
    def get_embedding(self, text):
        """
        Obtener embedding de un texto (usando tokens individuales)
        Nota: Esto es una simplificación - en producción usaríamos el tokenizer real
        """
        # Por ahora, promediar embeddings de caracteres/tokens
        # En producción, necesitaríamos el vocabulario real
        
        # Simular: buscar tokens que contengan el texto
        # Esto es un placeholder - necesitamos el vocabulario real
        return None
    
    def find_direction(self, concept_a_idx, concept_b_idx):
        """
        Calcular dirección entre dos conceptos
        """
        embd_a = self.embeddings[concept_a_idx]
        embd_b = self.embeddings[concept_b_idx]
        
        # Dirección = resta de vectores
        direction = embd_b - embd_a
        
        # Normalizar
        direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def find_similar_tokens(self, token_idx, top_k=20):
        """
        Encontrar tokens más similares a uno dado
        """
        target = self.embeddings[token_idx]
        
        # Calcular similitud con todos los tokens
        similarities = np.dot(self.embeddings, target)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(target)
        norms = np.where(norms == 0, 1, norms)
        similarities = similarities / norms
        
        # Obtener top-k (excluyendo el token mismo)
        similarities[token_idx] = -1
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(idx, similarities[idx]) for idx in top_indices]
    
    def project_along_direction(self, token_idx, direction, steps=10):
        """
        Proyectar un token a lo largo de una dirección
        """
        base = self.embeddings[token_idx]
        
        projections = []
        for step in range(-steps, steps + 1):
            # Punto a lo largo de la dirección
            point = base + step * 0.1 * direction
            
            # Encontrar token más cercano a este punto
            distances = np.linalg.norm(self.embeddings - point, axis=1)
            nearest_idx = np.argmin(distances)
            distance = distances[nearest_idx]
            
            projections.append({
                'step': step,
                'point': point,
                'nearest_token': nearest_idx,
                'distance': float(distance)
            })
        
        return projections
    
    def camera_view(self, focus_token_idx, radius=0.5, top_k=20):
        """
        Ver la "vista" desde un token específico
        """
        focus = self.embeddings[focus_token_idx]
        
        # Calcular distancias a todos los tokens
        distances = np.linalg.norm(self.embeddings - focus, axis=1)
        
        # Encontrar tokens dentro del radio
        nearby_mask = distances < radius
        nearby_indices = np.where(nearby_mask)[0]
        
        # Ordenar por distancia
        nearby_tokens = [(idx, distances[idx]) for idx in nearby_indices]
        nearby_tokens.sort(key=lambda x: x[1])
        
        return nearby_tokens[:top_k]
    
    def visualize_perspective(self, prompt_name, token_idx):
        """
        Visualizar la perspectiva de un prompt
        """
        print(f"\n{'='*60}")
        print(f"CÁMARA: '{prompt_name}' (token {token_idx})")
        print(f"{'='*60}")
        
        # 1. Encontrar tokens similares
        print(f"\n1. Tokens más similares:")
        similar = self.find_similar_tokens(token_idx, top_k=10)
        for idx, sim in similar:
            print(f"   Token {idx}: similitud = {sim:.4f}")
        
        # 2. Ver vecindario cercano
        print(f"\n2. Vecindario cercano (radio = 0.3):")
        nearby = self.camera_view(token_idx, radius=0.3, top_k=10)
        for idx, dist in nearby:
            print(f"   Token {idx}: distancia = {dist:.4f}")
        
        # 3. Dirección "hacia arriba" (conceptos más abstractos)
        print(f"\n3. Dirección hacia conceptos más abstractos:")
        if token_idx + 100 < self.vocab_size:
            direction = self.find_direction(token_idx, token_idx + 100)
            projections = self.project_along_direction(token_idx, direction, steps=5)
            for proj in projections:
                print(f"   Paso {proj['step']:+d}: token más cercano = {proj['nearest_token']}, "
                      f"distancia = {proj['distance']:.4f}")
        
        return similar, nearby


def main():
    """Función principal"""
    
    print("="*60)
    print("CÁMARA DEL LLM: VER EL MANIFOLD")
    print("="*60)
    
    # 1. Cargar modelo
    camera = LLMCamera("C:/tmp/tinyllama-1.1b.Q4_0.gguf")
    camera.load()
    
    # 2. Seleccionar prompts de ejemplo
    # Nota: Usamos índices de tokens arbitrarios
    # En producción, necesitaríamos el vocabulario real
    
    prompts = [
        ("token_100", 100),
        ("token_500", 500),
        ("token_1000", 1000),
        ("token_5000", 5000),
        ("token_10000", 10000),
    ]
    
    # 3. Analizar cada prompt
    results = {}
    for name, idx in prompts:
        if idx < camera.vocab_size:
            similar, nearby = camera.visualize_perspective(name, idx)
            results[name] = {
                'similar': similar,
                'nearby': nearby
            }
    
    # 4. Análisis de dirección entre conceptos
    print(f"\n{'='*60}")
    print("ANÁLISIS DE DIRECCIONES ENTRE CONCEPTOS")
    print(f"{'='*60}")
    
    # Tomar dos tokens y ver la dirección
    token_a = 100
    token_b = 200
    
    if token_a < camera.vocab_size and token_b < camera.vocab_size:
        direction = camera.find_direction(token_a, token_b)
        
        print(f"\nDirección de token {token_a} a token {token_b}:")
        print(f"  Magnitud: {np.linalg.norm(direction):.4f}")
        print(f"  Componentes top-5: {np.argsort(np.abs(direction))[-5:][::-1]}")
        
        # Proyectar a lo largo de esta dirección
        print(f"\nProyección a lo largo de esta dirección:")
        projections = camera.project_along_direction(token_a, direction, steps=5)
        for proj in projections:
            print(f"  Paso {proj['step']:+d}: token = {proj['nearest_token']}, "
                  f"dist = {proj['distance']:.4f}")
    
    # 5. Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    
    print(f"""
    Cámara configurada:
    - Tokens: {camera.vocab_size}
    - Dimensión: {camera.embd_dim}
    
    Análisis realizado:
    - {len(prompts)} prompts analizados
    - Similitud entre tokens calculada
    - Direcciones entre conceptos mapeadas
    
    Para usar con prompts reales:
    1. Obtener vocabulario del tokenizer
    2. Mapear texto a índices de tokens
    3. Ejecutar cámara con tokens reales
    """)
    
    return camera, results


if __name__ == "__main__":
    camera, results = main()
