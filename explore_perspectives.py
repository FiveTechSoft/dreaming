"""
Explorador de Perspectivas: Mapear todas las direcciones del manifold
Desde un punto dado, explorar todas las tangentes posibles
"""

import numpy as np
from gguf import GGUFReader
from collections import defaultdict

class PerspectiveExplorer:
    """
    Explorar todas las perspectivas desde un punto en el manifold
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.reader = None
        self.embeddings = None
        self.vocab_size = 0
        self.embd_dim = 0
        
    def load(self):
        """Cargar modelo"""
        print(f"Cargando modelo: {self.model_path}")
        self.reader = GGUFReader(self.model_path)
        
        for tensor in self.reader.tensors:
            if 'token_embd' in tensor.name:
                self.embeddings = tensor.data.astype(np.float32)
                self.vocab_size, self.embd_dim = self.embeddings.shape
                print(f"  Tokens: {self.vocab_size}, Dim: {self.embd_dim}")
                break
        
        return self
    
    def normalize(self, vec):
        """Normalizar vector"""
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return vec / norm
    
    def get_tangent_vectors(self, n_samples=1000):
        """
        Generar vectores tangentes aleatorios
        Cada vector representa una dirección posible
        """
        print(f"\nGenerando {n_samples} vectores tangentes...")
        
        # Generar vectores aleatorios en la esfera unitaria
        tangent_vectors = np.random.randn(n_samples, self.embd_dim)
        
        # Normalizar para que estén en la esfera unitaria
        norms = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
        tangent_vectors = tangent_vectors / norms
        
        print(f"  Vectores generados: {tangent_vectors.shape}")
        print(f"  Magnitud promedio: {np.mean(norms):.4f}")
        
        return tangent_vectors
    
    def project_to_tangent_space(self, base_token_idx, target_token_idx):
        """
        Proyectar la diferencia entre dos tokens al espacio tangente
        """
        base = self.embeddings[base_token_idx]
        target = self.embeddings[target_token_idx]
        
        # Diferencia
        diff = target - base
        
        # Proyectar al espacio tangente (normalizar)
        tangent = self.normalize(diff)
        
        return tangent
    
    def find_similar_perspectives(self, tangent, token_indices, top_k=10):
        """
        Encontrar tokens que apuntan en la misma dirección que la tangente
        """
        similarities = []
        
        for idx in token_indices:
            if idx < self.vocab_size:
                # Obtener tangente del token
                token_tangent = self.project_to_tangent_space(0, idx)
                
                # Similitud coseno con la tangente dada
                sim = np.dot(tangent, token_tangent)
                similarities.append((idx, float(sim)))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def cluster_perspectives(self, tangent_vectors, n_clusters=20):
        """
        Agrupar tangentes en clusters (familias de perspectivas)
        """
        print(f"\nClustering {len(tangent_vectors)} perspectivas en {n_clusters} familias...")
        
        # K-means simplificado
        n_samples = len(tangent_vectors)
        
        # Inicializar centros
        center_indices = np.random.choice(n_samples, n_clusters, replace=False)
        centers = tangent_vectors[center_indices].copy()
        
        assignments = np.zeros(n_samples, dtype=int)
        
        for iteration in range(10):
            # Asignar cada tangente al centro más cercano
            for i in range(n_samples):
                distances = [np.linalg.norm(tangent_vectors[i] - centers[j]) 
                           for j in range(n_clusters)]
                assignments[i] = np.argmin(distances)
            
            # Actualizar centros
            for j in range(n_clusters):
                mask = assignments == j
                if mask.any():
                    centers[j] = tangent_vectors[mask].mean(axis=0)
                    centers[j] = self.normalize(centers[j])
            
            print(f"  Iteración {iteration + 1}/10")
        
        return assignments, centers
    
    def analyze_cluster_characteristics(self, tangent_vectors, assignments, centers):
        """
        Analizar qué caracteriza a cada cluster
        """
        characteristics = {}
        
        for cluster_id in range(len(centers)):
            mask = assignments == cluster_id
            cluster_vectors = tangent_vectors[mask]
            
            if len(cluster_vectors) > 0:
                # Calcular propiedades del cluster
                center = centers[cluster_id]
                
                # Varianza interna (qué tan disperso es)
                internal_variance = np.mean([np.linalg.norm(v - center) 
                                           for v in cluster_vectors])
                
                # Dirección dominante (qué dimensiones son más importantes)
                dominant_dims = np.argsort(np.abs(center))[-10:][::-1]
                
                characteristics[cluster_id] = {
                    'size': int(mask.sum()),
                    'internal_variance': float(internal_variance),
                    'dominant_dimensions': dominant_dims.tolist(),
                    'center_magnitude': float(np.linalg.norm(center))
                }
        
        return characteristics
    
    def map_perspectives_to_concepts(self, centers, concept_embeddings):
        """
        Mapear centros de cluster a conceptos conocidos
        """
        mappings = []
        
        for i, center in enumerate(centers):
            # Encontrar concepto más cercano
            similarities = []
            for concept_name, concept_emb in concept_embeddings.items():
                sim = np.dot(self.normalize(center), self.normalize(concept_emb))
                similarities.append((concept_name, float(sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            mappings.append({
                'cluster': i,
                'best_match': similarities[0][0],
                'similarity': similarities[0][1],
                'top_3': similarities[:3]
            })
        
        return mappings
    
    def explore_from_point(self, base_token_idx, n_directions=100):
        """
        Explorar todas las perspectivas desde un punto dado
        """
        print(f"\n{'='*60}")
        print(f"EXPLORANDO DESDE TOKEN {base_token_idx}")
        print(f"{'='*60}")
        
        base = self.embeddings[base_token_idx]
        base_normalized = self.normalize(base)
        
        print(f"\nPunto base:")
        print(f"  Token: {base_token_idx}")
        print(f"  Magnitud: {np.linalg.norm(base):.4f}")
        print(f"  Top-5 dimensiones: {np.argsort(np.abs(base))[-5:][::-1]}")
        
        # 1. Generar tangentes aleatorias
        tangent_vectors = self.get_tangent_vectors(n_directions)
        
        # 2. Clusterizar
        assignments, centers = self.cluster_perspectives(tangent_vectors, n_clusters=10)
        
        # 3. Analizar características
        characteristics = self.analyze_cluster_characteristics(
            tangent_vectors, assignments, centers
        )
        
        # 4. Mostrar resultados
        print(f"\n{'='*60}")
        print(f"FAMILIAS DE PERSPECTIVAS ENCONTRADAS")
        print(f"{'='*60}")
        
        for cluster_id, info in sorted(characteristics.items(), 
                                       key=lambda x: x[1]['size'], 
                                       reverse=True):
            print(f"\nFamilia {cluster_id}:")
            print(f"  Tamaño: {info['size']} perspectivas")
            print(f"  Varianza interna: {info['internal_variance']:.4f}")
            print(f"  Dimensiones dominantes: {info['dominant_dimensions'][:5]}")
        
        return tangent_vectors, assignments, centers, characteristics
    
    def visualize_perspective_space(self, tangent_vectors, assignments, centers):
        """
        Visualizar el espacio de perspectivas (para 2D/3D)
        """
        # Reducir a 2D usando PCA
        from numpy.linalg import svd
        
        # Centrar datos
        mean = tangent_vectors.mean(axis=0)
        centered = tangent_vectors - mean
        
        # SVD
        U, S, Vt = svd(centered, full_matrices=False)
        
        # Proyectar a 2D
        projected = centered @ Vt[:2].T
        
        return projected


def main():
    """Función principal"""
    
    print("="*70)
    print("EXPLORADOR DE PERSPECTIVAS")
    print("="*70)
    
    # 1. Cargar modelo
    explorer = PerspectiveExplorer("C:/tmp/tinyllama-1.1b.Q4_0.gguf")
    explorer.load()
    
    # 2. Explorar desde un punto base
    base_token = 100  # Token arbitrario
    tangent_vectors, assignments, centers, characteristics = explorer.explore_from_point(
        base_token, n_directions=500
    )
    
    # 3. Visualizar (reducir a 2D)
    print(f"\n{'='*60}")
    print(f"VISUALIZACIÓN DEL ESPACIO")
    print(f"{'='*60}")
    
    projected = explorer.visualize_perspective_space(tangent_vectors, assignments, centers)
    
    print(f"\nProyección 2D:")
    print(f"  Rango X: [{projected[:, 0].min():.2f}, {projected[:, 0].max():.2f}]")
    print(f"  Rango Y: [{projected[:, 1].min():.2f}, {projected[:, 1].max():.2f}]")
    
    # 4. Encontrar perspectivas más extremas
    print(f"\n{'='*60}")
    print(f"PERSPECTIVAS MÁS EXTREMAS")
    print(f"{'='*60}")
    
    # Tangente más larga (más diferente del base)
    norms = np.linalg.norm(tangent_vectors, axis=1)
    most_different_idx = np.argmax(norms)
    print(f"\nMás diferente del base:")
    print(f"  Índice: {most_different_idx}")
    print(f"  Magnitud: {norms[most_different_idx]:.4f}")
    
    # Tangente más corta (más similar al base)
    most_similar_idx = np.argmin(norms)
    print(f"\nMás similar al base:")
    print(f"  Índice: {most_similar_idx}")
    print(f"  Magnitud: {norms[most_similar_idx]:.4f}")
    
    # 5. Resumen
    print(f"\n{'='*60}")
    print(f"RESUMEN")
    print(f"{'='*60}")
    
    print(f"""
    Exploración completada:
    
    1. Punto base: token {base_token}
    2. Perspectivas exploradas: {len(tangent_vectors)}
    3. Familias encontradas: {len(characteristics)}
    4. Dimensión del espacio: {explorer.embd_dim}
    
    Conclusión geométrica:
    - El espacio tangente tiene {explorer.embd_dim} dimensiones
    - Cada dirección = una perspectiva
    - Total perspectivas = infinitas
    - Familias distinguibles = ~{len(characteristics)}
    
    Interpretación:
    - Cada familia de perspectivas = un "estilo" o "personalidad"
    - Las perspectivas dentro de una familia son similares
    - Las perspectivas entre familias son diferentes
    """)
    
    return explorer, tangent_vectors, assignments, centers


if __name__ == "__main__":
    explorer, tangent_vectors, assignments, centers = main()
