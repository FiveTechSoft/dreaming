"""
Buscar perspectivas que NO sean mezclas
Encontrar direcciones fundamentales del espacio
"""

import numpy as np
from gguf import GGUFReader

class FundamentalPerspectives:
    """
    Encontrar las perspectivas fundamentales (no mezclas)
    Estas son las direcciones ORTOGONALES a las que ya conocemos
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.embeddings = None
        self.vocab_size = 0
        self.embd_dim = 0
        
    def load(self):
        """Cargar modelo"""
        print(f"Cargando modelo: {self.model_path}")
        reader = GGUFReader(self.model_path)
        
        for tensor in reader.tensors:
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
    
    def find_orthogonal_directions(self, known_directions, n_new=10):
        """
        Encontrar direcciones ortogonales a las ya conocidas
        Estas son las perspectivas que NO son mezclas
        """
        print(f"\nBuscando {n_new} direcciones ortogonales...")
        
        known = np.array(known_directions)
        n_known = len(known)
        
        # Gram-Schmidt: orthogonalizar
        orthogonal = []
        
        for i in range(n_new):
            # Generar vector aleatorio
            v = np.random.randn(self.embd_dim)
            
            # Proyectar y restar componentes conocidas
            for k in known:
                proj = np.dot(v, k) / np.dot(k, k)
                v = v - proj * k
            
            # También restar proyecciones de vectores ya encontrados
            for o in orthogonal:
                proj = np.dot(v, o) / np.dot(o, o)
                v = v - proj * o
            
            # Normalizar
            v = self.normalize(v)
            
            # Verificar que no es cero
            if np.linalg.norm(v) > 0.01:
                orthogonal.append(v)
                print(f"  Dirección {i+1}: norma = {np.linalg.norm(v):.4f}")
        
        return np.array(orthogonal)
    
    def find_maximally_different(self, base_point, known_directions, n_find=10):
        """
        Encontrar puntos MÁXIMAMENTE diferentes a los conocidos
        """
        print(f"\nBuscando {n_find} puntos máximamente diferentes...")
        
        # Para cada token, calcular cuán diferente es a todas las direcciones conocidas
        differences = []
        
        for idx in range(min(self.vocab_size, 5000)):  # Limitar para eficiencia
            token_emb = self.embeddings[idx]
            
            # Calcular distancia mínima a cualquier dirección conocida
            min_distance = float('inf')
            for known in known_directions:
                # Distancia angular (1 - coseno)
                cos_sim = np.dot(self.normalize(token_emb), self.normalize(known))
                distance = 1 - abs(cos_sim)
                min_distance = min(min_distance, distance)
            
            differences.append((idx, min_distance))
        
        # Ordenar por diferencia (mayor = más diferente)
        differences.sort(key=lambda x: x[1], reverse=True)
        
        return differences[:n_find]
    
    def find_novel_concepts(self, base_point, known_directions, threshold=0.3):
        """
        Encontrar conceptos que NO están cubiertos por las direcciones conocidas
        """
        print(f"\nBuscando conceptos nuevos (threshold={threshold})...")
        
        novel_tokens = []
        
        for idx in range(min(self.vocab_size, 5000)):
            token_emb = self.embeddings[idx]
            
            # Calcular cuánto de este token es "explicado" por las direcciones conocidas
            explained = np.zeros(self.embd_dim)
            for known in known_directions:
                proj = np.dot(token_emb, known) / np.dot(known, known)
                explained = explained + proj * known
            
            # Error de reconstrucción
            error = np.linalg.norm(token_emb - explained) / np.linalg.norm(token_emb)
            
            if error > threshold:
                novel_tokens.append((idx, error))
        
        # Ordenar por novedad
        novel_tokens.sort(key=lambda x: x[1], reverse=True)
        
        return novel_tokens
    
    def cluster_by_novelty(self, known_directions, n_clusters=5):
        """
        Agrupar tokens por su nivel de novedad
        """
        print(f"\nAgrupando tokens por novedad...")
        
        # Calcular novedad para cada token
        novelty_scores = []
        
        for idx in range(min(self.vocab_size, 5000)):
            token_emb = self.embeddings[idx]
            
            # Proyección sobre direcciones conocidas
            projection = np.zeros(self.embd_dim)
            for known in known_directions:
                proj = np.dot(token_emb, known) / np.dot(known, known)
                projection = projection + proj * known
            
            # Componente novedad (perpendicular a lo conocido)
            novelty = token_emb - projection
            novelty_norm = np.linalg.norm(novelty)
            
            novelty_scores.append((idx, novelty_norm))
        
        # Clustering simple por percentiles
        scores = [s for _, s in novelty_scores]
        percentiles = np.percentile(scores, np.linspace(0, 100, n_clusters + 1))
        
        clusters = {i: [] for i in range(n_clusters)}
        
        for idx, score in novelty_scores:
            for i in range(n_clusters):
                if percentiles[i] <= score <= percentiles[i + 1]:
                    clusters[i].append(idx)
                    break
        
        return clusters, percentiles
    
    def analyze_fundamental_directions(self, orthogonal_dirs):
        """
        Analizar las direcciones fundamentales encontradas
        """
        print(f"\n{'='*60}")
        print("ANÁLISIS DE DIRECCIONES FUNDAMENTALES")
        print(f"{'='*60}")
        
        for i, direction in enumerate(orthogonal_dirs):
            print(f"\nDirección Fundamental {i+1}:")
            print(f"  Norma: {np.linalg.norm(direction):.4f}")
            
            # Encontrar tokens más cercanos a esta dirección
            similarities = []
            for idx in range(min(self.vocab_size, 1000)):
                token_emb = self.embeddings[idx]
                sim = np.dot(self.normalize(token_emb), self.normalize(direction))
                similarities.append((idx, float(sim)))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Tokens más representativos:")
            for idx, sim in similarities[:5]:
                print(f"    Token {idx}: similitud = {sim:.4f}")
        
        return orthogonal_dirs


def main():
    """Función principal"""
    
    print("="*70)
    print("BÚSQUEDA DE PERSPECTIVAS FUNDAMENTALES")
    print("="*70)
    
    # 1. Cargar modelo
    finder = FundamentalPerspectives("C:/tmp/tinyllama-1.1b.Q4_0.gguf")
    finder.load()
    
    # 2. Definir direcciones conocidas (nuestras 10 familias)
    # En producción, usaríamos los centros reales de los clusters
    known_directions = [
        np.random.randn(finder.embd_dim) for _ in range(10)
    ]
    known_directions = [finder.normalize(d) for d in known_directions]
    
    print(f"\nDirecciones conocidas: {len(known_directions)}")
    
    # 3. Encontrar direcciones ortogonales (no mezclas)
    orthogonal = finder.find_orthogonal_directions(known_directions, n_new=5)
    
    # 4. Encontrar puntos máximamente diferentes
    most_different = finder.find_maximally_different(
        np.zeros(finder.embd_dim), known_directions, n_find=5
    )
    
    print(f"\nPuntos más diferentes a lo conocido:")
    for idx, diff in most_different:
        print(f"  Token {idx}: diferencia = {diff:.4f}")
    
    # 5. Encontrar conceptos nuevos
    novel = finder.find_novel_concepts(np.zeros(finder.embd_dim), known_directions)
    
    print(f"\nConceptos más nuevos:")
    for idx, novelty in novel[:10]:
        print(f"  Token {idx}: novedad = {novelty:.4f}")
    
    # 6. Clustering por novedad
    clusters, percentiles = finder.cluster_by_novelty(known_directions)
    
    print(f"\nClusters por novedad:")
    for i, indices in clusters.items():
        print(f"  Cluster {i} ({percentiles[i]:.2f}-{percentiles[i+1]:.2f}): {len(indices)} tokens")
    
    # 7. Analizar direcciones fundamentales
    finder.analyze_fundamental_directions(orthogonal)
    
    # 8. Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    
    print(f"""
PERSPECTIVAS FUNDAMENTALES ENCONTRADAS:

1. Direcciones ortogonales: {len(orthogonal)}
   Estas NO son mezclas de las familias conocidas
   Son direcciones completamente nuevas

2. Tokens mas diferentes: {len(most_different)}
   Estos tokens representan conceptos alejados de lo conocido

3. Conceptos nuevos: {len(novel)}
   Estos tokens no estan "explicados" por las direcciones conocidas

4. Clusters de novedad: {len(clusters)}
   Tokens agrupados por cuan nuevos son

CONCLUSION:
-----------
Las perspectivas fundamentales son las direcciones
que NO pueden formarse combinando las 10 familias conocidas.

Son como los "colores primarios" del espacio de significado.
    """)
    
    return finder, orthogonal, most_different, novel


if __name__ == "__main__":
    finder, orthogonal, most_different, novel = main()
