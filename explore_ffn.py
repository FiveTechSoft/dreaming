"""
EXPLORACIÓN DEL FFN: La Memoria del LLM
Vamos a encontrar qué números controlan qué conocimiento
"""

import numpy as np
from gguf import GGUFReader
import json
from pathlib import Path

class FFNExplorer:
    """
    Explorar la FFN para encontrar qué controla qué
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.tensors = {}
        self.n_layers = 22
        
    def load(self):
        """Cargar modelo"""
        print(f"Cargando modelo: {self.model_path}")
        reader = GGUFReader(self.model_path)
        
        for tensor in reader.tensors:
            self.tensors[tensor.name] = tensor.data.astype(np.float32)
        
        print(f"  Tensores cargados: {len(self.tensors)}")
        return self
    
    def get_ffn_weights(self, layer):
        """Obtener pesos FFN de una capa"""
        gate = self.tensors.get(f'blk.{layer}.ffn_gate.weight')
        up = self.tensors.get(f'blk.{layer}.ffn_up.weight')
        down = self.tensors.get(f'blk.{layer}.ffn_down.weight')
        return gate, up, down
    
    def perturb_single_neuron(self, layer, neuron_idx, factor=10.0):
        """
        Perturbar UNA sola neurona en la FFN
        """
        gate, up, down = self.get_ffn_weights(layer)
        
        if gate is None or up is None or down is None:
            return None
        
        # Guardar originales
        gate_orig = gate.copy()
        up_orig = up.copy()
        down_orig = down.copy()
        
        # Perturbar: hacer la neurona 10 veces más grande
        gate[neuron_idx, :] *= factor
        up[neuron_idx, :] *= factor
        
        # También perturbar la columna correspondiente en down
        if neuron_idx < down.shape[0]:
            down[:, neuron_idx] *= factor
        
        return {
            'gate': gate,
            'up': up,
            'down': down,
            'gate_orig': gate_orig,
            'up_orig': up_orig,
            'down_orig': down_orig
        }
    
    def restore_weights(self, layer, perturbed):
        """Restaurar pesos originales"""
        self.tensors[f'blk.{layer}.ffn_gate.weight'] = perturbed['gate_orig']
        self.tensors[f'blk.{layer}.ffn_up.weight'] = perturbed['up_orig']
        self.tensors[f'blk.{layer}.ffn_down.weight'] = perturbed['down_orig']
    
    def analyze_neuron_importance(self, layer, n_neurons=100):
        """
        Analizar qué neuronas son más importantes
        """
        gate, up, down = self.get_ffn_weights(layer)
        
        if gate is None:
            return []
        
        # Calcular importancia de cada neurona
        importance = []
        
        for i in range(min(n_neurons, gate.shape[0])):
            # Importancia = magnitud de los pesos
            gate_mag = np.linalg.norm(gate[i, :])
            up_mag = np.linalg.norm(up[i, :])
            
            if i < down.shape[1]:
                down_mag = np.linalg.norm(down[:, i])
            else:
                down_mag = 0
            
            total_importance = gate_mag + up_mag + down_mag
            
            importance.append({
                'neuron': i,
                'gate_mag': float(gate_mag),
                'up_mag': float(up_mag),
                'down_mag': float(down_mag),
                'total': float(total_importance)
            })
        
        # Ordenar por importancia
        importance.sort(key=lambda x: x['total'], reverse=True)
        
        return importance
    
    def find_cluster_neurons(self, layer, n_clusters=5):
        """
        Encontrar grupos de neuronas que trabajan juntas
        """
        gate, up, down = self.get_ffn_weights(layer)
        
        if gate is None:
            return []
        
        # Tomar las primeras 100 neuronas
        n_sample = min(100, gate.shape[0])
        gate_sample = gate[:n_sample, :]
        
        # K-means simple
        from numpy.linalg import norm
        
        # Inicializar centros aleatoriamente
        center_indices = np.random.choice(n_sample, n_clusters, replace=False)
        centers = gate_sample[center_indices].copy()
        
        assignments = np.zeros(n_sample, dtype=int)
        
        for _ in range(10):
            # Asignar
            for i in range(n_sample):
                distances = [norm(gate_sample[i] - centers[j]) for j in range(n_clusters)]
                assignments[i] = np.argmin(distances)
            
            # Actualizar centros
            for j in range(n_clusters):
                mask = assignments == j
                if mask.any():
                    centers[j] = gate_sample[mask].mean(axis=0)
        
        # Analizar cada cluster
        clusters = []
        for j in range(n_clusters):
            mask = assignments == j
            cluster_neurons = np.where(mask)[0]
            
            clusters.append({
                'id': j,
                'neurons': cluster_neurons.tolist(),
                'size': int(mask.sum()),
                'center_norm': float(norm(centers[j]))
            })
        
        return clusters
    
    def explore_all_layers(self, n_neurons_per_layer=50):
        """
        Explorar todas las capas
        """
        print(f"\n{'='*60}")
        print("EXPLORANDO FFN - TODAS LAS CAPAS")
        print(f"{'='*60}")
        
        all_results = {}
        
        for layer in range(self.n_layers):
            print(f"\nCapa {layer}:")
            
            # Analizar importancia
            importance = self.analyze_neuron_importance(layer, n_neurons_per_layer)
            
            # Encontrar clusters
            clusters = self.find_cluster_neurons(layer, n_clusters=5)
            
            all_results[layer] = {
                'top_neurons': importance[:10],
                'clusters': clusters
            }
            
            # Mostrar top 3 neuronas
            print(f"  Top 3 neuronas:")
            for neuron in importance[:3]:
                print(f"    Neurona {neuron['neuron']}: importancia = {neuron['total']:.2f}")
        
        return all_results
    
    def find_fundamental_directions(self, results):
        """
        Encontrar direcciones fundamentales (no mezclas)
        """
        print(f"\n{'='*60}")
        print("DIRECCIONES FUNDAMENTALES")
        print(f"{'='*60}")
        
        # Recopilar todas las top neuronas
        all_top_neurons = []
        
        for layer, data in results.items():
            for neuron in data['top_neurons'][:5]:
                all_top_neurons.append({
                    'layer': layer,
                    'neuron': neuron['neuron'],
                    'importance': neuron['total']
                })
        
        # Encontrar neuronas más importantes globales
        all_top_neurons.sort(key=lambda x: x['importance'], reverse=True)
        
        print(f"\nTop 10 neuronas más importantes (de todo el modelo):")
        for i, neuron in enumerate(all_top_neurons[:10]):
            print(f"  {i+1}. Capa {neuron['layer']}, Neurona {neuron['neuron']}: "
                  f"{neuron['importance']:.2f}")
        
        return all_top_neurons[:10]


def main():
    """Función principal"""
    
    print("="*60)
    print("EXPLORACIÓN DE FFN - LA MEMORIA DEL LLM")
    print("="*60)
    
    # 1. Cargar modelo
    explorer = FFNExplorer("C:/tmp/tinyllama-1.1b.Q4_0.gguf")
    explorer.load()
    
    # 2. Explorar todas las capas
    results = explorer.explore_all_layers(n_neurons_per_layer=50)
    
    # 3. Encontrar direcciones fundamentales
    fundamentals = explorer.find_fundamental_directions(results)
    
    # 4. Guardar resultados
    output_file = Path("C:/tmp/dreaming/ffn_analysis.json")
    
    # Convertir a formato serializable
    serializable = {}
    for layer, data in results.items():
        serializable[str(layer)] = {
            'top_neurons': data['top_neurons'],
            'clusters': data['clusters']
        }
    
    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResultados guardados en: {output_file}")
    
    # 5. Resumen
    print(f"\n{'='*60}")
    print("RESUMEN")
    print(f"{'='*60}")
    
    print(f"""
EXPLORACION DE FFN COMPLETADA

1. Capas analizadas: {len(results)}
2. Neuronas por capa: 50
3. Total neuronas analizadas: {len(results) * 50}
4. Direcciones fundamentales: {len(fundamentals)}

PROXIMOS PASOS:
1. Tomar las top 10 neuronas
2. Hacer cada una 10 veces mas grande
3. Preguntar: "Jesus de Nazareth"
4. Ver QUE cambia en la respuesta
5. Hacer el mapa: neurona -> tipo de conocimiento
    """)
    
    return explorer, results, fundamentals


if __name__ == "__main__":
    explorer, results, fundamentals = main()
