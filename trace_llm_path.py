"""
Ruta completa del LLM: Desde token hasta respuesta
Analizar el proceso de "Jesus de Nazareth" capa por capa
"""

import numpy as np
from gguf import GGUFReader

class LLMPathTracer:
    """
    Trazar la ruta completa del modelo capa por capa
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.reader = None
        self.tensors = {}
        self.metadata = {}
        
    def load(self):
        """Cargar modelo completo"""
        print(f"Cargando modelo: {self.model_path}")
        self.reader = GGUFReader(self.model_path)
        
        # Extraer todos los tensores organizados
        for tensor in self.reader.tensors:
            name = tensor.name
            self.tensors[name] = tensor.data
            
            # Organizar por tipo
            if 'token_embd' in name:
                print(f"  Token Embeddings: {tensor.data.shape}")
            elif 'blk.' in name:
                parts = name.split('.')
                if len(parts) >= 2:
                    block_num = parts[1]
                    layer_type = parts[2] if len(parts) > 2 else 'unknown'
                    if block_num not in self.metadata:
                        self.metadata[block_num] = {}
                    self.metadata[block_num][layer_type] = tensor.data.shape
        
        print(f"\n  Total bloques: {len(self.metadata)}")
        print(f"  Dimensión embeddings: {self.tensors['token_embd.weight'].shape[1]}")
        
        return self
    
    def get_token_embedding(self, token_idx):
        """Obtener embedding de un token"""
        embd = self.tensors['token_embd.weight']
        if token_idx < embd.shape[0]:
            return embd[token_idx]
        return None
    
    def get_attention_weights(self, block_num, head=None):
        """Obtener pesos de atención de un bloque"""
        q_key = f'blk.{block_num}.attn_q.weight'
        k_key = f'blk.{block_num}.attn_k.weight'
        v_key = f'blk.{block_num}.attn_v.weight'
        o_key = f'blk.{block_num}.attn_output.weight'
        
        weights = {}
        if q_key in self.tensors:
            weights['Q'] = self.tensors[q_key]
        if k_key in self.tensors:
            weights['K'] = self.tensors[k_key]
        if v_key in self.tensors:
            weights['V'] = self.tensors[v_key]
        if o_key in self.tensors:
            weights['O'] = self.tensors[o_key]
        
        return weights
    
    def get_ffn_weights(self, block_num):
        """Obtener pesos de la FFN (Feed-Forward Network)"""
        up_key = f'blk.{block_num}.ffn_up.weight'
        down_key = f'blk.{block_num}.ffn_down.weight'
        gate_key = f'blk.{block_num}.ffn_gate.weight'
        
        weights = {}
        if up_key in self.tensors:
            weights['up'] = self.tensors[up_key]
        if down_key in self.tensors:
            weights['down'] = self.tensors[down_key]
        if gate_key in self.tensors:
            weights['gate'] = self.tensors[gate_key]
        
        return weights
    
    def compute_attention(self, embedding, block_num):
        """Computar atención para un embedding"""
        attn_weights = self.get_attention_weights(block_num)
        
        if 'Q' in attn_weights and 'K' in attn_weights:
            # Normalize weights
            Q_w = attn_weights['Q'].astype(np.float32)
            K_w = attn_weights['K'].astype(np.float32)
            V_w = attn_weights['V'].astype(np.float32)
            O_w = attn_weights['O'].astype(np.float32) if 'O' in attn_weights else None
            
            # Normalize each weight matrix
            Q_w = Q_w / (np.linalg.norm(Q_w, axis=1, keepdims=True) + 1e-8)
            K_w = K_w / (np.linalg.norm(K_w, axis=1, keepdims=True) + 1e-8)
            V_w = V_w / (np.linalg.norm(V_w, axis=1, keepdims=True) + 1e-8)
            if O_w is not None:
                O_w = O_w / (np.linalg.norm(O_w, axis=1, keepdims=True) + 1e-8)
            
            Q = Q_w @ embedding  # (2048,)
            K = K_w @ embedding  # (256,)
            V = V_w @ embedding  # (256,)
            
            # GQA: Repeat K and V to match Q heads
            n_heads_q = 16
            n_heads_kv = 2
            head_dim = 128
            ratio = n_heads_q // n_heads_kv  # 8
            
            # Reshape K and V for each Q head
            K_reshaped = K.reshape(n_heads_kv, head_dim)
            V_reshaped = V.reshape(n_heads_kv, head_dim)
            
            # Repeat each KV head for corresponding Q heads
            K_expanded = np.repeat(K_reshaped, ratio, axis=0).reshape(-1)
            V_expanded = np.repeat(V_reshaped, ratio, axis=0).reshape(-1)
            
            # Attention score
            attn_score = np.dot(Q, K_expanded) / np.sqrt(head_dim)
            attn_probs = 1.0 / (1.0 + np.exp(-np.clip(attn_score, -10, 10)))
            
            # Output = weighted sum of V
            output = attn_probs * V_expanded
            
            # Output projection
            if O_w is not None:
                output = O_w.T @ output
            
            # Clip output to prevent overflow
            output = np.clip(output, -100, 100)
            
            return output, float(attn_probs)
        
        return None, None
    
    def compute_ffn(self, embedding, block_num):
        """Computar FFN para un embedding"""
        ffn_weights = self.get_ffn_weights(block_num)
        
        if 'up' in ffn_weights and 'gate' in ffn_weights:
            # Normalize weights
            gate_w = ffn_weights['gate'].astype(np.float32)
            up_w = ffn_weights['up'].astype(np.float32)
            down_w = ffn_weights['down'].astype(np.float32) if 'down' in ffn_weights else None
            
            # Normalize
            gate_w = gate_w / (np.linalg.norm(gate_w, axis=1, keepdims=True) + 1e-8)
            up_w = up_w / (np.linalg.norm(up_w, axis=1, keepdims=True) + 1e-8)
            if down_w is not None:
                down_w = down_w / (np.linalg.norm(down_w, axis=1, keepdims=True) + 1e-8)
            
            # Proyectar
            gate_proj = gate_w @ embedding
            up_proj = up_w @ embedding
            
            # Use only first 3168 elements
            gate = gate_proj[:3168]
            up = up_proj[:3168]
            
            # silu(gate) * up
            gate_clipped = np.clip(gate, -10, 10)
            gate_silu = gate_clipped * (1 / (1 + np.exp(-gate_clipped)))
            hidden = up * gate_silu
            
            # Clip hidden
            hidden = np.clip(hidden, -100, 100)
            
            # Down projection
            if down_w is not None:
                output = hidden @ down_w.T
                # Take first 1152 to match embedding dimension
                output = output[:1152]
                output = np.clip(output, -100, 100)
                return output, hidden
        
        return None, None
    
    def trace_path(self, token_idx, n_layers=None):
        """
        Trazar la ruta completa del token a través del modelo
        """
        if n_layers is None:
            n_layers = len(self.metadata)
        
        print(f"\n{'='*70}")
        print(f"RUTA COMPLETA: Token {token_idx}")
        print(f"{'='*70}")
        
        # 1. Token Embedding
        embedding = self.get_token_embedding(token_idx)
        if embedding is None:
            print("Error: Token no encontrado")
            return None
        
        # Normalizar embedding
        embedding = embedding.astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        print(f"\n[CAPA 0] TOKEN EMBEDDING")
        print(f"  Dimensión: {embedding.shape}")
        print(f"  Norma: {np.linalg.norm(embedding):.4f}")
        print(f"  Valor medio: {np.mean(embedding):.4f}")
        print(f"  Top-5 componentes: {np.argsort(np.abs(embedding))[-5:][::-1]}")
        
        path = {
            'token': token_idx,
            'embedding': embedding,
            'layers': []
        }
        
        current = embedding
        
        # 2. Recorrer capas
        for block_num in range(min(n_layers, 22)):
            block_str = str(block_num)
            
            if block_str not in self.metadata:
                continue
            
            print(f"\n[CAPA {block_num + 1}] BLOQUE TRANSFORMER")
            
            # 2.1 Atención
            attn_output, attn_score = self.compute_attention(current, block_num)
            
            if attn_output is not None:
                print(f"  ATENCIÓN:")
                print(f"    Q·K score: {attn_score:.4f}")
                print(f"    Output norm: {np.linalg.norm(attn_output):.4f}")
                print(f"    Cambio respecto a entrada: {np.linalg.norm(attn_output - current):.4f}")
            
            # 2.2 Residual connection
            if attn_output is not None:
                after_attn = current + attn_output
            else:
                after_attn = current
            
            # 2.3 FFN
            ffn_output, ffn_hidden = self.compute_ffn(after_attn, block_num)
            
            if ffn_output is not None:
                print(f"  FFN:")
                print(f"    Hidden dim: {ffn_hidden.shape if ffn_hidden is not None else 'N/A'}")
                print(f"    Output norm: {np.linalg.norm(ffn_output):.4f}")
                
                # Analizar qué "conceptos" activa la FFN
                top_activations = np.argsort(np.abs(ffn_hidden))[-5:][::-1]
                print(f"    Top-5 neuronas activas: {top_activations}")
            
            # 2.4 Residual connection
            if ffn_output is not None:
                current = after_attn + ffn_output
            else:
                current = after_attn
            
            print(f"  SALIDA:")
            print(f"    Norma final: {np.linalg.norm(current):.4f}")
            print(f"    Cambio total: {np.linalg.norm(current - embedding):.4f}")
            
            # Guardar en path
            path['layers'].append({
                'block': block_num,
                'attn_score': float(attn_score) if attn_score is not None else None,
                'attn_norm': float(np.linalg.norm(attn_output)) if attn_output is not None else None,
                'ffn_norm': float(np.linalg.norm(ffn_output)) if ffn_output is not None else None,
                'output_norm': float(np.linalg.norm(current)),
                'top_neurons': top_activations.tolist() if ffn_hidden is not None else []
            })
        
        path['final'] = current
        
        return path
    
    def analyze_concepts(self, path):
        """
        Analizar qué conceptos se activaron durante el procesamiento
        """
        print(f"\n{'='*70}")
        print("ANÁLISIS DE CONCEPTOS ACTIVADOS")
        print(f"{'='*70}")
        
        # Analizar patrones de activación
        all_top_neurons = []
        for layer_info in path['layers']:
            if layer_info['top_neurons']:
                all_top_neurons.extend(layer_info['top_neurons'])
        
        # Contar neuronas más frecuentes
        from collections import Counter
        neuron_counts = Counter(all_top_neurons)
        
        print("\nNeuronas más activas durante todo el procesamiento:")
        for neuron, count in neuron_counts.most_common(10):
            print(f"  Neurona {neuron}: activada {count} veces")
        
        # Analizar cambios por capa
        print("\nEvolución por capa:")
        for i, layer_info in enumerate(path['layers']):
            if layer_info['attn_score'] is not None:
                print(f"  Capa {i}: attn={layer_info['attn_score']:.3f}, "
                      f"output={layer_info['output_norm']:.3f}")
        
        return neuron_counts
    
    def summarize_path(self, path):
        """Resumir la ruta completa"""
        print(f"\n{'='*70}")
        print("RESUMEN DE LA RUTA")
        print(f"{'='*70}")
        
        print(f"\nToken inicial: {path['token']}")
        print(f"Embedding inicial norm: {np.linalg.norm(path['embedding']):.4f}")
        print(f"Embedding final norm: {np.linalg.norm(path['final']):.4f}")
        print(f"Cambio total: {np.linalg.norm(path['final'] - path['embedding']):.4f}")
        
        # Capas con mayor cambio
        if path['layers']:
            print("\nCapas con mayor impacto:")
            sorted_layers = sorted(
                [(i, l['output_norm']) for i, l in enumerate(path['layers']) if l['output_norm']],
                key=lambda x: x[1],
                reverse=True
            )
            for i, norm in sorted_layers[:5]:
                print(f"  Capa {i}: norma = {norm:.4f}")


def main():
    """Función principal"""
    
    print("="*70)
    print("TRAZADOR DE RUTA: JESUS DE NAZARETH")
    print("="*70)
    
    # 1. Cargar modelo
    tracer = LLMPathTracer("C:/tmp/tinyllama-1.1b.Q4_0.gguf")
    tracer.load()
    
    # 2. Token para "Jesus" (usando un token arbitrario)
    # En producción, necesitaríamos el vocabulario real
    token_idx = 100  # Placeholder
    
    print(f"\nAnalizando token {token_idx}...")
    
    # 3. Trazar ruta completa
    path = tracer.trace_path(token_idx, n_layers=10)
    
    if path:
        # 4. Analizar conceptos
        tracer.analyze_concepts(path)
        
        # 5. Resumen
        tracer.summarize_path(path)
    
    return tracer, path


if __name__ == "__main__":
    tracer, path = main()
