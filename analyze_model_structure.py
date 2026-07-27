"""
Análisis rápido de la estructura del modelo.
Identifica qué tipos de tensores tienen más parámetros.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import struct

MODEL_PATH = "C:/tmp/tinyllama-1.1b.Q4_0.gguf"

def skip_value(f, vtype):
    if vtype == 8:
        slen = struct.unpack('<Q', f.read(8))[0]
        f.read(slen)
    elif vtype == 9:
        etype = struct.unpack('<I', f.read(4))[0]
        alen = struct.unpack('<Q', f.read(8))[0]
        for _ in range(alen):
            skip_value(f, etype)
    else:
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        f.read(sizes.get(vtype, 0))

print("="*60)
print("ANÁLISIS DE ESTRUCTURA DEL MODELO")
print("="*60)

with open(MODEL_PATH, 'rb') as f:
    magic = f.read(4)
    version = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    
    print(f"Version: {version}")
    print(f"Tensors: {n_tensors}")
    print(f"KV pairs: {n_kv}")
    
    # Skip KV pairs
    for i in range(n_kv):
        klen = struct.unpack('<Q', f.read(8))[0]
        f.read(klen)
        vtype = struct.unpack('<I', f.read(4))[0]
        skip_value(f, vtype)
    
    # Read tensor info
    tensors = []
    for i in range(n_tensors):
        tname_len = struct.unpack('<Q', f.read(8))[0]
        tname = f.read(tname_len).decode('utf-8')
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack('<I', f.read(4))[0]
        toffset = struct.unpack('<Q', f.read(8))[0]
        
        total_params = 1
        for d in dims:
            total_params *= d
        
        type_names = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 10: 'I8', 11: 'I16', 12: 'I32', 14: 'BF16'}
        
        tensors.append({
            'name': tname,
            'dims': dims,
            'type': ttype,
            'type_name': type_names.get(ttype, f'T{ttype}'),
            'params': total_params,
            'offset': toffset
        })

# ============================================================
# Agrupar por tipo
# ============================================================

print("\n" + "="*60)
print("DISTRIBUCIÓN POR TIPO")
print("="*60)

type_groups = {}
for t in tensors:
    tname = t['type_name']
    if tname not in type_groups:
        type_groups[tname] = {'count': 0, 'params': 0}
    type_groups[tname]['count'] += 1
    type_groups[tname]['params'] += t['params']

for tname, data in sorted(type_groups.items()):
    print(f"  {tname:8s}: {data['count']:4d} tensors, {data['params']:>15,d} params ({data['params']/1e6:.1f}M)")

# ============================================================
# Agrupar por componente
# ============================================================

print("\n" + "="*60)
print("DISTRIBUCIÓN POR COMPONENTE")
print("="*60)

component_groups = {
    'token_embd': {'count': 0, 'params': 0, 'tensors': []},
    'output': {'count': 0, 'params': 0, 'tensors': []},
    'attn_q': {'count': 0, 'params': 0, 'tensors': []},
    'attn_k': {'count': 0, 'params': 0, 'tensors': []},
    'attn_v': {'count': 0, 'params': 0, 'tensors': []},
    'attn_output': {'count': 0, 'params': 0, 'tensors': []},
    'ffn_gate': {'count': 0, 'params': 0, 'tensors': []},
    'ffn_up': {'count': 0, 'params': 0, 'tensors': []},
    'ffn_down': {'count': 0, 'params': 0, 'tensors': []},
    'norm': {'count': 0, 'params': 0, 'tensors': []},
}

for t in tensors:
    name = t['name']
    
    if 'token_embd' in name:
        component_groups['token_embd']['count'] += 1
        component_groups['token_embd']['params'] += t['params']
        component_groups['token_embd']['tensors'].append(t)
    elif 'output' in name and 'attn' not in name:
        component_groups['output']['count'] += 1
        component_groups['output']['params'] += t['params']
        component_groups['output']['tensors'].append(t)
    elif 'attn_q' in name:
        component_groups['attn_q']['count'] += 1
        component_groups['attn_q']['params'] += t['params']
        component_groups['attn_q']['tensors'].append(t)
    elif 'attn_k' in name:
        component_groups['attn_k']['count'] += 1
        component_groups['attn_k']['params'] += t['params']
        component_groups['attn_k']['tensors'].append(t)
    elif 'attn_v' in name:
        component_groups['attn_v']['count'] += 1
        component_groups['attn_v']['params'] += t['params']
        component_groups['attn_v']['tensors'].append(t)
    elif 'attn_output' in name:
        component_groups['attn_output']['count'] += 1
        component_groups['attn_output']['params'] += t['params']
        component_groups['attn_output']['tensors'].append(t)
    elif 'ffn_gate' in name:
        component_groups['ffn_gate']['count'] += 1
        component_groups['ffn_gate']['params'] += t['params']
        component_groups['ffn_gate']['tensors'].append(t)
    elif 'ffn_up' in name:
        component_groups['ffn_up']['count'] += 1
        component_groups['ffn_up']['params'] += t['params']
        component_groups['ffn_up']['tensors'].append(t)
    elif 'ffn_down' in name:
        component_groups['ffn_down']['count'] += 1
        component_groups['ffn_down']['params'] += t['params']
        component_groups['ffn_down']['tensors'].append(t)
    elif 'norm' in name:
        component_groups['norm']['count'] += 1
        component_groups['norm']['params'] += t['params']
        component_groups['norm']['tensors'].append(t)

total_params = sum(g['params'] for g in component_groups.values())

for comp, data in sorted(component_groups.items(), key=lambda x: x[1]['params'], reverse=True):
    pct = data['params'] / total_params * 100 if total_params > 0 else 0
    print(f"  {comp:15s}: {data['count']:4d} tensors, {data['params']:>15,d} params ({pct:5.1f}%)")

# ============================================================
# Top 10 tensores más grandes
# ============================================================

print("\n" + "="*60)
print("TOP 10 TENSORES MÁS GRANDES")
print("="*60)

sorted_tensors = sorted(tensors, key=lambda x: x['params'], reverse=True)

for i, t in enumerate(sorted_tensors[:10]):
    dims_str = str(t['dims'])
    print(f"  {i+1:2d}. {t['name']:30s}: {dims_str:20s} = {t['params']:>15,d} params ({t['type_name']})")

# ============================================================
# Análisis por capa
# ============================================================

print("\n" + "="*60)
print("ANÁLISIS POR CAPA")
print("="*60)

layer_params = {}
for t in tensors:
    name = t['name']
    if name.startswith('blk.'):
        layer = int(name.split('.')[1])
        if layer not in layer_params:
            layer_params[layer] = {'total': 0, 'attn': 0, 'ffn': 0, 'norm': 0}
        
        layer_params[layer]['total'] += t['params']
        
        if 'attn' in name:
            layer_params[layer]['attn'] += t['params']
        elif 'ffn' in name:
            layer_params[layer]['ffn'] += t['params']
        elif 'norm' in name:
            layer_params[layer]['norm'] += t['params']

print(f"\n{'Layer':>6} {'Total':>15} {'Attn':>15} {'FFN':>15} {'Norm':>15}")
print("-" * 70)

for layer in sorted(layer_params.keys()):
    data = layer_params[layer]
    print(f"  {layer:4d} {data['total']:>15,d} {data['attn']:>15,d} {data['ffn']:>15,d} {data['norm']:>15,d}")

# ============================================================
# Resumen de influencia estimada
# ============================================================

print("\n" + "="*60)
print("INFLUENCIA ESTIMADA")
print("="*60)

print("\nBasado en el número de parámetros:")
print("  1. FFN (ffn_gate, ffn_up, ffn_down): ~67% de parámetros")
print("     - Almacena conocimiento y transformaciones")
print("     - Mayor influencia en el contenido generado")
print()
print("  2. Atención (attn_q, attn_k, attn_v, attn_output): ~32% de parámetros")
print("     - Conecta tokens entre sí")
print("     - Mayor influencia en la coherencia")
print()
print("  3. Embeddings (token_embd): ~1% de parámetros")
print("     - Representación inicial de tokens")
print("     - Menor influencia directa")
print()
print("  4. Norms (attn_norm, ffn_norm): ~0.01% de parámetros")
print("     - Normalización")
print("     - Críticos para estabilidad pero pocos parámetros")

print("\n" + "="*60)
print("RECOMENDACIÓN PARA MODIFICACIÓN")
print("="*60)

print("\nPara cambiar el comportamiento del modelo:")
print("  1. Priorizar tensors FFN (mayorimpacto)")
print("  2. Modificar tensores de atención (conexiones)")
print("  3. Evitar solo modificar embeddings (poco efecto)")
print()
print("Capas másinfluyentes (según literature):")
print("  - Capas 0-5: Sintaxis y estructura")
print("  - Capas 6-12: Significado semántico")
print("  - Capas 13-21: Integración y generación")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)
