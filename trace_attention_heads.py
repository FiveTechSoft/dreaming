import os
import sys
import json
import math
import argparse
import numpy as np

def trace_attention_tinyllama(prompt, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", output_prefix="attention_trace"):
    print(f"==================================================")
    print(f"   Trazador de Cabezas de Atención de TinyLlama   ")
    print(f"==================================================")
    print(f"Cargando modelo: {model_id}...")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: PyTorch y HuggingFace Transformers son necesarios.")
        print("Instala con: pip install torch transformers")
        sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32
    )
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    seq_len = len(tokens)
    
    print(f"\nFrase analizada ({seq_len} tokens):")
    for i, tok in enumerate(tokens):
        print(f"  [{i:02d}] {repr(tok)}")
        
    print("\nEjecutando forward pass con capturador de atención (output_attentions=True)...")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # outputs.attentions es una tupla de 22 elementos (capas),
    # cada uno con forma (1, 32, seq_len, seq_len)
    attentions = outputs.attentions
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    
    print(f"Model Layers: {num_layers}, Attention Heads per Layer: {num_heads}")
    print(f"Total Attention Heads Analyzed: {num_layers * num_heads}")
    
    layer_head_analysis = []
    head_class_counts = {
        "Sink / BOS Head": 0,
        "Previous Token Head": 0,
        "Long-Range Context Head": 0,
        "Selective / Focused Head": 0,
        "Distributed / General Head": 0
    }
    
    # Estructura de almacenamiento
    all_matrices = []
    
    for l_idx in range(num_layers):
        layer_attn = attentions[l_idx][0].cpu().numpy() # (32, seq_len, seq_len)
        layer_heads_info = []
        
        for h_idx in range(num_heads):
            head_matrix = layer_attn[h_idx] # (seq_len, seq_len)
            
            # Métricas agregadas sobre las posiciones activas (i >= 1)
            bos_attns = []
            prev_attns = []
            long_attns = []
            entropies = []
            head_targets = []
            
            for i in range(seq_len):
                row = head_matrix[i, :i+1] # Causal mask
                # Normalizar si es necesario
                sum_r = np.sum(row)
                if sum_r > 0:
                    row = row / sum_r
                
                # BOS attention
                bos_attns.append(float(row[0]))
                
                # Prev token attention
                if i > 0:
                    prev_attns.append(float(row[i-1]))
                    
                # Long range attention (pos < i - 2)
                if i >= 3:
                    long_attns.append(float(np.sum(row[:i-2])))
                    
                # Entropia
                non_zero = row[row > 1e-8]
                ent = -np.sum(non_zero * np.log2(non_zero)) if len(non_zero) > 0 else 0.0
                entropies.append(float(ent))
                
                # Top attended token
                top_idx = int(np.argmax(row))
                head_targets.append({
                    "src_pos": i,
                    "src_token": tokens[i],
                    "target_pos": top_idx,
                    "target_token": tokens[top_idx],
                    "weight": float(row[top_idx])
                })
                
            avg_bos = float(np.mean(bos_attns))
            avg_prev = float(np.mean(prev_attns)) if prev_attns else 0.0
            avg_long = float(np.mean(long_attns)) if long_attns else 0.0
            avg_entropy = float(np.mean(entropies))
            
            # Clasificación heurística de la cabeza
            if avg_bos > 0.55:
                classification = "Sink / BOS Head"
            elif avg_prev > 0.45:
                classification = "Previous Token Head"
            elif avg_long > 0.35 and avg_entropy > 1.2:
                classification = "Long-Range Context Head"
            elif avg_entropy < 1.0:
                classification = "Selective / Focused Head"
            else:
                classification = "Distributed / General Head"
                
            head_class_counts[classification] += 1
            
            head_info = {
                "layer": l_idx,
                "head": h_idx,
                "classification": classification,
                "metrics": {
                    "avg_bos_attn": round(avg_bos, 4),
                    "avg_prev_attn": round(avg_prev, 4),
                    "avg_long_range_attn": round(avg_long, 4),
                    "avg_entropy": round(avg_entropy, 4)
                },
                "matrix": head_matrix.tolist(),
                "top_targets": head_targets
            }
            layer_heads_info.append(head_info)
            
        layer_head_analysis.append(layer_heads_info)
        
    print("\n==================================================")
    print("      RESUMEN DE CLASIFICACIÓN DE CABEZAS         ")
    print("==================================================")
    for cat, count in head_class_counts.items():
        pct = (count / (num_layers * num_heads)) * 100
        print(f"  - {cat:30s}: {count:3d} cabezas ({pct:5.1f}%)")
        
    # Guardar JSON completo
    json_path = f"{output_prefix}_results.json"
    result_data = {
        "prompt": prompt,
        "tokens": tokens,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "classification_summary": head_class_counts,
        "layers": layer_head_analysis
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\nDatos de trazado guardados en: {json_path}")
    
    # Generar Visualizador HTML
    generate_html_visualizer(result_data, f"{output_prefix}_vis.html")
    return result_data

def generate_html_visualizer(data, output_html_path):
    tokens = data["tokens"]
    num_layers = data["num_layers"]
    num_heads = data["num_heads"]
    
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyLlama - Trazador de Cabezas de Atención</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-color: #f8fafc;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        h1, h2, h3 {{ color: var(--accent-color); }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        .tokens-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
        }}
        .token-badge {{
            background: #334155;
            padding: 6px 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 14px;
            border: 1px solid var(--border-color);
        }}
        .grid-layout {{
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 20px;
        }}
        .controls-panel {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            height: fit-content;
        }}
        .main-panel {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}
        select, input {{
            width: 100%;
            padding: 10px;
            background: #0f172a;
            border: 1px solid var(--border-color);
            color: white;
            border-radius: 6px;
            margin-bottom: 15px;
        }}
        .heads-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .head-card {{
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .head-card:hover, .head-card.selected {{
            border-color: var(--accent-color);
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
            transform: translateY(-2px);
        }}
        .head-card .title {{
            font-weight: bold;
            color: var(--accent-color);
            margin-bottom: 4px;
        }}
        .badge {{
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-block;
            margin-top: 4px;
        }}
        .badge-sink {{ background: #991b1b; color: #fecaca; }}
        .badge-prev {{ background: #1e3a8a; color: #bfdbfe; }}
        .badge-long {{ background: #065f46; color: #a7f3d0; }}
        .badge-selective {{ background: #7c2d12; color: #fed7aa; }}
        .badge-general {{ background: #374151; color: #d1d5db; }}
        
        .heatmap-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-family: monospace;
            font-size: 13px;
        }}
        .heatmap-table th, .heatmap-table td {{
            border: 1px solid var(--border-color);
            padding: 6px;
            text-align: center;
        }}
        .heatmap-table th {{
            background: #0f172a;
            color: var(--accent-color);
        }}
        .target-list {{
            margin-top: 20px;
            background: #0f172a;
            border-radius: 8px;
            padding: 15px;
        }}
        .target-item {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #1e293b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-card">
            <h1>🔬 TinyLlama - Trazador Interactivo de Cabezas de Atención</h1>
            <p><strong>Prompt trazado:</strong> "{data['prompt']}"</p>
            <div class="tokens-bar">
                {"".join([f'<div class="token-badge"><span style="color:#94a3b8;font-size:10px;">#{i}</span> {tok}</div>' for i, tok in enumerate(tokens)])}
            </div>
        </div>

        <div class="grid-layout">
            <div class="controls-panel">
                <h3>Selección de Capa</h3>
                <select id="layerSelect" onchange="updateView()">
                    {"".join([f'<option value="{l}">Capa {l}</option>' for l in range(num_layers)])}
                </select>

                <h3>Resumen de la Capa</h3>
                <div id="layerSummary"></div>
            </div>

            <div class="main-panel">
                <h2 id="viewTitle">Capa 0 - Visión General de las 32 Cabezas</h2>
                <div id="headsContainer" class="heads-grid"></div>
                
                <div id="detailView" style="margin-top: 30px;">
                    <h3>Detalle de la Cabeza Seleccionada</h3>
                    <div id="headDetailContent">Selecciona una cabeza arriba para ver la matriz de atención detallada.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const data = {json.dumps(data, ensure_ascii=False)};
        let selectedHead = 0;

        function getBadgeClass(cls) {{
            if (cls.includes("Sink")) return "badge-sink";
            if (cls.includes("Previous")) return "badge-prev";
            if (cls.includes("Long-Range")) return "badge-long";
            if (cls.includes("Selective")) return "badge-selective";
            return "badge-general";
        }}

        function updateView() {{
            const layerIdx = parseInt(document.getElementById("layerSelect").value);
            const layerData = data.layers[layerIdx];
            
            document.getElementById("viewTitle").innerText = `Capa ${{layerIdx}} - 32 Cabezas de Atención`;
            
            // Render heads grid
            const headsContainer = document.getElementById("headsContainer");
            headsContainer.innerHTML = "";
            
            layerData.forEach((h, hIdx) => {{
                const card = document.createElement("div");
                card.className = `head-card ${{hIdx === selectedHead ? 'selected' : ''}}`;
                card.onclick = () => {{ selectedHead = hIdx; updateView(); }};
                card.innerHTML = `
                    <div class="title">Cabeza ${{hIdx}}</div>
                    <div>Entropía: ${{h.metrics.avg_entropy}}</div>
                    <div>BOS: ${{Math.round(h.metrics.avg_bos_attn * 100)}}%</div>
                    <div>Prev: ${{Math.round(h.metrics.avg_prev_attn * 100)}}%</div>
                    <span class="badge ${{getBadgeClass(h.classification)}}">${{h.classification.split(' ')[0]}}</span>
                `;
                headsContainer.appendChild(card);
            }});
            
            // Render selected head detail
            renderHeadDetail(layerIdx, selectedHead);
        }}

        function renderHeadDetail(layerIdx, headIdx) {{
            const headData = data.layers[layerIdx][headIdx];
            const detailContainer = document.getElementById("headDetailContent");
            const tokens = data.tokens;
            
            let tableHtml = `
                <div style="margin-bottom:15px;">
                    <strong>Capa ${{layerIdx}}, Cabeza ${{headIdx}}</strong> | 
                    Clasificación: <span class="badge ${{getBadgeClass(headData.classification)}}">${{headData.classification}}</span> | 
                    Entropía Media: ${{headData.metrics.avg_entropy}}
                </div>
                <h4>Matriz de Atención (Softmax)</h4>
                <div style="overflow-x:auto;">
                <table class="heatmap-table">
                    <tr>
                        <th>Origen / Destino</th>
                        ${{tokens.map((t, idx) => `<th>#${{idx}}<br>${{t}}</th>`).join('')}}
                    </tr>
            `;
            
            for (let i = 0; i < tokens.length; i++) {{
                tableHtml += `<tr><th>#${{i}} ${{tokens[i]}}</th>`;
                for (let j = 0; j < tokens.length; j++) {{
                    if (j > i) {{
                        tableHtml += `<td style="background:#0f172a;color:#334155;">-</td>`;
                    }} else {{
                        const val = headData.matrix[i][j];
                        const alpha = Math.min(1.0, val * 1.5);
                        const bg = `rgba(56, 189, 248, ${{alpha}})`;
                        const fg = alpha > 0.5 ? '#000' : '#fff';
                        tableHtml += `<td style="background:${{bg}};color:${{fg}};">${{val.toFixed(2)}}</td>`;
                    }}
                }}
                tableHtml += `</tr>`;
            }}
            
            tableHtml += `</table></div>`;
            
            tableHtml += `
                <div class="target-list">
                    <h4>Top Tokens Atendidos por Cada Posición</h4>
                    ${{headData.top_targets.map(t => `
                        <div class="target-item">
                            <span>Token <strong>"${{t.src_token}}"</strong> (#${{t.src_pos}}) busca a:</span>
                            <span style="color:var(--accent-color);">👉 <strong>"${{t.target_token}}"</strong> (#${{t.target_pos}}) (peso: ${{t.weight.toFixed(3)}})</span>
                        </div>
                    `).join('')}}
                </div>
            `;
            
            detailContainer.innerHTML = tableHtml;
        }}

        window.onload = updateView;
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Visualizador interactivo HTML generado en: {output_html_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trazador de Cabezas de Atención de TinyLlama")
    parser.add_argument("--prompt", type=str, default="El perro corrió hacia el jardín porque él tenía mucha hambre.", help="Prompt a analizar")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="ID o ruta del modelo TinyLlama")
    parser.add_argument("--out", type=str, default="attention_trace", help="Prefijo de archivos de salida")
    
    args = parser.parse_args()
    trace_attention_tinyllama(args.prompt, args.model, args.out)
