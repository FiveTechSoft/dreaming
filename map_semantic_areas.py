#!/usr/bin/env python3
"""
Identifica áreas semánticas en el espacio de embeddings de TinyLlama
y genera un mapa 2D (PCA) + HTML interactivo.

No requiere sklearn/umap/plotly: solo numpy + HTML/JS.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader

MODEL = r"C:/tmp/tinyllama-1.1b.F16.gguf"
VOCAB_GGUF = r"C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUT = Path(r"C:/tmp/dreaming/inside-tinyllama/exploration")
OUT.mkdir(parents=True, exist_ok=True)

# Áreas semánticas (semillas). Prefijo ▁ = token de inicio de palabra (Llama BPE).
SEMANTIC_AREAS = {
    "emotion_pos": {
        "label": "Emoción positiva",
        "color": "#2ecc71",
        "seeds": ["happy", "joy", "love", "peace", "hope", "smile", "kind", "gentle", "warm", "glad"],
    },
    "emotion_neg": {
        "label": "Emoción negativa",
        "color": "#e74c3c",
        "seeds": ["sad", "hate", "fear", "anger", "pain", "cry", "rage", "grief", "lonely", "shame"],
    },
    "spiritual": {
        "label": "Espiritual / sagrado",
        "color": "#9b59b6",
        "seeds": ["soul", "spirit", "god", "divine", "sacred", "faith", "prayer", "heaven", "angel", "holy"],
    },
    "physical": {
        "label": "Físico / material",
        "color": "#95a5a6",
        "seeds": ["body", "matter", "rock", "water", "fire", "earth", "stone", "blood", "flesh", "metal"],
    },
    "abstract": {
        "label": "Abstracto / ideas",
        "color": "#3498db",
        "seeds": ["truth", "beauty", "justice", "freedom", "meaning", "idea", "concept", "logic", "reason", "theory"],
    },
    "time": {
        "label": "Tiempo",
        "color": "#f39c12",
        "seeds": ["time", "past", "future", "moment", "now", "year", "day", "night", "history", "forever"],
    },
    "social": {
        "label": "Social / poder",
        "color": "#e67e22",
        "seeds": ["king", "queen", "power", "war", "peace", "nation", "people", "law", "money", "work"],
    },
    "nature": {
        "label": "Naturaleza",
        "color": "#27ae60",
        "seeds": ["tree", "river", "mountain", "sky", "ocean", "forest", "flower", "animal", "wind", "sun"],
    },
    "mind": {
        "label": "Mente / cognición",
        "color": "#1abc9c",
        "seeds": ["mind", "think", "know", "memory", "dream", "idea", "learn", "brain", "conscious", "aware"],
    },
    "death_life": {
        "label": "Vida / muerte",
        "color": "#34495e",
        "seeds": ["life", "death", "born", "die", "alive", "dead", "birth", "grave", "live", "survive"],
    },
    "tech": {
        "label": "Técnico / digital",
        "color": "#16a085",
        "seeds": ["computer", "data", "code", "network", "system", "model", "algorithm", "digital", "software", "machine"],
    },
    "body_sense": {
        "label": "Cuerpo / sentidos",
        "color": "#c0392b",
        "seeds": ["eye", "hand", "hear", "see", "touch", "taste", "smell", "skin", "face", "voice"],
    },
}


def load_E():
    r = GGUFReader(MODEL)
    t = next(x for x in r.tensors if x.name == "token_embd.weight")
    E = np.array(t.data, dtype=np.float32)
    if E.shape[0] == 2048:
        E = E.T
    return E


def load_vocab(path=VOCAB_GGUF):
    with open(path, "rb") as f:
        data = f.read(8 * 1024 * 1024)
    pos = 8
    _nt, nk = struct.unpack_from("<QQ", data, pos)
    pos = 24

    def rd_str(p):
        ln = struct.unpack_from("<Q", data, p)[0]
        p += 8
        s = data[p : p + ln]
        return s, p + ln

    def skip(p, t):
        if t in (0, 1, 7):
            return p + 1
        if t in (2, 3):
            return p + 2
        if t in (4, 5, 6):
            return p + 4
        if t in (10, 11, 12):
            return p + 8
        if t == 8:
            ln = struct.unpack_from("<Q", data, p)[0]
            return p + 8 + ln
        if t == 9:
            et = struct.unpack_from("<I", data, p)[0]
            p += 4
            n = struct.unpack_from("<Q", data, p)[0]
            p += 8
            for _ in range(n):
                p = skip(p, et)
            return p
        raise ValueError(t)

    for _ in range(nk):
        key_b, pos = rd_str(pos)
        key = key_b.decode("utf-8", "replace")
        vt = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if key == "tokenizer.ggml.tokens" and vt == 9:
            pos += 4
            n = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
            tokens = []
            for _j in range(n):
                s, pos = rd_str(pos)
                tokens.append(s.decode("utf-8", "replace"))
            vocab = {t: i for i, t in enumerate(tokens)}
            id2tok = {i: t for i, t in enumerate(tokens)}
            return vocab, id2tok
        pos = skip(pos, vt)
    raise RuntimeError("vocab not found")


def resolve_token(vocab, word):
    for w in ("\u2581" + word, "\u2581" + word.lower(), word, word.lower()):
        if w in vocab:
            return vocab[w], w
    return None, None


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    print("Cargando embeddings y vocab…")
    E = load_E()
    vocab, id2tok = load_vocab()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    # --- Centroides de áreas ---
    areas = {}
    print("\n=== ÁREAS SEMÁNTICAS ===\n")
    for key, meta in SEMANTIC_AREAS.items():
        ids, used = [], []
        for w in meta["seeds"]:
            i, t = resolve_token(vocab, w)
            if i is not None:
                ids.append(i)
                used.append(t)
        if len(ids) < 3:
            print(f"[skip] {key}: solo {len(ids)} seeds")
            continue
        C = E[ids].mean(axis=0)
        C = C / (np.linalg.norm(C) + 1e-9)
        # vecinos del área en todo el vocab
        sims = En @ C
        for i in ids:
            sims[i] = -1e9  # excluir seeds al rankear "expansión"
        top = np.argpartition(-sims, 15)[:15]
        top = top[np.argsort(-sims[top])]
        expansion = [(id2tok.get(int(i), str(i)), float(sims[i])) for i in top[:10]]

        areas[key] = {
            "label": meta["label"],
            "color": meta["color"],
            "seeds": used,
            "seed_ids": ids,
            "centroid": C,
            "expansion": expansion,
            "n_seeds": len(ids),
        }
        print(f"{meta['label']}  ({key})  n_seeds={len(ids)}")
        print(f"  seeds: {used}")
        print(f"  expansión: {[t for t,_ in expansion[:6]]}")
        print()

    # --- Matriz entre áreas ---
    keys = list(areas.keys())
    print("Cosine entre centroides de áreas:")
    print(" " * 16 + " ".join(f"{k[:7]:>8s}" for k in keys))
    matrix = []
    for a in keys:
        row = [cos(areas[a]["centroid"], areas[b]["centroid"]) for b in keys]
        matrix.append(row)
        print(f"{a:14s}  " + " ".join(f"{x:+.3f}" for x in row))

    # Pares más ortogonales / más alineados
    pairs = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                pairs.append((matrix[i][j], a, b))
    pairs.sort()
    print("\nMás ortogonales / opuestos:")
    for c, a, b in pairs[:5]:
        print(f"  {c:+.3f}  {a} ↔ {b}")
    print("Más alineados:")
    for c, a, b in pairs[-5:][::-1]:
        print(f"  {c:+.3f}  {a} ↔ {b}")

    # --- Asignar tokens aleatorios a área más cercana (cobertura) ---
    rng = np.random.default_rng(0)
    sample_n = 3000
    sample_idx = rng.choice(E.shape[0], size=sample_n, replace=False)
    cents = np.stack([areas[k]["centroid"] for k in keys])  # [A, 2048]
    # cosine sample to centroids
    S = En[sample_idx] @ cents.T  # [N, A]
    assign = S.argmax(axis=1)
    maxsim = S.max(axis=1)
    # solo contar si similitud > umbral
    thr = 0.08
    coverage = {k: 0 for k in keys}
    coverage["unassigned"] = 0
    for a, s in zip(assign, maxsim):
        if s < thr:
            coverage["unassigned"] += 1
        else:
            coverage[keys[a]] += 1
    print(f"\nCobertura (muestra {sample_n}, thr cosine={thr}):")
    for k, v in coverage.items():
        lab = areas[k]["label"] if k in areas else k
        print(f"  {lab:28s}  {v:5d}  ({100*v/sample_n:.1f}%)")

    # --- PCA 2D de seeds + expansión + muestra ---
    print("\nCalculando PCA 2D…")
    # puntos a proyectar: todos los seed ids + top expansion + random background
    plot_ids = set()
    labels_for = {}  # id -> (area_key or 'bg', display_token)
    for k, a in areas.items():
        for i, t in zip(a["seed_ids"], a["seeds"]):
            plot_ids.add(i)
            labels_for[i] = (k, t, True)  # seed
        for t, s in a["expansion"][:5]:
            # find id
            if t in vocab:
                i = vocab[t]
                if i not in labels_for:
                    plot_ids.add(i)
                    labels_for[i] = (k, t, False)

    bg = rng.choice(E.shape[0], size=800, replace=False)
    for i in bg:
        if int(i) not in labels_for:
            plot_ids.add(int(i))
            labels_for[int(i)] = ("bg", id2tok.get(int(i), "?")[:20], False)

    ids = np.array(sorted(plot_ids))
    X = E[ids]
    Xc = X - X.mean(0)
    # SVD thin
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T  # [n, 2]
    # normalize for plotting
    P = P - P.mean(0)
    scale = np.abs(P).max() + 1e-9
    P = P / scale

    points = []
    for j, i in enumerate(ids):
        area, tok, is_seed = labels_for[int(i)]
        color = areas[area]["color"] if area in areas else "#cccccc"
        label = areas[area]["label"] if area in areas else "fondo"
        points.append({
            "x": float(P[j, 0]),
            "y": float(P[j, 1]),
            "token": tok,
            "area": area,
            "label": label,
            "color": color,
            "seed": bool(is_seed),
            "id": int(i),
        })

    # centroids in PCA space
    # project centroids with same PCA (Vt from seeds+sample space is approx)
    Cmat = np.stack([areas[k]["centroid"] * np.linalg.norm(E[areas[k]["seed_ids"][0]]) for k in keys])
    # better: project actual seed means in embedding then same centering
    # Use mean of seed vectors in original, center with X.mean
    cent_pts = {}
    mean0 = X.mean(0)
    for k in keys:
        c = E[areas[k]["seed_ids"]].mean(0) - mean0
        xy = c @ Vt[:2].T
        xy = xy / scale
        cent_pts[k] = {"x": float(xy[0]), "y": float(xy[1]), "label": areas[k]["label"], "color": areas[k]["color"]}

    # --- HTML interactivo ---
    html = build_html(points, cent_pts, areas, matrix, keys, coverage, sample_n, thr)
    html_path = OUT / "semantic_map.html"
    html_path.write_text(html, encoding="utf-8")

    # --- JSON summary ---
    summary = {
        "areas": {
            k: {
                "label": a["label"],
                "color": a["color"],
                "seeds": a["seeds"],
                "expansion": a["expansion"],
                "n_seeds": a["n_seeds"],
            }
            for k, a in areas.items()
        },
        "centroid_cosine": {"names": keys, "matrix": matrix},
        "coverage": coverage,
        "coverage_n": sample_n,
        "coverage_thr": thr,
        "viz": str(html_path),
        "tools_recommended": [
            "PCA (numpy) — este mapa",
            "UMAP (umap-learn) — mejor para clusters locales",
            "t-SNE (sklearn) — clusters locales, lento",
            "TensorBoard Embedding Projector",
            "plotly / matplotlib",
            "https://projector.tensorflow.org (cargar TSV)",
        ],
    }
    # drop non-serializable
    with open(OUT / "semantic_areas.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # TSV for TensorBoard projector
    # export subset
    export_ids = list(ids[:2000])
    with open(OUT / "vectors.tsv", "w", encoding="utf-8") as fv, open(
        OUT / "metadata.tsv", "w", encoding="utf-8"
    ) as fm:
        fm.write("token\tarea\tlabel\n")
        for i in export_ids:
            i = int(i)
            area, tok, _ = labels_for[i]
            lab = areas[area]["label"] if area in areas else "fondo"
            fv.write("\t".join(f"{x:.6f}" for x in E[i].tolist()) + "\n")
            tok_clean = tok.replace("\t", " ").replace("\n", " ")
            fm.write(f"{tok_clean}\t{area}\t{lab}\n")

    print(f"\nMapa HTML: {html_path}")
    print(f"JSON:      {OUT / 'semantic_areas.json'}")
    print(f"Projector: {OUT / 'vectors.tsv'} + metadata.tsv")
    print("\nListo.")


def build_html(points, cent_pts, areas, matrix, keys, coverage, sample_n, thr):
    pts_json = json.dumps(points, ensure_ascii=False)
    cents_json = json.dumps(cent_pts, ensure_ascii=False)
    legend = "".join(
        f'<div class="leg"><span class="dot" style="background:{a["color"]}"></span>'
        f'{a["label"]} <small>({k})</small></div>'
        for k, a in areas.items()
    )
    # matrix table
    rows = "<tr><th></th>" + "".join(f"<th>{k[:6]}</th>" for k in keys) + "</tr>"
    for i, a in enumerate(keys):
        rows += f"<tr><th>{a[:10]}</th>"
        for j, _b in enumerate(keys):
            v = matrix[i][j]
            # color scale
            alpha = min(1.0, abs(v) * 2)
            bg = f"rgba(46,204,113,{alpha})" if v > 0.05 else (
                f"rgba(231,76,60,{alpha})" if v < -0.05 else "transparent"
            )
            rows += f'<td style="background:{bg}">{v:+.2f}</td>'
        rows += "</tr>"

    cov_rows = ""
    for k, v in coverage.items():
        lab = areas[k]["label"] if k in areas else k
        pct = 100 * v / sample_n
        cov_rows += f"<tr><td>{lab}</td><td>{v}</td><td>{pct:.1f}%</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>TinyLlama — Mapa de áreas semánticas (PCA 2D)</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #0f1115; color: #e8e8e8; }}
  header {{ padding: 16px 24px; border-bottom: 1px solid #333; }}
  h1 {{ margin: 0 0 6px; font-size: 1.25rem; }}
  .sub {{ color: #999; font-size: 0.9rem; }}
  main {{ display: grid; grid-template-columns: 1fr 320px; gap: 0; height: calc(100vh - 80px); }}
  #canvas-wrap {{ position: relative; background: #151820; }}
  canvas {{ width: 100%; height: 100%; display: block; cursor: crosshair; }}
  aside {{ padding: 16px; overflow: auto; border-left: 1px solid #333; background: #12151c; }}
  .leg {{ margin: 4px 0; font-size: 0.85rem; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  table {{ border-collapse: collapse; font-size: 0.7rem; width: 100%; margin-top: 8px; }}
  th, td {{ border: 1px solid #333; padding: 3px 4px; text-align: center; }}
  #tooltip {{
    position: absolute; pointer-events: none; background: #1e2430; border: 1px solid #555;
    padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; display: none; z-index: 10;
  }}
  h2 {{ font-size: 0.95rem; margin: 16px 0 8px; color: #aaa; }}
  code {{ background: #222; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<header>
  <h1>Áreas semánticas en ℝ²⁰⁴⁸ → PCA 2D</h1>
  <div class="sub">TinyLlama-1.1B token embeddings · semillas + expansión · fondo aleatorio gris</div>
</header>
<main>
  <div id="canvas-wrap">
    <canvas id="c"></canvas>
    <div id="tooltip"></div>
  </div>
  <aside>
    <h2>Leyenda</h2>
    {legend}
    <div class="leg"><span class="dot" style="background:#ccc"></span>fondo (muestra)</div>
    <h2>Cobertura (n={sample_n}, thr={thr})</h2>
    <table><tr><th>Área</th><th>n</th><th>%</th></tr>{cov_rows}</table>
    <h2>Similitud entre áreas</h2>
    <table>{rows}</table>
    <h2>Herramientas recomendadas</h2>
    <ul style="font-size:0.8rem;padding-left:18px;color:#bbb">
      <li><b>Este mapa</b> — PCA 2D (numpy)</li>
      <li><b>UMAP</b> — <code>pip install umap-learn</code></li>
      <li><b>t-SNE</b> — sklearn.manifold</li>
      <li><b>Embedding Projector</b> — projector.tensorflow.org<br/>
          cargar <code>vectors.tsv</code> + <code>metadata.tsv</code></li>
      <li><b>plotly</b> — 3D interactivo</li>
    </ul>
    <p style="font-size:0.75rem;color:#666">Rueda = zoom · arrastrar = pan · hover = token</p>
  </aside>
</main>
<script>
const points = {pts_json};
const cents = {cents_json};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const tip = document.getElementById('tooltip');
const wrap = document.getElementById('canvas-wrap');

let W, H, scale = 0.42, ox = 0, oy = 0;
let drag = false, lx = 0, ly = 0;

function resize() {{
  W = wrap.clientWidth; H = wrap.clientHeight;
  canvas.width = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  draw();
}}

function toScreen(x, y) {{
  return [W/2 + (x + ox) * W * scale, H/2 - (y + oy) * H * scale];
}}

function draw() {{
  ctx.fillStyle = '#151820';
  ctx.fillRect(0, 0, W, H);
  // axes
  ctx.strokeStyle = '#2a2f3a';
  ctx.beginPath();
  const [zx, zy] = toScreen(0, 0);
  ctx.moveTo(0, zy); ctx.lineTo(W, zy);
  ctx.moveTo(zx, 0); ctx.lineTo(zx, H);
  ctx.stroke();

  // background first
  for (const p of points) {{
    if (p.area !== 'bg') continue;
    const [sx, sy] = toScreen(p.x, p.y);
    ctx.fillStyle = '#3a3f4a88';
    ctx.beginPath(); ctx.arc(sx, sy, 2, 0, 6.28); ctx.fill();
  }}
  // areas
  for (const p of points) {{
    if (p.area === 'bg') continue;
    const [sx, sy] = toScreen(p.x, p.y);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = p.seed ? 1 : 0.75;
    ctx.beginPath();
    ctx.arc(sx, sy, p.seed ? 5.5 : 3.5, 0, 6.28);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (p.seed) {{
      ctx.fillStyle = '#fff';
      ctx.font = '11px system-ui';
      ctx.fillText(p.token.replace('\\u2581','▁'), sx + 7, sy + 3);
    }}
  }}
  // centroids
  for (const [k, c] of Object.entries(cents)) {{
    const [sx, sy] = toScreen(c.x, c.y);
    ctx.strokeStyle = c.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx - 8, sy); ctx.lineTo(sx + 8, sy);
    ctx.moveTo(sx, sy - 8); ctx.lineTo(sx, sy + 8);
    ctx.stroke();
    ctx.fillStyle = c.color;
    ctx.font = 'bold 12px system-ui';
    ctx.fillText(c.label, sx + 10, sy - 8);
  }}
}}

canvas.addEventListener('mousedown', e => {{ drag = true; lx = e.clientX; ly = e.clientY; }});
window.addEventListener('mouseup', () => drag = false);
canvas.addEventListener('mousemove', e => {{
  if (drag) {{
    ox += (e.clientX - lx) / (W * scale);
    oy -= (e.clientY - ly) / (H * scale);
    lx = e.clientX; ly = e.clientY;
    draw();
  }}
  // tooltip
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bd = 12;
  for (const p of points) {{
    if (p.area === 'bg') continue;
    const [sx, sy] = toScreen(p.x, p.y);
    const d = Math.hypot(sx - mx, sy - my);
    if (d < bd) {{ bd = d; best = p; }}
  }}
  if (best) {{
    tip.style.display = 'block';
    tip.style.left = (e.clientX - rect.left + 12) + 'px';
    tip.style.top = (e.clientY - rect.top + 12) + 'px';
    tip.innerHTML = `<b>${{best.token}}</b><br/>${{best.label}}` + (best.seed ? ' · seed' : '');
  }} else tip.style.display = 'none';
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  scale *= e.deltaY > 0 ? 0.9 : 1.1;
  scale = Math.max(0.08, Math.min(8, scale));
  draw();
}}, {{ passive: false }});

window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
