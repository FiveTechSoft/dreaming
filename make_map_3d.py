#!/usr/bin/env python3
"""Genera mapa 3D PCA con rotación al arrastrar (Plotly vía CDN)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import struct
from pathlib import Path

import numpy as np
from gguf import GGUFReader

MODEL = r"C:/tmp/tinyllama-1.1b.F16.gguf"
VOCAB = r"C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUT = Path(r"C:/tmp/dreaming/inside-tinyllama/exploration")


def load_E():
    r = GGUFReader(MODEL)
    E = np.array(next(t for t in r.tensors if t.name == "token_embd.weight").data, np.float32)
    if E.shape[0] == 2048:
        E = E.T
    return E


def load_vocab():
    with open(VOCAB, "rb") as f:
        data = f.read(8 * 1024 * 1024)
    pos = 24
    _nt, nk = struct.unpack_from("<QQ", data, 8)

    def rd_str(p):
        ln = struct.unpack_from("<Q", data, p)[0]
        p += 8
        return data[p : p + ln], p + ln

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
            return {t: i for i, t in enumerate(tokens)}, {i: t for i, t in enumerate(tokens)}
        pos = skip(pos, vt)
    raise RuntimeError("vocab")


def resolve(vocab, w):
    for c in ("\u2581" + w, w, "\u2581" + w.lower(), w.lower()):
        if c in vocab:
            return vocab[c], c
    return None, None


def main():
    E = load_E()
    vocab, id2tok = load_vocab()
    areas = json.load(open(OUT / "semantic_areas.json", encoding="utf-8"))["areas"]
    try:
        arch = json.load(open(OUT / "archetypes.json", encoding="utf-8"))["archetypes"]
    except FileNotFoundError:
        arch = {}

    pts = []
    ids_set = set()
    rng = np.random.default_rng(0)

    def add(i, label, color, kind):
        if i in ids_set and kind != "seed":
            return
        ids_set.add(i)
        pts.append(
            {
                "token": id2tok.get(i, "?"),
                "label": label,
                "color": color,
                "kind": kind,
                "i": i,
            }
        )

    for k, a in areas.items():
        for s in a["seeds"]:
            w = s.lstrip("\u2581")
            i, _ = resolve(vocab, w)
            if i is not None:
                add(i, a["label"], a["color"], "seed")
    for k, a in arch.items():
        for s in a.get("seeds", []):
            w = s.lstrip("\u2581")
            i, _ = resolve(vocab, w)
            if i is not None:
                add(i, a["label"], a["color"], "seed")
    for i in rng.choice(E.shape[0], 600, replace=False):
        i = int(i)
        if i not in ids_set:
            add(i, "fondo", "#666666", "bg")

    idx = np.array([p["i"] for p in pts])
    X = E[idx]
    Xc = X - X.mean(0)
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:3].T
    P = P / (np.abs(P).max() + 1e-9)
    for j, p in enumerate(pts):
        p["x"], p["y"], p["z"] = float(P[j, 0]), float(P[j, 1]), float(P[j, 2])
        del p["i"]

    data_json = json.dumps(pts, ensure_ascii=False)
    # write without f-string nesting issues
    html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TinyLlama — Mapa 3D (girar con el ratón)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { margin:0; font-family: system-ui, sans-serif; background:#0b0d12; color:#e8e8e8; }
  header { padding:12px 18px; border-bottom:1px solid #333; }
  h1 { margin:0; font-size:1.15rem; }
  .sub { color:#888; font-size:0.85rem; margin-top:6px; line-height:1.5; }
  kbd { background:#1e2430; padding:2px 7px; border-radius:4px; border:1px solid #444; font-size:0.8rem; }
  #plot { width:100vw; height:calc(100vh - 88px); }
</style>
</head>
<body>
<header>
  <h1>Espacio de embeddings TinyLlama — PCA 3D</h1>
  <div class="sub">
    <kbd>Arrastrar</kbd> girar &nbsp;·&nbsp;
    <kbd>Rueda</kbd> zoom &nbsp;·&nbsp;
    <kbd>Clic derecho + arrastrar</kbd> pan &nbsp;·&nbsp;
    Hover = token · puntos de color = semillas de áreas/arquetipos · gris = fondo
  </div>
</header>
<div id="plot"></div>
<script>
const pts = """ + data_json + """;
const seeds = pts.filter(p => p.kind === 'seed');
const bg = pts.filter(p => p.kind === 'bg');
const tBg = {
  type: 'scatter3d', mode: 'markers', name: 'fondo (muestra)',
  x: bg.map(p => p.x), y: bg.map(p => p.y), z: bg.map(p => z => p.z),
  // fix below
};
</script>
</body>
</html>
"""
    # Fix the accidental bug in template - write clean version
    html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>TinyLlama — Mapa 3D (girar con el ratón)</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body { margin:0; font-family: system-ui, sans-serif; background:#0b0d12; color:#e8e8e8; }
  header { padding:12px 18px; border-bottom:1px solid #333; }
  h1 { margin:0; font-size:1.15rem; }
  .sub { color:#888; font-size:0.85rem; margin-top:6px; line-height:1.55; }
  kbd { background:#1e2430; padding:2px 7px; border-radius:4px; border:1px solid #444; font-size:0.8rem; }
  #plot { width:100vw; height:calc(100vh - 88px); }
</style>
</head>
<body>
<header>
  <h1>Espacio de embeddings TinyLlama — PCA 3D</h1>
  <div class="sub">
    <kbd>Clic izquierdo + arrastrar</kbd> girar la cámara &nbsp;·&nbsp;
    <kbd>Rueda</kbd> zoom &nbsp;·&nbsp;
    <kbd>Clic derecho + arrastrar</kbd> desplazar &nbsp;·&nbsp;
    Hover sobre un punto = token y área/arquetipo
  </div>
</header>
<div id="plot"></div>
<script>
const pts = """ + data_json + """;
const seeds = pts.filter(p => p.kind === 'seed');
const bg = pts.filter(p => p.kind === 'bg');
const tBg = {
  type: 'scatter3d', mode: 'markers', name: 'fondo (muestra)',
  x: bg.map(p => p.x), y: bg.map(p => p.y), z: bg.map(p => p.z),
  text: bg.map(p => p.token),
  hovertemplate: '%{text}<extra></extra>',
  marker: { size: 2.5, color: '#555555', opacity: 0.4 }
};
const tSeeds = {
  type: 'scatter3d', mode: 'markers', name: 'semillas (áreas / arquetipos)',
  x: seeds.map(p => p.x), y: seeds.map(p => p.y), z: seeds.map(p => p.z),
  text: seeds.map(p => p.token + ' · ' + p.label),
  hovertemplate: '%{text}<extra></extra>',
  marker: { size: 7, color: seeds.map(p => p.color), opacity: 0.95 }
};
const layout = {
  paper_bgcolor: '#0b0d12',
  font: { color: '#ccc', size: 12 },
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: true,
  legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(20,24,32,0.85)' },
  scene: {
    bgcolor: '#10141c',
    xaxis: { title: 'PC1', gridcolor: '#2a3040', zerolinecolor: '#444', color: '#888' },
    yaxis: { title: 'PC2', gridcolor: '#2a3040', zerolinecolor: '#444', color: '#888' },
    zaxis: { title: 'PC3', gridcolor: '#2a3040', zerolinecolor: '#444', color: '#888' },
    camera: { eye: { x: 1.45, y: 1.2, z: 0.95 } },
    aspectmode: 'cube',
    dragmode: 'orbit'
  }
};
const config = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['resetCameraLastSave3d'],
  scrollZoom: true
};
Plotly.newPlot('plot', [tBg, tSeeds], layout, config);
</script>
</body>
</html>
"""
    path = OUT / "map_3d_orbit.html"
    path.write_text(html, encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes, {len(pts)} points)")
    print("Open locally or via htmlpreview after push to GitHub.")


if __name__ == "__main__":
    main()
