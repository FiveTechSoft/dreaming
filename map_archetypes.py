#!/usr/bin/env python3
"""
Mapea arquetipos (Pearson/Jung + perspectivas Dreaming) en embeddings TinyLlama
e identifica constelaciones (estrellas = tokens vecinos del centroide).
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

# ---------------------------------------------------------------------------
# Arquetipos: semillas en inglés (BPE ▁) — lenguaje dominante del preentreno
# ---------------------------------------------------------------------------
ARCHETYPES = {
    "hero": {
        "label": "Héroe",
        "symbol": "⚔",
        "color": "#e74c3c",
        "myth": "Afronta el peligro, supera la prueba, salva",
        "seeds": ["hero", "courage", "brave", "quest", "victory", "warrior", "fight", "strength", "honor", "triumph"],
    },
    "shadow": {
        "label": "Sombra",
        "symbol": "🌑",
        "color": "#2c3e50",
        "myth": "Lo reprimido, el enemigo interior, el monstruo",
        "seeds": ["shadow", "dark", "evil", "fear", "hate", "monster", "demon", "rage", "corrupt", "sin"],
    },
    "sage": {
        "label": "Sabio",
        "symbol": "📜",
        "color": "#3498db",
        "myth": "Conocimiento, verdad, enseñanza, análisis",
        "seeds": ["wisdom", "truth", "knowledge", "scholar", "theory", "reason", "logic", "study", "philosophy", "mind"],
    },
    "caregiver": {
        "label": "Cuidador",
        "symbol": "💚",
        "color": "#27ae60",
        "myth": "Cuidar, sanar, proteger, compasión",
        "seeds": ["care", "heal", "love", "kind", "help", "protect", "gentle", "mercy", "nurse", "comfort"],
    },
    "explorer": {
        "label": "Explorador",
        "symbol": "🧭",
        "color": "#f39c12",
        "myth": "Viaje, frontera, descubrimiento, libertad",
        "seeds": ["explore", "journey", "discover", "travel", "freedom", "adventure", "path", "wild", "seek", "horizon"],
    },
    "creator": {
        "label": "Creador",
        "symbol": "✨",
        "color": "#9b59b6",
        "myth": "Arte, invención, imaginación, obra",
        "seeds": ["create", "art", "imagine", "beauty", "music", "poem", "invent", "craft", "design", "dream"],
    },
    "ruler": {
        "label": "Gobernante",
        "symbol": "👑",
        "color": "#e67e22",
        "myth": "Orden, poder, ley, control",
        "seeds": ["king", "power", "law", "order", "rule", "empire", "throne", "command", "authority", "nation"],
    },
    "magician": {
        "label": "Mago",
        "symbol": "🔮",
        "color": "#8e44ad",
        "myth": "Transformación, misterio, ritual, lo oculto",
        "seeds": ["magic", "spirit", "soul", "divine", "sacred", "ritual", "mystery", "transform", "alchemy", "vision"],
    },
    "innocent": {
        "label": "Inocente",
        "symbol": "🌸",
        "color": "#1abc9c",
        "myth": "Pureza, esperanza, fe simple, paraíso",
        "seeds": ["innocent", "hope", "faith", "pure", "happy", "child", "peace", "trust", "simple", "good"],
    },
    "lover": {
        "label": "Amante",
        "symbol": "❤",
        "color": "#e91e63",
        "myth": "Deseo, unión, intimidad, pasión",
        "seeds": ["love", "desire", "kiss", "passion", "heart", "romance", "beauty", "embrace", "beloved", "tender"],
    },
    "jester": {
        "label": "Bufón",
        "symbol": "🃏",
        "color": "#f1c40f",
        "myth": "Juego, ironía, risa, subversión",
        "seeds": ["laugh", "joke", "funny", "play", "fool", "comic", "smile", "wit", "mock", "silly"],
    },
    "orphan": {
        "label": "Huérfano / realista",
        "symbol": "🏚",
        "color": "#7f8c8d",
        "myth": "Pertenencia, supervivencia, realismo crudo",
        "seeds": ["alone", "lost", "survive", "pain", "real", "ordinary", "poor", "need", "belong", "home"],
    },
    # Perspectivas Dreaming como arquetipos operativos
    "mystic_voice": {
        "label": "Voz mística (Dreaming)",
        "symbol": "🕯",
        "color": "#6c3483",
        "myth": "Ego disuelto, universo, eternidad — clima mystical",
        "seeds": ["soul", "spirit", "ego", "eternal", "universe", "consciousness", "divine", "silence", "being", "transcend"],
    },
    "practical_voice": {
        "label": "Voz práctica (FFN / Regla de Oro)",
        "symbol": "🔧",
        "color": "#16a085",
        "myth": "Acción, pasos, utilidad",
        "seeds": ["should", "step", "action", "goal", "plan", "work", "build", "fix", "method", "practice"],
    },
    "academic_voice": {
        "label": "Voz académica (Attn / Regla de Oro)",
        "symbol": "🎓",
        "color": "#2980b9",
        "myth": "Análisis, teoría, crítica, estructura",
        "seeds": ["theory", "analysis", "study", "research", "argument", "concept", "framework", "evidence", "scholar", "critique"],
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


def resolve(vocab, word):
    for w in ("\u2581" + word, "\u2581" + word.lower(), word, word.lower()):
        if w in vocab:
            return vocab[w], w
    return None, None


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    print("Cargando…")
    E = load_E()
    vocab, id2tok = load_vocab()
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    arch = {}
    print("\n" + "=" * 70)
    print("ARQUETIPOS Y CONSTELACIONES")
    print("=" * 70)

    for key, meta in ARCHETYPES.items():
        ids, seeds_used = [], []
        for w in meta["seeds"]:
            i, t = resolve(vocab, w)
            if i is not None:
                ids.append(i)
                seeds_used.append(t)
        if len(ids) < 3:
            print(f"[skip] {key}: {len(ids)} seeds")
            continue
        C = E[ids].mean(0)
        Cn = C / (np.linalg.norm(C) + 1e-9)
        sims = En @ Cn
        for i in ids:
            sims[i] = -1e9
        # constellation: nearest *word-like* tokens (▁ + letters)
        order = np.argsort(-sims)
        stars = []
        for i in order:
            if len(stars) >= 12:
                break
            tok = id2tok.get(int(i), str(i))
            core = tok[1:] if tok.startswith("\u2581") else tok
            if len(core) < 3 or not core.isalpha():
                continue
            if not tok.startswith("\u2581") and len(tok) < 4:
                continue
            stars.append({"token": tok, "id": int(i), "cos": float(sims[i])})
        # fallback if filter too strict
        if len(stars) < 4:
            for i in order[:12]:
                tok = id2tok.get(int(i), str(i))
                stars.append({"token": tok, "id": int(i), "cos": float(sims[i])})

        # cohesion: mean pairwise cos among seeds
        S = E[ids]
        Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
        # average off-diagonal
        G = Sn @ Sn.T
        n = len(ids)
        coh = float((G.sum() - n) / (n * (n - 1) + 1e-9))

        arch[key] = {
            "label": meta["label"],
            "symbol": meta["symbol"],
            "color": meta["color"],
            "myth": meta["myth"],
            "seeds": seeds_used,
            "seed_ids": ids,
            "centroid": Cn,
            "constellation": stars,
            "cohesion": coh,
            "n_seeds": n,
        }

        print(f"\n{meta['symbol']} {meta['label']}  ({key})  cohesión={coh:+.3f}  n={n}")
        print(f"   mito: {meta['myth']}")
        print(f"   semillas: {seeds_used}")
        print(f"   constelación (estrellas vecinas):")
        for s in stars[:8]:
            print(f"      {s['cos']:+.3f}  {s['token']!r}")

    keys = list(arch.keys())

    # --- Matriz entre arquetipos ---
    print("\n" + "=" * 70)
    print("ÁNGULOS ENTRE ARQUETIPOS (cosine centroides)")
    print("=" * 70)
    mat = []
    for a in keys:
        row = [cos(arch[a]["centroid"], arch[b]["centroid"]) for b in keys]
        mat.append(row)
    # print compact
    print(" " * 14 + " ".join(f"{k[:5]:>6s}" for k in keys))
    for i, a in enumerate(keys):
        print(f"{a:12s}  " + " ".join(f"{x:+.2f}" for x in mat[i]))

    pairs = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                pairs.append((mat[i][j], a, b))
    pairs.sort(reverse=True)
    print("\nAlineaciones (atracción arquetípica):")
    for c, a, b in pairs[:10]:
        print(f"  {c:+.3f}  {arch[a]['label']:20s} ↔ {arch[b]['label']}")
    print("\nOposiciones / ortogonalidad:")
    for c, a, b in pairs[-8:]:
        print(f"  {c:+.3f}  {arch[a]['label']:20s} ↔ {arch[b]['label']}")

    # --- Asignación: qué arquetipo “gana” en una muestra ---
    rng = np.random.default_rng(1)
    sample = rng.choice(E.shape[0], 4000, replace=False)
    Cmat = np.stack([arch[k]["centroid"] for k in keys])
    S = En[sample] @ Cmat.T
    assign = S.argmax(1)
    maxs = S.max(1)
    thr = 0.10
    counts = {k: 0 for k in keys}
    counts["_void"] = 0
    for a, s in zip(assign, maxs):
        if s < thr:
            counts["_void"] += 1
        else:
            counts[keys[a]] += 1
    print("\n" + "=" * 70)
    print(f"DOMINANCIA EN MUESTRA (n=4000, thr={thr})")
    print("=" * 70)
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        if k == "_void":
            lab = "(sin arquetipo cercano)"
        else:
            lab = arch[k]["label"]
        print(f"  {lab:28s}  {v:5d}  ({100*v/4000:.1f}%)")

    # --- PCA 2D de semillas + estrellas de constelación ---
    print("\nProyectando mapa de arquetipos…")
    plot_ids = set()
    labels = {}
    for k, a in arch.items():
        for i, t in zip(a["seed_ids"], a["seeds"]):
            plot_ids.add(i)
            labels[i] = (k, t, "seed")
        for st in a["constellation"][:6]:
            i = st["id"]
            if i not in labels:
                plot_ids.add(i)
                labels[i] = (k, st["token"], "star")
    for i in rng.choice(E.shape[0], 600, replace=False):
        i = int(i)
        if i not in labels:
            plot_ids.add(i)
            labels[i] = ("bg", id2tok.get(i, "?")[:16], "bg")

    ids = np.array(sorted(plot_ids))
    X = E[ids]
    mean0 = X.mean(0)
    Xc = X - mean0
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = (Xc @ Vt[:2].T)
    P = P / (np.abs(P).max() + 1e-9)

    points = []
    for j, i in enumerate(ids):
        k, tok, role = labels[int(i)]
        color = arch[k]["color"] if k in arch else "#555555"
        lab = arch[k]["label"] if k in arch else "fondo"
        points.append({
            "x": float(P[j, 0]),
            "y": float(P[j, 1]),
            "token": tok,
            "archetype": k,
            "label": lab,
            "color": color,
            "role": role,
            "id": int(i),
        })

    cents = {}
    for k in keys:
        c = E[arch[k]["seed_ids"]].mean(0) - mean0
        xy = (c @ Vt[:2].T) / (np.abs(P).max() * (np.abs(P).max()) + 1e-9)
        # consistent scale with P already normalized by max abs of P before / max
        xy = c @ Vt[:2].T
        scale = np.abs((X - mean0) @ Vt[:2].T).max() + 1e-9
        xy = xy / scale
        cents[k] = {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "label": arch[k]["label"],
            "symbol": arch[k]["symbol"],
            "color": arch[k]["color"],
        }

    # fix points scale to same
    scale = np.abs((X - mean0) @ Vt[:2].T).max() + 1e-9
    P2 = ((X - mean0) @ Vt[:2].T) / scale
    for j, p in enumerate(points):
        p["x"] = float(P2[j, 0])
        p["y"] = float(P2[j, 1])

    html = build_html(points, cents, arch, mat, keys, counts, 4000, thr)
    (OUT / "archetype_map.html").write_text(html, encoding="utf-8")

    # serializable export
    export = {
        "archetypes": {
            k: {
                "label": a["label"],
                "symbol": a["symbol"],
                "color": a["color"],
                "myth": a["myth"],
                "seeds": a["seeds"],
                "constellation": a["constellation"],
                "cohesion": a["cohesion"],
                "n_seeds": a["n_seeds"],
            }
            for k, a in arch.items()
        },
        "cosine_matrix": {"names": keys, "matrix": mat},
        "alignments": [{"cos": c, "a": a, "b": b} for c, a, b in pairs[:12]],
        "oppositions": [{"cos": c, "a": a, "b": b} for c, a, b in pairs[-10:]],
        "dominance_sample": counts,
        "map": "archetype_map.html",
    }
    with open(OUT / "archetypes.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)

    # markdown report for book
    md = build_md_chapter(arch, pairs, counts, mat, keys)
    Path(r"C:/tmp/dreaming/inside-tinyllama/chapters/21_arquetipos_constelaciones.md").write_text(
        md, encoding="utf-8"
    )

    print(f"\nHTML: {OUT / 'archetype_map.html'}")
    print(f"JSON: {OUT / 'archetypes.json'}")
    print("Capítulo: chapters/21_arquetipos_constelaciones.md")


def build_md_chapter(arch, pairs, counts, mat, keys):
    lines = [
        "# Capítulo 21: Arquetipos y Constelaciones",
        "",
        "## Qué es un arquetipo aquí",
        "",
        "No es un personaje de novela instalado en el modelo.",
        "Es un **atractor geométrico**: el centroide en ℝ²⁰⁴⁸ de un",
        "racimo de tokens-semilla (BPE `▁…`) que, en la cultura",
        "del preentreno, co-ocurren con un mito recurrente",
        "(Héroe, Sombra, Sabio, Mago… y las voces Dreaming).",
        "",
        "Una **constelación** es el conjunto de estrellas-token",
        "más cercanas a ese centroide (vecinos cosine), más las semillas.",
        "",
        "## Catálogo de arquetipos medidos",
        "",
        "| Símbolo | Arquetipo | Cohesión | Semillas (n) | Estrellas (muestra) |",
        "|---------|-----------|----------|--------------|---------------------|",
    ]
    for k, a in arch.items():
        stars = ", ".join(s["token"] for s in a["constellation"][:5])
        lines.append(
            f"| {a['symbol']} | **{a['label']}** | {a['cohesion']:+.3f} | "
            f"{a['n_seeds']} | {stars} |"
        )
    lines += [
        "",
        "### Mito operativo de cada uno",
        "",
    ]
    for k, a in arch.items():
        lines.append(f"- **{a['label']}** — {a['myth']}")
        lines.append(f"  - semillas: `{', '.join(a['seeds'][:6])}…`")
        lines.append(
            f"  - constelación: "
            + ", ".join(f"`{s['token']}`({s['cos']:+.2f})" for s in a["constellation"][:6])
        )
        lines.append("")

    lines += [
        "## Alineaciones (constelaciones que se rozan)",
        "",
        "| cos | Arquetipo A | Arquetipo B |",
        "|-----|-------------|-------------|",
    ]
    for c, a, b in pairs[:8]:
        lines.append(f"| {c:+.3f} | {arch[a]['label']} | {arch[b]['label']} |")

    lines += [
        "",
        "## Oposiciones / casi ortogonales",
        "",
        "| cos | A | B |",
        "|-----|---|---|",
    ]
    for c, a, b in pairs[-6:]:
        lines.append(f"| {c:+.3f} | {arch[a]['label']} | {arch[b]['label']} |")

    lines += [
        "",
        "## Lectura",
        "",
        "1. **Cohesión alta** ⇒ semillas forman un racimo tenso (constelación nítida).",
        "2. **Alineación** Sabio↔Académico, Mago↔Místico Dreaming ⇒",
        "   las voces del proyecto *no son ajenas* a arquetipos culturales.",
        "3. **Sombra ↔ Inocente / Héroe** en ángulos abiertos ⇒ polos de drama.",
        "4. Casi todo el vocab BPE sigue en el **vacío** entre constelaciones",
        "   (fragmentos, no mitos).",
        "",
        "## Mapa",
        "",
        "HTML: `exploration/archetype_map.html`",
        "",
        "Preview (tras push a main):",
        "https://htmlpreview.github.io/?https://raw.githubusercontent.com/FiveTechSoft/dreaming/main/inside-tinyllama/exploration/archetype_map.html",
        "",
        "## Cómo orbitar un arquetipo",
        "",
        "```bash",
        "# Viento hacia el mago / místico",
        "./llm_inference modelo.F16.gguf \"The meaning of life is\" 50 0.7 40 \\\\",
        "  --seed 42 --steer soul --steer-strength 0.2",
        "",
        "# Lente de pesos mística (perspectiva Dreaming)",
        "./llm_inference modelo.F16.gguf \"When we dissolve the ego\" 50 0.7 40 \\\\",
        "  --seed 42 --perturb mystical --intensity 0.50",
        "```",
        "",
        "Prompt + isla arquetípica + lente de pesos = viaje con coordenadas.",
        "",
        "---",
        "",
        "*Siguiente: usar arquetipos como brújula de --steer y de evaluación de perspectiva.*",
        "",
    ]
    return "\n".join(lines)


def build_html(points, cents, arch, mat, keys, counts, n_sample, thr):
    pts = json.dumps(points, ensure_ascii=False)
    cts = json.dumps(cents, ensure_ascii=False)
    legend = ""
    for k, a in arch.items():
        legend += (
            f'<div class="leg"><span class="dot" style="background:{a["color"]}"></span>'
            f'{a["symbol"]} {a["label"]}</div>\n'
        )
    # top alignments text
    pairs = []
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i < j:
                pairs.append((mat[i][j], a, b))
    pairs.sort(reverse=True)
    al = "<br/>".join(
        f"{c:+.2f} {arch[a]['label']} ↔ {arch[b]['label']}" for c, a, b in pairs[:6]
    )
    return f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>TinyLlama — Arquetipos y constelaciones</title>
<style>
body{{margin:0;font-family:system-ui,sans-serif;background:#0b0d12;color:#eee}}
header{{padding:14px 20px;border-bottom:1px solid #333}}
h1{{margin:0;font-size:1.2rem}}
.sub{{color:#888;font-size:.85rem;margin-top:4px}}
main{{display:grid;grid-template-columns:1fr 300px;height:calc(100vh - 72px)}}
#wrap{{position:relative;background:#10141c}}
canvas{{width:100%;height:100%;display:block;cursor:crosshair}}
aside{{padding:14px;overflow:auto;border-left:1px solid #333;background:#0e1218;font-size:.82rem}}
.leg{{margin:3px 0}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
#tip{{position:absolute;display:none;background:#1a2030;border:1px solid #555;padding:6px 10px;border-radius:6px;font-size:.8rem;pointer-events:none;z-index:5}}
h2{{font-size:.9rem;color:#9ab;margin:14px 0 6px}}
</style></head><body>
<header>
  <h1>Arquetipos y constelaciones en ℝ²⁰⁴⁸ → PCA 2D</h1>
  <div class="sub">Semillas = discos grandes · Estrellas de constelación = puntos de color · Fondo gris = vocabulario</div>
</header>
<main>
<div id="wrap"><canvas id="c"></canvas><div id="tip"></div></div>
<aside>
  <h2>Arquetipos</h2>
  {legend}
  <h2>Alineaciones fuertes</h2>
  <div style="color:#aaa;line-height:1.45">{al}</div>
  <h2>Controles</h2>
  <p style="color:#666">Rueda zoom · Arrastrar pan · Hover token</p>
</aside>
</main>
<script>
const points={pts};
const cents={cts};
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
const tip=document.getElementById('tip');
const wrap=document.getElementById('wrap');
let W,H,scale=0.4,ox=0,oy=0,drag=false,lx=0,ly=0;
function resize(){{W=wrap.clientWidth;H=wrap.clientHeight;canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw();}}
function scr(x,y){{return[W/2+(x+ox)*W*scale,H/2-(y+oy)*H*scale];}}
function draw(){{
  ctx.fillStyle='#10141c';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#222836';ctx.beginPath();
  const[zx,zy]=scr(0,0);ctx.moveTo(0,zy);ctx.lineTo(W,zy);ctx.moveTo(zx,0);ctx.lineTo(zx,H);ctx.stroke();
  for(const p of points){{
    if(p.role!=='bg') continue;
    const[sx,sy]=scr(p.x,p.y);
    ctx.fillStyle='#2a3040aa';ctx.beginPath();ctx.arc(sx,sy,1.8,0,6.3);ctx.fill();
  }}
  for(const p of points){{
    if(p.role==='bg') continue;
    const[sx,sy]=scr(p.x,p.y);
    const r=p.role==='seed'?6:3.2;
    ctx.globalAlpha=p.role==='seed'?1:0.8;
    ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(sx,sy,r,0,6.3);ctx.fill();
    ctx.globalAlpha=1;
    if(p.role==='seed'){{ctx.fillStyle='#fff';ctx.font='11px system-ui';ctx.fillText(p.token.replace('\\u2581','▁'),sx+8,sy+3);}}
  }}
  for(const[k,c] of Object.entries(cents)){{
    const[sx,sy]=scr(c.x,c.y);
    ctx.strokeStyle=c.color;ctx.lineWidth=2;ctx.beginPath();
    ctx.moveTo(sx-9,sy);ctx.lineTo(sx+9,sy);ctx.moveTo(sx,sy-9);ctx.lineTo(sx,sy+9);ctx.stroke();
    ctx.fillStyle=c.color;ctx.font='bold 12px system-ui';
    ctx.fillText((c.symbol||'')+' '+c.label,sx+11,sy-8);
  }}
}}
canvas.onmousedown=e=>{{drag=true;lx=e.clientX;ly=e.clientY;}};
window.onmouseup=()=>drag=false;
canvas.onmousemove=e=>{{
  if(drag){{ox+=(e.clientX-lx)/(W*scale);oy-=(e.clientY-ly)/(H*scale);lx=e.clientX;ly=e.clientY;draw();}}
  const rect=canvas.getBoundingClientRect();const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  let best=null,bd=14;
  for(const p of points){{if(p.role==='bg')continue;const[sx,sy]=scr(p.x,p.y);const d=Math.hypot(sx-mx,sy-my);if(d<bd){{bd=d;best=p;}}}}
  if(best){{tip.style.display='block';tip.style.left=(mx+12)+'px';tip.style.top=(my+12)+'px';
    tip.innerHTML='<b>'+best.token+'</b><br/>'+best.label+(best.role==='seed'?' · semilla':best.role==='star'?' · estrella':'');}}
  else tip.style.display='none';
}};
canvas.onwheel=e=>{{e.preventDefault();scale*=e.deltaY>0?0.9:1.1;scale=Math.max(0.06,Math.min(8,scale));draw();}};
window.onresize=resize;resize();
</script></body></html>"""


if __name__ == "__main__":
    main()
