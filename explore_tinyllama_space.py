#!/usr/bin/env python3
"""Recorrido del espacio multidimensional de TinyLlama (embeddings 2048-D)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
from pathlib import Path

import numpy as np
from gguf import GGUFReader

MODEL = r"C:/tmp/tinyllama-1.1b.F16.gguf"
# Vocab completo (el F16 a veces trae tokenizer truncado)
VOCAB_GGUF = r"C:/tmp/tinyllama-1.1b.Q4_0.gguf"
OUT = Path(r"C:/tmp/dreaming/inside-tinyllama/exploration")
OUT.mkdir(parents=True, exist_ok=True)


def load_embeddings():
    print("Cargando", MODEL)
    r = GGUFReader(MODEL)
    emb_t = next(t for t in r.tensors if t.name == "token_embd.weight")
    print(f"  raw shape={emb_t.data.shape} dtype={emb_t.data.dtype}")
    E = np.array(emb_t.data, dtype=np.float32)
    if E.shape[0] == 2048 and E.shape[1] == 32000:
        E = E.T
    assert E.shape == (32000, 2048), E.shape
    print(f"  E={E.shape}  mean||e||={np.linalg.norm(E, axis=1).mean():.3f}")
    return E


def load_vocab_from_gguf(path=VOCAB_GGUF):
    """Lee tokenizer.ggml.tokens del GGUF (Llama BPE con prefijo ▁)."""
    import struct

    print("Cargando vocab desde", path)
    with open(path, "rb") as f:
        # metadata de tokens cabe en los primeros MB
        data = f.read(8 * 1024 * 1024)
    pos = 4
    pos += 4  # version
    _nt, nk = struct.unpack_from("<QQ", data, pos)
    pos += 16

    def rd_str(p):
        ln = struct.unpack_from("<Q", data, p)[0]
        p += 8
        s = data[p : p + ln]
        p += ln
        return s, p

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

    tokens = None
    for _ in range(nk):
        key_b, pos = rd_str(pos)
        key = key_b.decode("utf-8", "replace")
        vt = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if key == "tokenizer.ggml.tokens" and vt == 9:
            _et = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            n = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
            tokens = []
            for _j in range(n):
                s, pos = rd_str(pos)
                tokens.append(s.decode("utf-8", "replace"))
            break
        pos = skip(pos, vt)

    if not tokens or len(tokens) < 1000:
        raise RuntimeError("No se pudo leer vocabulario del GGUF")

    id2tok = {i: t for i, t in enumerate(tokens)}
    vocab = {t: i for i, t in enumerate(tokens)}
    print(f"  vocab={len(vocab)}  sample={tokens[5000:5005]}")
    return vocab, id2tok


def main():
    E = load_embeddings()
    vocab, id2tok = load_vocab_from_gguf()

    def emb(word):
        # Llama BPE: palabras de inicio de palabra suelen llevar ▁ (U+2581)
        candidates = [
            "\u2581" + word,
            "\u2581" + word.lower(),
            "\u2581" + word.capitalize(),
            word,
            word.lower(),
            word.capitalize(),
        ]
        for w in candidates:
            if w in vocab:
                i = vocab[w]
                return E[i], i, w
        return None, None, None

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def top_k_sim(v, k=8, exclude=None):
        v = v / (np.linalg.norm(v) + 1e-9)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sims = En @ v
        if exclude:
            for e in exclude:
                if e is not None and 0 <= e < len(sims):
                    sims[e] = -1e9
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), id2tok.get(int(i), f"id{i}"), float(sims[i])) for i in idx]

    report = []

    def sec(title):
        line = "\n" + "=" * 70 + f"\n{title}\n" + "=" * 70
        print(line)
        report.append(line)

    def log(s=""):
        print(s)
        report.append(s)

    # ------------------------------------------------------------------
    sec("MAPA DE ESPACIOS MULTIDIMENSIONALES — TinyLlama-1.1B")
    log(
        """
Espacio                         Dims      Objetos / notas
------------------------------- --------- ----------------------------------
1. Embeddings de token          R^2048    32.000 puntos (vocabulario)
2. Residual stream              R^2048    22 capas × longitud de secuencia
3. Cabezas de atención          R^64      32 Q / 4 KV (GQA)
4. FFN intermedio               R^5632    ~69% de los parámetros
5. Logits                       R^32000   softmax → siguiente token
6. Espacio de pesos             R^~1.1e9  superficie de coherencia (modelos)
7. Espacio de perspectivas      subvar.   trayectorias por perturbación
"""
    )

    # ------------------------------------------------------------------
    sec("REGION 1 — Polos semánticos (cosine entre opuestos)")
    pairs = [
        ("love", "hate"),
        ("life", "death"),
        ("light", "dark"),
        ("good", "evil"),
        ("happy", "sad"),
        ("peace", "war"),
        ("mind", "body"),
        ("soul", "body"),
        ("true", "false"),
        ("hot", "cold"),
        ("day", "night"),
        ("king", "queen"),
        ("man", "woman"),
    ]
    log(f"{'par':28s}  cosine")
    pole_rows = []
    for a, b in pairs:
        va, ia, ta = emb(a)
        vb, ib, tb = emb(b)
        if va is None or vb is None:
            log(f"{a}/{b}: MISSING ({ta}/{tb})")
            continue
        c = cos(va, vb)
        log(f"{ta!r:14s} / {tb!r:14s}  {c:+.4f}")
        pole_rows.append({"a": ta, "b": tb, "cos": c})

    # ------------------------------------------------------------------
    sec("REGION 2 — Clusters: centroides y vecinos")
    clusters = {
        "emotion_pos": ["happy", "joy", "love", "peace", "hope", "smile"],
        "emotion_neg": ["sad", "hate", "fear", "anger", "pain", "cry"],
        "spiritual": ["soul", "spirit", "god", "divine", "sacred", "faith"],
        "physical": ["body", "matter", "rock", "water", "fire", "earth"],
        "abstract": ["truth", "beauty", "justice", "freedom", "meaning", "idea"],
        "time": ["time", "past", "future", "moment", "eternity", "now"],
    }
    centroids = {}
    cluster_info = {}
    for name, words in clusters.items():
        vecs, used = [], []
        for w in words:
            v, i, t = emb(w)
            if v is not None:
                vecs.append(v)
                used.append(t)
        if not vecs:
            log(f"[{name}] vacío")
            continue
        C = np.mean(np.stack(vecs), axis=0)
        centroids[name] = C
        neigh = top_k_sim(C, k=6)
        cluster_info[name] = {"tokens": used, "neighbors": neigh}
        log(f"\n[{name}] n={len(used)}  tokens={used}")
        log("  vecinos del centroide:")
        for i, t, s in neigh:
            log(f"    {s:+.3f}  {t!r}")

    names = list(centroids.keys())
    log("\nMatriz cosine entre centroides:")
    header = " " * 14 + " ".join(f"{n[:8]:>8s}" for n in names)
    log(header)
    mat = []
    for a in names:
        row = [cos(centroids[a], centroids[b]) for b in names]
        mat.append(row)
        log(f"{a:12s}  " + " ".join(f"{x:+.3f}" for x in row))

    # ------------------------------------------------------------------
    sec("REGION 3 — Analogías vectoriales (a - b + c)")
    analogies = [
        ("king", "man", "woman"),
        ("walk", "walked", "run"),
        ("good", "better", "bad"),
        ("love", "hate", "war"),
        ("life", "death", "light"),
        ("paris", "france", "italy"),
    ]
    ana_rows = []
    for a, b, c in analogies:
        va, ia, ta = emb(a)
        vb, ib, tb = emb(b)
        vc, ic, tc = emb(c)
        if va is None or vb is None or vc is None:
            log(f"{a}-{b}+{c}: missing")
            continue
        v = va - vb + vc
        hits = top_k_sim(v, k=6, exclude=[ia, ib, ic])
        log(f"\n{ta!r} - {tb!r} + {tc!r}  =>")
        for i, t, s in hits:
            log(f"  {s:+.3f}  {t!r}")
        ana_rows.append({"expr": f"{ta}-{tb}+{tc}", "hits": hits})

    # ------------------------------------------------------------------
    sec("REGION 4 — Estructura global de R^2048 (PCA)")
    rng = np.random.default_rng(0)
    idx = rng.choice(E.shape[0], size=4000, replace=False)
    X = E[idx] - E[idx].mean(0)
    _U, S, _Vt = np.linalg.svd(X, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    cum = np.cumsum(var)
    d50 = int(np.searchsorted(cum, 0.50) + 1)
    d90 = int(np.searchsorted(cum, 0.90) + 1)
    d99 = int(np.searchsorted(cum, 0.99) + 1)
    mean_v = E.mean(0)
    anisotropy = float(np.linalg.norm(mean_v) / (np.linalg.norm(E, axis=1).mean() + 1e-9))
    log("PCA sobre 4000 tokens aleatorios:")
    log(f"  var top-1 = {var[0]*100:.2f}%")
    log(f"  var top-10 = {cum[9]*100:.2f}%")
    log(f"  var top-50 = {cum[49]*100:.2f}%")
    log(f"  var top-100 = {cum[99]*100:.2f}%")
    log(f"  dims para 50% / 90% / 99% de var: {d50} / {d90} / {d99}")
    log(f"  anisotropía (||mean|| / mean||e||): {anisotropy:.4f}")
    log(f"  S[0]={S[0]:.1f}  S[10]={S[10]:.1f}  S[100]={S[100]:.1f}  S[500]={S[500]:.1f}")

    # ------------------------------------------------------------------
    sec("REGION 5 — Direcciones semánticas (diferencias de centroides)")
    directions = [
        ("emotion", "emotion_pos", "emotion_neg"),
        ("spirit_vs_matter", "spiritual", "physical"),
        ("abstract_vs_physical", "abstract", "physical"),
    ]
    dir_info = {}
    for label, pos, neg in directions:
        if pos not in centroids or neg not in centroids:
            continue
        d = centroids[pos] - centroids[neg]
        d = d / (np.linalg.norm(d) + 1e-9)
        dir_info[label] = d
        log(f"\ndirección «{label}» = centroid({pos}) - centroid({neg})")
        log("  extremo + :")
        for i, t, s in top_k_sim(d, k=5):
            log(f"    {s:+.3f}  {t!r}")
        log("  extremo - :")
        for i, t, s in top_k_sim(-d, k=5):
            log(f"    {s:+.3f}  {t!r}")

    # ------------------------------------------------------------------
    sec("REGION 6 — Normas: ¿qué tokens 'pesan' más?")
    norms = np.linalg.norm(E, axis=1)
    strong = np.argsort(-norms)[:12]
    weak = np.argsort(norms)[:12]
    log("Más fuertes (||e|| alta):")
    for i in strong:
        log(f"  {norms[i]:7.3f}  id={i:5d}  {id2tok.get(int(i), '?')!r}")
    log("Más débiles (||e|| baja):")
    for i in weak:
        log(f"  {norms[i]:7.3f}  id={i:5d}  {id2tok.get(int(i), '?')!r}")

    # save artifacts
    np.save(OUT / "token_emb_2048.npy", E)
    stats = {
        "shape": list(E.shape),
        "pca": {
            "var_top1": float(var[0]),
            "var_top10": float(cum[9]),
            "var_top50": float(cum[49]),
            "var_top100": float(cum[99]),
            "dims_50": d50,
            "dims_90": d90,
            "dims_99": d99,
            "anisotropy": anisotropy,
        },
        "poles": pole_rows,
        "clusters": {
            k: {"tokens": v["tokens"], "neighbors": [(t, s) for _i, t, s in v["neighbors"]]}
            for k, v in cluster_info.items()
        },
        "centroid_matrix": {"names": names, "cos": mat},
    }
    with open(OUT / "space_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(OUT / "space_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    log(f"\nGuardado en {OUT}")


if __name__ == "__main__":
    main()
