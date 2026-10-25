# -*- coding: utf-8 -*-
"""
Emotion circuits core pipeline (model-agnostic, LLaMA architecture).

Adaptation for TinyLlama-1.1B of the method in
"Do LLMs Feel? Emotion Circuits Discovery and Control" (arXiv:2510.11328),
as described in https://x.com/thesupermannx/status/2113933324775489904

Stages implemented here (original implementation, GPL-free):
  2. Emotion direction extraction  -> compute_emotion_directions()
  4. Local component identification -> compute_component_contributions()
  6. Global circuit integration     -> integrate_circuits()
  7. Circuit-based modulation       -> EmotionCircuitInjector

Stage 1 (elicited generation) and stage 3/5 helpers live in run_extraction.py.
"""

import json
import numpy as np
import torch

EMOTIONS = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------

def capture_last_token_hidden(model, input_ids):
    """Last-token residual stream per layer. Returns (n_layers, d_model) float32.

    hidden_states[0] is the embedding output; hidden_states[l+1] is the output
    of decoder layer l, so we drop index 0.
    """
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hs = out.hidden_states[1:]
    return np.stack([h[0, -1].float().cpu().numpy() for h in hs])


def capture_components(model, input_ids):
    """Capture per-head attention outputs and per-neuron MLP activations at the
    last token, for one forward pass.

    Returns (attn_out, mlp_act):
      attn_out: (n_layers, n_heads * head_dim) concatenated head outputs (o_proj input)
      mlp_act:  (n_layers, intermediate) MLP hidden activation act(gate(x)) * up(x)
    """
    n_layers = model.config.num_hidden_layers
    attn_out = [None] * n_layers
    mlp_act = [None] * n_layers
    handles = []

    def mk_attn_hook(idx):
        def hook(module, inputs):
            attn_out[idx] = inputs[0][0, -1].detach().float().cpu().numpy()
        return hook

    def mk_mlp_hook(idx):
        def hook(module, inputs):
            x = inputs[0][:, -1:]
            a = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
            mlp_act[idx] = a[0, 0].detach().float().cpu().numpy()
        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk_attn_hook(i)))
        handles.append(layer.mlp.register_forward_pre_hook(mk_mlp_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    return np.stack(attn_out), np.stack(mlp_act)


# ---------------------------------------------------------------------------
# Stage 2: emotion directions
# ---------------------------------------------------------------------------

def compute_emotion_directions(hidden_by_group, n_layers, d_model):
    """Context-agnostic emotion directions from grouped activations.

    hidden_by_group: list of groups (one per skeleton). Each group is a dict
      emotion -> (n_layers, d_model) last-token hidden state.
    Following the paper: within each group subtract the group mean
    (de-contextualization), average the centered vectors across groups to get
    the emotion centroid, subtract the global centroid mean, then L2-normalize.

    Returns dirs: (6, n_layers, d_model), unit norm per (emotion, layer).
    """
    centroids = {e: np.zeros((n_layers, d_model), np.float64) for e in EMOTIONS}
    n_used = 0
    for group in hidden_by_group:
        if not all(e in group for e in EMOTIONS):
            continue
        group_mean = np.mean([group[e] for e in EMOTIONS], axis=0)
        for e in EMOTIONS:
            centroids[e] += group[e] - group_mean
        n_used += 1
    if n_used == 0:
        raise ValueError("no complete groups: every group needs all 6 emotions")
    for e in EMOTIONS:
        centroids[e] /= n_used
    global_mean = np.mean([centroids[e] for e in EMOTIONS], axis=0)
    dirs = np.zeros((len(EMOTIONS), n_layers, d_model), np.float32)
    for i, e in enumerate(EMOTIONS):
        v = (centroids[e] - global_mean).astype(np.float32)
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        dirs[i] = v / np.maximum(norms, 1e-12)
    return dirs


# ---------------------------------------------------------------------------
# Stage 4: local component identification (analytical decomposition)
# ---------------------------------------------------------------------------

def compute_component_contributions(model, records, dirs, n_layers):
    """Rank attention heads and MLP neurons by their contribution to each
    emotion direction.

    records: list of groups; each group is dict emotion -> (attn_out, mlp_act)
      with shapes (n_layers, n_heads*head_dim) and (n_layers, intermediate).
    dirs: (6, n_layers, d_model) from compute_emotion_directions.

    Head score (layer l, head h, emotion e):
      mean over groups of (attn_e - attn_group_mean)[segment_h] . dirs[e, l, segment_h]
    Neuron score (layer l, neuron j, emotion e):
      mean over groups of (act_e - act_group_mean)[j] * (W_down[:, j] . dirs[e, l])

    Returns (head_scores, neuron_scores):
      head_scores:   (6, n_layers, n_heads)
      neuron_scores: (6, n_layers, intermediate)
    """
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = d_model // n_heads
    intermediate = model.config.intermediate_size
    down_w = [layer.mlp.down_proj.weight.detach().float().cpu().numpy()  # (d_model, intermediate)
              for layer in model.model.layers]

    head_scores = np.zeros((len(EMOTIONS), n_layers, n_heads), np.float64)
    neuron_scores = np.zeros((len(EMOTIONS), n_layers, intermediate), np.float64)
    n_used = 0
    for group in records:
        if not all(e in group for e in EMOTIONS):
            continue
        attn_mean = np.mean([group[e][0] for e in EMOTIONS], axis=0)
        mlp_mean = np.mean([group[e][1] for e in EMOTIONS], axis=0)
        for ei, e in enumerate(EMOTIONS):
            attn_diff = group[e][0] - attn_mean          # (L, H*Dh)
            mlp_diff = group[e][1] - mlp_mean            # (L, I)
            for l in range(n_layers):
                seg = attn_diff[l].reshape(n_heads, head_dim)
                dseg = dirs[ei, l].reshape(n_heads, head_dim)
                head_scores[ei, l] += (seg * dseg).sum(axis=1)
                proj = down_w[l].T @ dirs[ei, l]         # (intermediate,): column j of W_down is the d_model vector of neuron j
                neuron_scores[ei, l] += mlp_diff[l] * proj
        n_used += 1
    if n_used == 0:
        raise ValueError("no complete groups for component analysis")
    return head_scores / n_used, neuron_scores / n_used


# ---------------------------------------------------------------------------
# Stage 6: global circuit integration
# ---------------------------------------------------------------------------

def integrate_circuits(head_scores, neuron_scores, mass_threshold=0.9, max_per_layer=32):
    """Select, per emotion and layer, the heads and neurons that explain
    `mass_threshold` of the positive contribution mass (paper: causal
    importance allocation).

    Returns circuits: dict emotion -> {"layers": [{"layer", "heads", "neurons"}]}
    """
    circuits = {}
    for ei, e in enumerate(EMOTIONS):
        layers = []
        for l in range(head_scores.shape[1]):
            hs = np.clip(head_scores[ei, l], 0, None)
            ns = np.clip(neuron_scores[ei, l], 0, None)
            heads = _top_mass(hs, mass_threshold, max_per_layer)
            neurons = _top_mass(ns, mass_threshold, max_per_layer * 16)
            if heads or neurons:
                layers.append({"layer": int(l), "heads": heads, "neurons": neurons})
        circuits[e] = {"layers": layers}
    return circuits


def _top_mass(scores, threshold, cap):
    total = scores.sum()
    if total <= 0:
        return []
    order = np.argsort(scores)[::-1]
    cum = np.cumsum(scores[order]) / total
    k = int(np.searchsorted(cum, threshold) + 1)
    return sorted(int(i) for i in order[: min(k, cap)])


def compute_diff_vectors(records, n_layers, d_model, intermediate):
    """Stage 5: per-layer diff vectors used at modulation time.

    head_diffs[e]:  (n_layers, n_heads*head_dim) mean emotion-minus-group attn output
    neuron_diffs[e]: (n_layers, intermediate)   mean emotion-minus-group mlp activation
    """
    head_diffs = {e: np.zeros((n_layers, d_model), np.float32) for e in EMOTIONS}
    neuron_diffs = {e: np.zeros((n_layers, intermediate), np.float32) for e in EMOTIONS}
    n_used = 0
    for group in records:
        if not all(e in group for e in EMOTIONS):
            continue
        attn_mean = np.mean([group[e][0] for e in EMOTIONS], axis=0)
        mlp_mean = np.mean([group[e][1] for e in EMOTIONS], axis=0)
        for e in EMOTIONS:
            head_diffs[e] += (group[e][0] - attn_mean).astype(np.float32)
            neuron_diffs[e] += (group[e][1] - mlp_mean).astype(np.float32)
        n_used += 1
    if n_used == 0:
        raise ValueError("no complete groups for diff vectors")
    for e in EMOTIONS:
        head_diffs[e] /= n_used
        neuron_diffs[e] /= n_used
    return head_diffs, neuron_diffs


# ---------------------------------------------------------------------------
# Stage 7: circuit-based modulation
# ---------------------------------------------------------------------------

class EmotionCircuitInjector:
    """Inject an emotion circuit during generation.

    - o_proj pre-hook: add scale * masked head diff (only selected heads) at the
      last position.
    - mlp pre-hook: recompute the MLP, adding scale * neuron diff at the
      selected neurons of the hidden activation, and return the modified
      down_proj output (same trick as the paper's quick_start).
    """

    def __init__(self, model, emotion, circuit, head_diffs, neuron_diffs, scale=1.0):
        self.model = model
        self.emotion = emotion
        self.scale = scale
        self.handles = []
        self.head_sel = {}
        self.neuron_sel = {}
        for info in circuit["layers"]:
            l = info["layer"]
            if info["heads"]:
                self.head_sel[l] = info["heads"]
            if info["neurons"]:
                self.neuron_sel[l] = info["neurons"]
        self.head_diffs = head_diffs[emotion]
        self.neuron_diffs = neuron_diffs[emotion]
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads

    def _head_mask(self, layer_idx, device, dtype):
        mask = torch.zeros(self.n_heads * self.head_dim, device=device, dtype=dtype)
        for h in self.head_sel[layer_idx]:
            mask[h * self.head_dim:(h + 1) * self.head_dim] = 1.0
        return mask

    def _hook_o_proj(self, layer_idx):
        def hook(module, inputs):
            x = inputs[0]
            diff = torch.from_numpy(self.head_diffs[layer_idx]).to(x.device, x.dtype)
            diff = diff * self._head_mask(layer_idx, x.device, x.dtype)
            x = x.clone()
            x[:, -1] = x[:, -1] + self.scale * diff
            return (x,)
        return hook

    def _hook_mlp(self, layer_idx):
        def hook(module, inputs, _output):
            x = inputs[0]
            a = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
            diff = torch.from_numpy(self.neuron_diffs[layer_idx]).to(a.device, a.dtype)
            idx = torch.tensor(self.neuron_sel[layer_idx], device=a.device, dtype=torch.long)
            a = a.clone()
            a[:, -1, idx] = a[:, -1, idx] + self.scale * diff[idx]
            return module.down_proj(a)
        return hook

    def register(self):
        for i, layer in enumerate(self.model.model.layers):
            if i in self.head_sel:
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook_o_proj(i)))
            if i in self.neuron_sel:
                self.handles.append(layer.mlp.register_forward_hook(self._hook_mlp(i)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class ResidualDirectionSteerer:
    """Simpler ablation of the method: add the scaled emotion direction directly
    to the residual stream (last token) at its top-|direction| layers. Useful to
    reproduce the paper's steering-based baseline (their stage 3)."""

    def __init__(self, model, emotion, dirs, scale=1.0, top_layers=5):
        ei = EMOTIONS.index(emotion)
        v = dirs[ei]                                    # (L, d_model)
        order = np.argsort(np.linalg.norm(v, axis=-1))[::-1][:top_layers]
        self.vectors = {int(l): torch.from_numpy(v[int(l)]) for l in order}
        self.model = model
        self.scale = scale
        self.handles = []

    def _hook(self, vec):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            hs[:, -1] = hs[:, -1] + self.scale * vec.to(hs.device, hs.dtype)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook

    def register(self):
        for l, vec in self.vectors.items():
            self.handles.append(self.model.model.layers[l].register_forward_hook(self._hook(vec)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def save_outputs(path_prefix, dirs, head_diffs, neuron_diffs, circuits):
    np.savez_compressed(
        path_prefix + ".npz",
        directions=dirs,
        **{f"head_diff/{e}": head_diffs[e] for e in EMOTIONS},
        **{f"neuron_diff/{e}": neuron_diffs[e] for e in EMOTIONS},
    )
    with open(path_prefix + "_circuits.json", "w") as f:
        json.dump(circuits, f, indent=1)


def load_outputs(path_prefix):
    data = np.load(path_prefix + ".npz")
    dirs = data["directions"]
    head_diffs = {e: data[f"head_diff/{e}"] for e in EMOTIONS}
    neuron_diffs = {e: data[f"neuron_diff/{e}"] for e in EMOTIONS}
    with open(path_prefix + "_circuits.json") as f:
        circuits = json.load(f)
    return dirs, head_diffs, neuron_diffs, circuits


# ---------------------------------------------------------------------------
# Ablation of selected components (mechanics check, not causal proof)
# ---------------------------------------------------------------------------

class EmotionCircuitAblator:
    """Zero out the circuit's selected heads/neurons at the last position.

    Complement of EmotionCircuitInjector for mechanics checks. NOTE: measuring
    the projection onto the same direction that is injected/ablated is
    self-referential - it verifies that hooks move what they target, not that
    the components cause emotional expression. Behavioral evidence (held-out
    scenarios, matched-norm controls, external labeling) is still pending.
    """

    def __init__(self, model, circuit):
        self.model = model
        self.head_sel = {}
        self.neuron_sel = {}
        for info in circuit["layers"]:
            if info["heads"]:
                self.head_sel[info["layer"]] = info["heads"]
            if info["neurons"]:
                self.neuron_sel[info["layer"]] = info["neurons"]
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.n_heads
        self.handles = []

    def _hook_o_proj(self, layer_idx):
        def hook(module, inputs):
            x = inputs[0].clone()
            for h in self.head_sel[layer_idx]:
                x[:, -1, h * self.head_dim:(h + 1) * self.head_dim] = 0.0
            return (x,)
        return hook

    def _hook_mlp(self, layer_idx):
        def hook(module, inputs, _output):
            x = inputs[0]
            a = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
            a = a.clone()
            a[:, -1, self.neuron_sel[layer_idx]] = 0.0
            return module.down_proj(a)
        return hook

    def register(self):
        for i, layer in enumerate(self.model.model.layers):
            if i in self.head_sel:
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook_o_proj(i)))
            if i in self.neuron_sel:
                self.handles.append(layer.mlp.register_forward_hook(self._hook_mlp(i)))
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def projection_scores(hidden, dirs):
    """Projection of a per-layer hidden state onto each emotion direction,
    averaged over layers. hidden: (n_layers, d_model); dirs: (6, L, D).
    Returns (6,) scores; argmax is the predicted emotion."""
    return np.stack([(dirs[i] * hidden).sum(axis=-1).mean() for i in range(dirs.shape[0])])
