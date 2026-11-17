
# ===== cell 1 =====
forward({"type": "score", "model": "original", "split": "dev"})


# ===== cell 2 =====
forward({"type": "teacher_forced", "model": "original", "texts": ["en bil är", "ett hus är"], "return": {"hidden_states": True}})


# ===== cell 3 =====
import numpy as np
H = np.load("/work/fwd_2/hidden_states.npy")
bil_hidden = H[0, 0, 1]  # layer 0, token "bil"
hus_hidden = H[1, 0, 1]  # layer 0, token "hus"
diff = bil_hidden - hus_hidden
print(f"Gender difference in layer 0: {diff[:4]} (first 4 dims)")


# ===== cell 4 =====
in_proj_weight = load("blocks.layers.0.self_attn.in_proj_weight")
edited_in_proj = in_proj_weight.copy()
edited_in_proj[0:32, 0] += 0.1  # Add 0.1 to first row of query part
save_edit("blocks.layers.0.self_attn.in_proj_weight", edited_in_proj)


# ===== cell 5 =====
forward({"type": "score", "model": "both", "split": "dev"})


# ===== cell 6 =====
in_proj_weight = load("blocks.layers.0.self_attn.in_proj_weight")
edited_in_proj = in_proj_weight.copy()
edited_in_proj[0:32, 0] += 0.3  # Add 0.3 to first row of query part
save_edit("blocks.layers.0.self_attn.in_proj_weight", edited_in_proj)


# ===== cell 7 =====
forward({"type": "score", "model": "both", "split": "dev"})


# ===== cell 8 =====
forward({"type": "teacher_forced", "model": "edited", "texts": ["en bil är", "ett hus är"], "return": {"hidden_states": True}})


# ===== cell 9 =====
forward({"type": "score", "model": "both", "split": "dev"})


# ===== cell 10 =====
forward({"type": "teacher_forced", "model": "edited", "texts": ["en bil är", "ett hus är"], "return": {"hidden_states": True}})

