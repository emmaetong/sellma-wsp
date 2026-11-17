
# ===== cell 1 =====
forward({"type": "score", "model": "original", "split": "dev"})


# ===== cell 2 =====
forward({"type": "teacher_forced", "model": "original", "texts": ["en bil är", "ett hus är"], "return": {"hidden_states": True, "attention": True}})


# ===== cell 3 =====
import numpy as np
H = np.load("/work/fwd_2/hidden_states.npy")
meta = np.load("/work/fwd_2/meta.json")
bil_state = H[0, 1, 1, :]  # layer 1, token 1 (bil)
hus_state = H[1, 1, 1, :]  # layer 1, token 1 (hus)
print(f"Difference in hidden states: {np.linalg.norm(bil_state - hus_state):.4f}")


# ===== cell 4 =====
import numpy as np
H = np.load("/work/fwd_2/hidden_states.npy", allow_pickle=True)
meta = np.load("/work/fwd_2/meta.json")
bil_state = H[0, 1, 1, :]  # layer 1, token 1 (bil)
hus_state = H[1, 1, 1, :]  # layer 1, token 1 (hus)
print(f"Difference in hidden states: {np.linalg.norm(bil_state - hus_state):.4f}")


# ===== cell 5 =====
import pickle
H = pickle.load(open("/work/fwd_2/hidden_states.npy", "rb"))
meta = np.load("/work/fwd_2/meta.json")
bil_state = H[0, 1, 1, :]  # layer 1, token 1 (bil)
hus_state = H[1, 1, 1, :]  # layer 1, token 1 (hus)
print(f"Difference in hidden states: {np.linalg.norm(bil_state - hus_state):.4f}")


# ===== cell 6 =====
save_edit("blocks.layers.0.norm1.weight", 1.1 * load("blocks.layers.0.norm1.weight"))


# ===== cell 7 =====
import numpy as np
original = load("blocks.layers.0.norm1.weight")
edited = np.copy(original)
edited[0] += 0.1
save_edit("blocks.layers.0.norm1.weight", edited)


# ===== cell 8 =====
forward({"type": "score", "model": "both", "split": "dev"})

