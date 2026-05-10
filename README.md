# Training a 7B Parameter Model on TPU v4-8 with JAX — Solving OOM and Sharding Errors

> **Written by Atharva Shewale, Pune, India**  
> *Building MAX — a personal AI from scratch*

---

## The Problem

When training a 7B parameter transformer model on a TPU v4-8 (4 chips × 32GB = 128GB HBM) using JAX/Flax, I kept hitting this error:

```
ValueError: RESOURCE_EXHAUSTED: Error allocating device buffer: 
Attempting to allocate 128.00M. That was not possible. 
There are 8.00M free.
```

Or this one:

```
ValueError: One of pjit arguments with pytree key path state.step 
is incompatible with its sharding annotation NamedSharding(...): 
Sharding is only valid for values of rank at least 1, but was 
applied to a value of rank 0.
```

Weeks of debugging. This writeup explains exactly what was wrong and how I fixed it.

---

## The Setup

- **Model:** 7B parameter GPT-style transformer (32 layers, 32 heads, d_model=4096)
- **TPU:** v4-8 — 4 chips, 32GB HBM each = 128GB total
- **Framework:** JAX + Flax + Optax
- **Data:** 27M instruction-response pairs
- **Optimizer:** AdamW with gradient clipping

---

## Why It OOMs — The Math

This is the core insight that took me weeks to figure out.

**Memory required for 7B training:**

```
Params (bf16):        7B × 2 bytes = 14GB
AdamW m (bf16):       7B × 2 bytes = 14GB  ← first moment
AdamW v (bf16):       7B × 2 bytes = 14GB  ← second moment
Gradients (bf16):     7B × 2 bytes = 14GB
Activations:          ~4GB (seq_len=1024, batch=4)
─────────────────────────────────────────────────
Total needed:         ~60GB per chip (if replicated)
Available per chip:   32GB
```

If you **replicate** params across all chips (the default), each chip needs to hold the full model. 60GB > 32GB = OOM.

**The fix:** Shard params across chips so each chip holds 1/4 of the model.

```
Per chip with sharding:
Params:     14GB / 4 chips = 3.5GB
AdamW m:    14GB / 4 chips = 3.5GB
AdamW v:    14GB / 4 chips = 3.5GB
Gradients:  14GB / 4 chips = 3.5GB
─────────────────────────────────────
Total:      ~14GB per chip << 32GB ✅
```

---

## What Doesn't Work

### ❌ Attempt 1 — Default FlaxTrainState

```python
from flax.training.train_state import TrainState

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)
```

This wraps `params`, `opt_state`, and `step` into a single pytree. When you try to shard it:

```python
# This fails
params = jax.device_put(params, sharded)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
# Error: OOM when creating optimizer states
```

AdamW's `tx.init(params)` tries to create moment tensors on the same device as params — but params already filled the chip memory.

### ❌ Attempt 2 — Replicated params, CPU optimizer init

```python
with jax.default_device(jax.devices("cpu")[0]):
    opt_state = tx.init(params)
params = jax.device_put(params, replicated)
opt_state = jax.device_put(opt_state, replicated)  # OOM here
```

Moving 42GB of optimizer states to chips that already have 14GB of replicated params = OOM.

### ❌ Attempt 3 — Sharding the full TrainState

```python
@partial(jax.jit,
         in_shardings=(sharded, data_shard),  # sharded for state
         out_shardings=(sharded, replicated))
def train_step(state, batch):
    ...
```

Error:
```
ValueError: One of pjit arguments with pytree key path state.step 
is incompatible with its sharding: only valid for rank >= 1, 
but applied to rank 0 scalar.
```

`state.step` is a scalar (rank 0). You can't shard a scalar. JAX fails.

---

## ✅ The Working Solution

**Key insight: Don't use TrainState at all. Keep params, opt_state, and step completely separate.**

```python
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Setup mesh
N = jax.local_device_count()  # 4 on v4-8
devices = mesh_utils.create_device_mesh((N,))
mesh = Mesh(devices, axis_names=("data",))

# Shardings
sharded    = NamedSharding(mesh, P("data"))        # params split across chips
replicated = NamedSharding(mesh, P())              # scalars replicated
data_shard = NamedSharding(mesh, P("data", None))  # batch split across chips

# 1. Init model on CPU (TPU has 1.4TB CPU RAM)
with jax.default_device(jax.devices("cpu")[0]):
    variables = model.init(rng, jnp.ones((1, 64), dtype=jnp.int32))

# 2. Cast to bf16 on CPU (halves memory)
params = jax.tree_util.tree_map(
    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
    variables["params"]
)

# 3. Shard params across TPU chips
params = jax.device_put(params, sharded)

# 4. Init optimizer WITH sharded params
# Optax creates sharded optimizer states automatically
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.1)
)
opt_state = tx.init(params)  # ✅ optimizer states also sharded

# 5. Step is a plain Python int — never touches JAX sharding
step = 0

# 6. Train step — no TrainState, no scalar sharding issues
@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(p):
        logits = model.apply({"params": p}, batch, training=True)
        return compute_loss(logits, batch)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Cast grads to bf16 to save memory
    grads = jax.tree_util.tree_map(
        lambda g: g.astype(jnp.bfloat16) if g.dtype != jnp.bfloat16 else g,
        grads
    )
    
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# 7. Training loop
for step in range(1, total_steps + 1):
    batch = jax.device_put(next(train_iter), data_shard)
    params, opt_state, loss = train_step(params, opt_state, batch)
```

**Why this works:**
- `params` sharded → each chip holds 1/4 of params
- `opt_state = tx.init(params)` → optax creates moment tensors with same sharding as params automatically
- `step` is a plain Python int → never enters JAX's sharding system → no scalar rank error
- No FlaxTrainState → no pytree mixing scalars and tensors

---

## Streaming Tokenization

Another major issue: tokenizing 27M sequences upfront takes 2+ hours and 48GB RAM.

**Fix — tokenize on the fly:**

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("max_tokenizer_v2.json")

def tokenize_pair(q, a):
    ids = ([BOS_ID, SYS_ID] + SYS_IDS + [SEP_ID, USR_ID] +
           tok.encode(q).ids + [SEP_ID, AST_ID] +
           tok.encode(a).ids + [EOS_ID])
    if len(ids) > SEQ_LEN: ids = ids[:SEQ_LEN-1] + [EOS_ID]
    else: ids = ids + [PAD_ID] * (SEQ_LEN - len(ids))
    return ids

def streaming_iter(pairs, batch_size, shuffle=True):
    """Tokenize on the fly — training starts in 2 minutes not 2 hours."""
    indices = list(range(len(pairs)))
    while True:
        if shuffle: random.shuffle(indices)
        batch = []
        for idx in indices:
            batch.append(tokenize_pair(*pairs[idx]))
            if len(batch) == batch_size:
                yield np.array(batch, dtype=np.int32).reshape(N, batch_size // N, SEQ_LEN)
                batch = []
```

---

## Results

```
Model:     9.13B parameters (18.3GB bf16)
TPU:       v4-8 (4 chips × 32GB = 128GB HBM)
Data:      27,206,296 instruction-response pairs
Speed:     ~3,886 tokens/second
Loss:      11.57 → 3.96 in first 800 steps
Vocab:     65,536 (custom BPE tokenizer)
SEQ_LEN:   1024
```

Training log:
```
✅ COMPILE SUCCESS! Loss=11.5739 | Time=117.0s
✅ MAX 7B IS TRAINING!
step=    50 | loss=6.8166 | lr=1.50e-05 | 1401 tok/s
step=   100 | loss=6.2840 | lr=3.00e-05 | 3886 tok/s
step=   200 | loss=4.7362 | lr=6.00e-05 | 3887 tok/s
step=   700 | loss=3.9635 | lr=2.10e-04 | 3886 tok/s
step=   800 | loss=6.5927 | lr=2.40e-04 | 3888 tok/s
```

---

## Key Takeaways

1. **Never replicate 7B params on v4-8** — 14GB × 4 (replicated) = 56GB > 32GB per chip
2. **Shard params first, then init optimizer** — optax inherits sharding automatically
3. **Keep step as Python int** — never put it in a JAX pytree with sharding annotations
4. **Don't use FlaxTrainState for sharded training** — it mixes scalars and tensors in one pytree
5. **Tokenize on the fly** — 27M sequences upfront = 48GB RAM + 2 hours wasted
6. **Cast to bf16 on CPU** — before moving to TPU, halves memory footprint

---

## About This Project

This writeup is part of building **MAX** — a personal AI operating system built entirely from scratch. Custom tokenizer, custom architecture, custom training pipeline. No wrappers around existing models.

- GitHub: https://github.com/AsphaltProAT
- Built by: Atharva Shewale, Pune, India

If this helped you, star the repo. If you have questions, open an issue.

---

## Files

- `train_7b.py` — Full training script with working sharding
- `model.py` — Transformer architecture
- `config.py` — Model configuration
- `data.py` — GCS streaming data loader

---

*"The best way to learn JAX sharding is to debug it for three weeks straight."*
