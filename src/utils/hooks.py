"""Forward-hook infrastructure shared by the interp battery (S4 cache, S6 patch, S7/S8 steer).

This is the one piece of real infrastructure in the project: thin context managers over
PyTorch forward hooks on `model.model.layers[i]`. Every downstream interp script is a
consumer of this module, which is what keeps the components swappable.

`residual_cache` is what S4 needs and is implemented here. `patch` / `steer` land with
S6 / S7 (kept out for now so this module stays small and only ships what is tested).

torch is imported lazily inside the context managers so this file is importable on a box
without PyTorch (the pure offset logic in s4 lives there, not here).
"""

from contextlib import contextmanager


@contextmanager
def residual_cache(model, layers):
    """Capture the residual-stream output of each decoder layer in `layers`.

    Yields a dict `store` that, after the forward pass inside the `with` block, holds
    `{layer_idx: tensor[B, T, D]}` -- the hidden states *after* that decoder block (the
    residual stream). Tensors are detached but stay on the model's device; the caller
    indexes the positions it wants and moves the result to CPU.

    A Gemma-2 / Llama decoder layer returns a tuple whose first element is the hidden
    state; we take `out[0]` (and tolerate a bare-tensor return for safety).
    """
    store = {}
    handles = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            store[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for i in layers:
        handles.append(model.model.layers[i].register_forward_hook(make_hook(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
