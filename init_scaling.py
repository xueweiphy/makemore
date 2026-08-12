"""Break the initialization on purpose and watch the statistics fail.

    python init_scaling.py

A deep (Linear -> Tanh)^L stack with NO BatchNorm.  Set w_scale below to
1.0, 10.0 or 0.1 and compare the diagnostics at initialization:

    w_scale = 10   activations saturate at +-1; gradients amplified toward
                   the input (early layers broad, deep layers concentrated)
    w_scale = 0.1  activations collapse to 0, sharper with depth; gradients
                   vanish toward the input
    w_scale = 1    layer-independent histograms -- criticality

Derivations in exec/Init_scaling.tex.  Converted from BatchNorm_xw2.ipynb.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from MLP import MLP  # reused for vocabulary and train/dev/test split


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------

class Linear:
    def __init__(self, fan_in, fan_out, bias=True, gain=1.0):
        # /sqrt(fan_in): a pre-activation is a sum of fan_in independent
        # terms, so its std would grow as sqrt(fan_in) -- random-walk
        # scaling.  gain carries both the principled tanh factor (5/3)
        # and the deliberate sabotage (w_scale).
        self.W = torch.randn((fan_in, fan_out)) * gain / (fan_in ** 0.5)
        self.b = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.W
        if self.b is not None:
            self.out = self.out + self.b
        return self.out

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

block_size = 3
embd_dim   = 4
h_dim      = 100
minb_size  = 32
numRun     = 0            # 0 = diagnostics at initialization only
lr         = 0.1

gain       = 5. / 3.      # tanh gain (Kaiming), the principled knob
w_scale    = 10.0         # the sabotage knob: try 1.0, 10.0, 0.1
last_squash = 0.1         # last-layer factor: small logits, honest initial loss

seed       = 2147483647
split_seed = 42


if __name__ == "__main__":
    words = open("names.txt", "r").read().splitlines()

    model = MLP(words, block_size, embd_dim, h_dim, seed=seed)
    model.makeXY()
    model.build_data(split_seed=split_seed)
    Xdata, Ydata = model.Xdic["train"], model.Ydic["train"]
    volc_dim = model.volc_dim

    C = torch.randn((volc_dim, embd_dim))
    g = gain * w_scale
    Layers = [
        Linear(embd_dim * block_size, h_dim, gain=g), Tanh(),
        Linear(h_dim, h_dim, gain=g), Tanh(),
        Linear(h_dim, h_dim, gain=g), Tanh(),
        Linear(h_dim, h_dim, gain=g), Tanh(),
        Linear(h_dim, volc_dim),
    ]
    with torch.no_grad():
        Layers[-1].W *= last_squash

    params = [C] + [pp for ll in Layers for pp in ll.parameters()]
    for p in params:
        p.requires_grad = True
    print(f"parameters {sum(p.nelement() for p in params)}   "
          f"gain {gain:.3f} x w_scale {w_scale} = {g:.3f}")

    # ------------------------------------------------------------------
    # Optional training (numRun = 0 skips straight to diagnostics)
    # ------------------------------------------------------------------

    lossi = []
    for i in range(numRun):
        ix = torch.randint(0, len(Ydata), (minb_size,))
        xin = C[Xdata[ix]].view(-1, block_size * embd_dim)
        for ll in Layers:
            xin = ll(xin)
        loss = F.cross_entropy(xin, Ydata[ix])
        lossi.append(loss.item())

        for p in params:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in params:
                p.data += -lr * p.grad
        if i % 10000 == 0:
            print(f"{i:6d}  |  loss = {loss.item():.4f}")

    # ------------------------------------------------------------------
    # Diagnostic forward/backward pass, gradients retained
    # ------------------------------------------------------------------

    ix = torch.randint(0, len(Ydata), (minb_size,))
    xin = C[Xdata[ix]].view(-1, block_size * embd_dim)
    for ll in Layers:
        xin = ll(xin)
        xin.retain_grad()
    loss = F.cross_entropy(xin, Ydata[ix])
    print(f"loss at step {numRun}: {loss.item():.4f}   (uniform baseline 3.2958)")

    for p in params:
        p.grad = None
    loss.backward()

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    for k, ll in enumerate(ll2 for ll2 in Layers if isinstance(ll2, Tanh)):
        t = ll.out
        sat = (t.abs() > 0.97).float().mean()
        print(f"tanh {k}: act mean {t.mean():+.3f}  std {t.std():.3f}  "
              f"sat {sat:5.1%}   grad std {t.grad.std():.3e}")
        hy, hx = torch.histogram(t.detach().flatten(), density=True)
        axes[0].plot(hx[:-1], hy, label=f"tanh {k}  sat {sat:.0%}")
        hy, hx = torch.histogram(t.grad.detach().flatten(), density=True)
        axes[1].plot(hx[:-1], hy, label=f"tanh {k}  std {t.grad.std():.1e}")

    axes[0].set_title(f"activations  (w_scale = {w_scale})")
    axes[0].legend()
    axes[1].set_title("activation gradients")
    axes[1].legend()

    plt.tight_layout()
    fname = f"init_scaling_{w_scale}.png"
    plt.savefig(fname, dpi=150)
    print(f"\nwrote {fname}")
