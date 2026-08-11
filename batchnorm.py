"""makemore 3: module-ized layers, BatchNorm, and the activation diagnostics.

    python batchnorm.py

Builds a `Layer -> BatchNorm1d -> Tanh` stack on top of the embedding lookup,
trains it, then plots the two diagnostics from the lecture: tanh activation
histograms (looking for saturation at +-1) and their gradient histograms
(looking for the collapse toward zero).  Converted from BatchNorm_xw2.ipynb.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from MLP import MLP  # reused only for makeXY / build_data / vocabulary


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------

class Layer:
    def __init__(self, fan_in, fan_out, bias=True):
        # randn has std 1; a pre-activation is a sum of fan_in such terms, so
        # its std would be sqrt(fan_in).  Dividing by sqrt(fan_in) keeps the
        # variance at 1 layer after layer.
        self.W = torch.randn((fan_in, fan_out)) / (fan_in ** 0.5)
        self.b = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.W
        if self.b is not None:
            self.out = self.out + self.b
        return self.out

    def parameters(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class BatchNorm1d:
    def __init__(self, fan_out, momentum=1e-2, eps=1e-5, training=True):
        self.gamma = torch.ones(fan_out)
        self.beta = torch.zeros(fan_out)
        self.momentum = momentum
        self.eps = eps
        self.training = training
        # running statistics, updated during training, used at inference
        self.running_mean = torch.zeros((1, fan_out))
        self.running_var = torch.ones((1, fan_out))

    def __call__(self, x):
        if self.training:
            xmean = torch.mean(x, dim=0, keepdim=True)
            xvar = torch.var(x, dim=0, keepdim=True)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        else:
            xmean = self.running_mean
            xvar = self.running_var
        self.out = self.gamma * (x - xmean) / torch.sqrt(xvar + self.eps) + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

block_size = 3           # characters of context
embd_dim   = 4           # embedding dimension per character
h_dim      = 100         # hidden units
minb_size  = 32          # minibatch size
numRun     = 50000       # training steps
lr         = 0.1         # learning rate, first phase
lr2        = 0.01        # learning rate after lr_decay_step
lr_decay_step = 100000

seed       = 2147483647
split_seed = 42


# ----------------------------------------------------------------------
# Data and model
# ----------------------------------------------------------------------

if __name__ == "__main__":
    words = open("names.txt", "r").read().splitlines()

    model = MLP(words, block_size, embd_dim, h_dim, seed=seed)
    model.makeXY()
    model.build_data(split_seed=split_seed)

    Xdata = model.Xdic["train"]
    Ydata = model.Ydic["train"]
    volc_dim = model.volc_dim

    C = torch.randn((volc_dim, embd_dim))
    Layers = [
        Layer(embd_dim * block_size, h_dim), BatchNorm1d(h_dim), Tanh(),
        Layer(h_dim, volc_dim),
    ]

    # C must be in params too -- left out, the embeddings stay frozen at their
    # random initial values and never learn (no error is raised; the model is
    # just silently worse).
    params = [C] + [pp for ll in Layers for pp in ll.parameters()]
    for p in params:
        p.requires_grad = True
    print("parameters:", sum(p.nelement() for p in params))

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    lossi = []
    for i in range(numRun):
        ix = torch.randint(0, len(Ydata), (minb_size,))
        Xbatch, Ybatch = Xdata[ix], Ydata[ix]

        # forward
        xin = C[Xbatch].view(-1, block_size * embd_dim)
        for ll in Layers:
            xin = ll(xin)
        loss = F.cross_entropy(xin, Ybatch)
        lossi.append(loss.item())

        # backward
        for p in params:
            p.grad = None
        loss.backward()

        # update
        lr0 = lr if i < lr_decay_step else lr2
        with torch.no_grad():
            for p in params:
                p.data += -lr0 * p.grad

        if i % 10000 == 0:
            print(f"{i:6d}  |  loss = {loss.item():.4f}")

    print(f"final minibatch loss = {loss.item():.4f}")

    # ------------------------------------------------------------------
    # Diagnostics: one more forward/backward pass, retaining gradients
    # on the intermediate activations
    # ------------------------------------------------------------------

    xin = C[Xbatch].view(-1, block_size * embd_dim)
    for ll in Layers:
        xin = ll(xin)
        xin.retain_grad()          # keep .grad on non-leaf tensors
    loss = F.cross_entropy(xin, Ybatch)
    for p in params:
        p.grad = None
    loss.backward()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # activation histograms: saturated tanh piles up at +-1
    for k, ll in enumerate(Layers):
        if isinstance(ll, Tanh):
            t = ll.out
            sat = (t.abs() > 0.97).float().mean()
            print(f"tanh layer {k}: mean {t.mean():+.3f}  std {t.std():.3f}  saturated {sat:.1%}")
            hy, hx = torch.histogram(t.detach().flatten(), density=True)
            axes[0].plot(hx[:-1], hy, label=f"layer {k}  sat {sat:.0%}")
    axes[0].set_title("tanh activations")
    axes[0].legend()

    # gradient histograms: watch the spread shrink layer by layer
    for k, ll in enumerate(Layers):
        if isinstance(ll, Tanh):
            tg = ll.out.grad
            hy, hx = torch.histogram(tg.detach().flatten(), density=True)
            axes[1].plot(hx[:-1], hy, label=f"layer {k}  std {tg.std():.2e}")
    axes[1].set_title("activation gradients")
    axes[1].legend()

    axes[2].plot(lossi, linewidth=0.5, alpha=0.6)
    axes[2].set_title("minibatch loss")
    axes[2].set_xlabel("step")

    plt.tight_layout()
    plt.savefig("batchnorm_diag.png", dpi=150)
    print("\nwrote batchnorm_diag.png")
