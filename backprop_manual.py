"""makemore 4 ("backprop ninja"): the backward pass of the BatchNorm MLP by hand.

    python backprop_manual.py

Exercise 1  one forward pass decomposed into ~20 named intermediate tensors,
            then the backward pass written manually for each and checked
            against autograd with cmp().  Every line prints exact: True
            (dlogit_maxes is zero by the shift-invariance of softmax; the
            manual row-sum form is bit-exact).
Exercise 2  the cross-entropy backward collapsed to one line, (P - Y)/n.
Exercise 3  the BatchNorm backward collapsed to one expression, with no
            intermediate tensors at all.
Exercise 4  train the network using only the manual backward pass -- autograd
            is switched off for the whole loop.  max_steps below defaults to a
            1000-step smoke test; raise it to 200000 for the real run.

The fused forms of exercises 2 and 3 are algebraically identical to the chains
they replace but are *not* bit-exact against autograd: different summation
orders, same limit.  They print approximate: True with maxdiff ~1e-9.

Bit-exactness in exercise 1 is a property of the torch build, not of the
algebra: on newer versions the tanh backward uses a different kernel, so every
line from dhpreact onward drops to approximate: True at ~1e-9 while staying
exactly right.  Only a line that fails `approximate` is a real error.

Derivations in index notation: exec/Backprop_manual.tex.
Converted from backprop_xw.ipynb.
"""

import torch
import torch.nn.functional as F

try:
    from MLP import MLP
except ImportError:
    from .MLP import MLP


results = []   # (name, bit-exact, allclose) for every comparison made


def cmp(s, dt, t, exact=True):
    """Compare a manual gradient against autograd.

    exact=True  demands bit-for-bit equality (exercise 1)
    exact=False accepts allclose (the fused forms of exercises 2 and 3, which
                sum the same terms in a different order)
    """
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}")
    results.append((s, ex, app))
    return ex if exact else app


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    block_size = 3
    embd_dim   = 10
    h_dim      = 100
    minb_size  = 32
    seed       = 2147483647
    split_seed = 42

    # Exercise 4.  1000 is a smoke test: a second of training, enough to see
    # the hand-written gradients move the loss (ending near train/dev 2.43).
    # 200000 is the lecture's run -- about a minute on CPU, train 2.11 / dev
    # 2.14.  0 skips training and leaves only the gradient checks.
    max_steps     = 1000
    lr            = 0.1
    lr2           = 0.01
    lr_decay_step = 100000

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    words = open("names.txt", "r").read().splitlines()
    model = MLP(words, block_size, embd_dim, h_dim, seed=seed)
    model.makeXY()
    model.build_data(split_seed=split_seed)
    Xdata, Ydata = model.Xdic["train"], model.Ydic["train"]
    volc_dim = model.volc_dim

    # ------------------------------------------------------------------
    # Parameters (Karpathy's init: gain/sqrt(fan_in), last layer squashed)
    # ------------------------------------------------------------------

    g = torch.Generator().manual_seed(seed)
    C  = torch.randn((volc_dim, embd_dim), generator=g)
    W1 = torch.randn((embd_dim * block_size, h_dim), generator=g) * (5/3) / ((embd_dim * block_size) ** 0.5)
    b1 = torch.randn(h_dim, generator=g) * 0.1   # useless under BN -- kept to see its gradient vanish
    W2 = torch.randn((h_dim, volc_dim), generator=g) * 0.1
    b2 = torch.randn(volc_dim, generator=g) * 0.1
    bngain = torch.randn((1, h_dim)) * 0.1 + 1.0
    bnbias = torch.randn((1, h_dim)) * 0.1

    parameters = [C, W1, b1, W2, b2, bngain, bnbias]
    print("parameters:", sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True

    n = minb_size
    ix = torch.randint(0, Xdata.shape[0], (minb_size,), generator=g)
    Xb, Yb = Xdata[ix], Ydata[ix]

    # ------------------------------------------------------------------
    # Forward pass, chunked into one tensor per backward step
    # ------------------------------------------------------------------

    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)
    # Linear 1
    hprebn = embcat @ W1 + b1
    # BatchNorm
    bnmeani = 1/n * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff ** 2
    bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)   # Bessel's correction
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # Nonlinearity
    h = torch.tanh(hpreact)
    # Linear 2
    logits = h @ W2 + b2
    # Cross-entropy, decomposed
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum ** -1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()
    print(f"loss: {loss.item():.4f}")

    for p in parameters:
        p.grad = None
    for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
              norm_logits, logit_maxes, logits, h, hpreact, bnraw,
              bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
              embcat, emb]:
        t.retain_grad()
    loss.backward()

    # ------------------------------------------------------------------
    # Exercise 1: manual backward pass.  Two rules do most of the work:
    #   - the gradient of a tensor has the shape of the tensor
    #   - broadcast forward  => sum backward (and vice versa)
    # ------------------------------------------------------------------

    print("\n--- exercise 1: every intermediate gradient by hand ---")

    ok = True

    # loss = -mean of logprobs at the targets: sparse, -1/n at [beta, y_beta]
    Yb_onehot = F.one_hot(Yb, num_classes=volc_dim).float()
    dlogprobs = -1/n * Yb_onehot
    ok &= cmp("dlogprobs", dlogprobs, logprobs)

    dprobs = 1/probs * dlogprobs
    ok &= cmp("dprobs", dprobs, probs)

    # counts_sum_inv was broadcast across the row -> sum over dim 1
    dcounts_sum_inv = (counts * dprobs).sum(dim=1, keepdim=True)
    ok &= cmp("dcounts_sum_inv", dcounts_sum_inv, counts_sum_inv)

    dcounts_sum = -1/counts_sum**2 * dcounts_sum_inv
    ok &= cmp("dcounts_sum", dcounts_sum, counts_sum)

    # counts feeds probs directly AND counts_sum: two contributions, +=
    dcounts = counts_sum_inv * dprobs + dcounts_sum
    ok &= cmp("dcounts", dcounts, counts)

    dnorm_logits = counts * dcounts        # d(exp) = exp
    ok &= cmp("dnorm_logits", dnorm_logits, norm_logits)

    # Zero by the shift-invariance of softmax (gauge choice); the row-sum
    # form is what autograd computes, bit-exact at ~1e-9.
    dlogit_maxes = -dnorm_logits.sum(dim=1, keepdim=True)
    ok &= cmp("dlogit_maxes", dlogit_maxes, logit_maxes)

    # route the (null) max gradient back through the argmax positions
    max_mask = F.one_hot(logits.argmax(dim=1), num_classes=logits.shape[1]).to(logits.dtype)
    dlogits = dnorm_logits + max_mask * dlogit_maxes
    ok &= cmp("dlogits", dlogits, logits)

    # matmul backward: grad_x = grad_out @ W.T, grad_W = x.T @ grad_out
    dh = dlogits @ W2.T
    ok &= cmp("dh", dh, h)
    dW2 = h.T @ dlogits
    ok &= cmp("dW2", dW2, W2)
    db2 = dlogits.sum(dim=0)
    ok &= cmp("db2", db2, b2)

    dhpreact = (1 - h**2) * dh             # tanh'
    ok &= cmp("dhpreact", dhpreact, hpreact)

    dbngain = (bnraw * dhpreact).sum(dim=0, keepdim=True)
    ok &= cmp("dbngain", dbngain, bngain)
    dbnbias = dhpreact.sum(dim=0, keepdim=True)
    ok &= cmp("dbnbias", dbnbias, bnbias)

    dbnraw = bngain * dhpreact
    ok &= cmp("dbnraw", dbnraw, bnraw)

    dbnvar_inv = (bndiff * dbnraw).sum(dim=0, keepdim=True)
    ok &= cmp("dbnvar_inv", dbnvar_inv, bnvar_inv)

    # d/dv (v+eps)^-1/2 = -1/2 (v+eps)^-3/2 -- the eps must ride along
    dbnvar = -0.5 * (bnvar + 1e-5)**-1.5 * dbnvar_inv
    ok &= cmp("dbnvar", dbnvar, bnvar)

    # bnvar summed bndiff2 over the batch -> broadcast back to (n, h)
    dbndiff2 = (1/(n-1) * dbnvar).expand(n, -1) * torch.ones(1)
    ok &= cmp("dbndiff2", dbndiff2, bndiff2)

    # bndiff feeds bndiff2 AND bnraw: two contributions again
    dbndiff = 2 * bndiff * dbndiff2 + dbnraw * bnvar_inv
    ok &= cmp("dbndiff", dbndiff, bndiff)

    dbnmeani = -dbndiff.sum(dim=0, keepdims=True)
    ok &= cmp("dbnmeani", dbnmeani, bnmeani)

    # the batch-mean coupling: every example's hprebn entered bnmeani with 1/n
    dhprebn = dbndiff + 1/n * dbnmeani
    ok &= cmp("dhprebn", dhprebn, hprebn)

    dembcat = dhprebn @ W1.T
    ok &= cmp("dembcat", dembcat, embcat)
    dW1 = embcat.T @ dhprebn
    ok &= cmp("dW1", dW1, W1)
    db1 = dhprebn.sum(dim=0)
    ok &= cmp("db1", db1, b1)

    demb = dembcat.view(-1, block_size, embd_dim)
    ok &= cmp("demb", demb, emb)

    # the embedding scatter: the same character index appears many times per
    # batch, so gradients accumulate.  Written as an einsum against the
    # one-hot encoding -- contracting the (batch, position) indices IS the
    # scatter-add, made explicit:  dC_{ve} = sum_{b,c} onehot_{bcv} demb_{bce}
    # (equivalent to dC.index_add_(0, Xb.view(-1), demb.view(-1, embd_dim)),
    # which allocates no one-hot but hides the sum)
    xbonehot = F.one_hot(Xb, num_classes=volc_dim).float()
    dC = torch.einsum(
        "bce,bcv->ve",
        demb,       # [n, block_size, embd_dim]
        xbonehot,   # [n, block_size, volc_dim]
    )
    ok &= cmp("dC", dC, C)

    close = all(app for _, _, app in results)
    if ok:
        print("ALL EXACT")
    elif close:
        print("all correct, not all bit-exact (torch build) -- maxdiff ~1e-9")
    else:
        print("SOME GRADIENTS WRONG -- check the approximate column above")

    # ------------------------------------------------------------------
    # Exercise 2: the whole softmax + cross-entropy block in one line.
    #
    #   dL/dZ_{beta j} = (P_{beta j} - Y_{beta j}) / n
    #
    # Nine intermediate tensors (logit_maxes ... logprobs) collapse because
    # the max-shift contributes nothing and the log and exp cancel.  This is
    # what F.cross_entropy does internally, and why it is both faster and
    # numerically better behaved than the explicit chain above.
    # ------------------------------------------------------------------

    print("\n--- exercise 2: fused cross-entropy backward ---")

    loss_fast = F.cross_entropy(logits, Yb)
    print(f"loss fast {loss_fast.item():.6f}   vs decomposed {loss.item():.6f}")

    dlogits_fast = (F.softmax(logits, 1) - Yb_onehot) / n
    ok_fused = cmp("dlogits", dlogits_fast, logits, exact=False)

    # ------------------------------------------------------------------
    # Exercise 3: the whole BatchNorm block in one expression.
    #
    #   dL/dhprebn = bnvar_inv * [ dbnraw - mean(dbnraw)
    #                              - bnraw * sum(bnraw * dbnraw)/(n-1) ]
    #
    # The second term is the mean coupling, the third the variance coupling:
    # both are batch sums, which is exactly why BatchNorm's backward cannot be
    # written example by example.  The 1/(n-1) rather than 1/n is Bessel's
    # correction surviving into the gradient.
    # ------------------------------------------------------------------

    print("\n--- exercise 3: fused BatchNorm backward ---")

    dhprebn_fast = bnvar_inv * (
        dbnraw
        - 1/n * dbnraw.sum(dim=0, keepdim=True)
        - 1/(n-1) * bnraw * (bnraw * dbnraw).sum(dim=0, keepdim=True)
    )
    ok_fused &= cmp("dhprebn", dhprebn_fast, hprebn, exact=False)

    print("FUSED FORMS AGREE" if ok_fused else "FUSED FORMS DISAGREE -- check above")

    # ------------------------------------------------------------------
    # Exercise 4: train with the manual backward pass only.
    #
    # Everything runs inside torch.no_grad(): no graph is built, .backward()
    # is never called, and the gradients below are the only ones there are.
    # ------------------------------------------------------------------

    if max_steps > 0:

        print(f"\n--- exercise 4: training {max_steps} steps, no autograd ---")

        # fresh parameters, same init as above
        g = torch.Generator().manual_seed(seed)
        C  = torch.randn((volc_dim, embd_dim), generator=g)
        W1 = torch.randn((embd_dim * block_size, h_dim), generator=g) * (5/3) / ((embd_dim * block_size) ** 0.5)
        b1 = torch.randn(h_dim, generator=g) * 0.1
        W2 = torch.randn((h_dim, volc_dim), generator=g) * 0.1
        b2 = torch.randn(volc_dim, generator=g) * 0.1
        bngain = torch.randn((1, h_dim)) * 0.1 + 1.0
        bnbias = torch.randn((1, h_dim)) * 0.1
        parameters = [C, W1, b1, W2, b2, bngain, bnbias]

        # a separate stream for minibatch sampling, so changing max_steps
        # cannot change the initial weights
        gs = torch.Generator().manual_seed(split_seed)

        lossi = []
        print_every = max(1, max_steps // 10)
        with torch.no_grad():
            for i in range(max_steps):
                ix = torch.randint(0, len(Ydata), (minb_size,), generator=gs)
                Xb, Yb = Xdata[ix], Ydata[ix]

                # ---- forward ----
                emb = C[Xb]
                embcat = emb.view(n, -1)
                hprebn = embcat @ W1 + b1
                bnmean = hprebn.mean(dim=0, keepdim=True)
                bnvar = hprebn.var(dim=0, keepdim=True)        # unbiased, matches 1/(n-1) below
                bnvar_inv = (bnvar + 1e-5) ** -0.5
                bnraw = (hprebn - bnmean) * bnvar_inv
                hpreact = bngain * bnraw + bnbias
                h = torch.tanh(hpreact)
                logits = h @ W2 + b2
                loss = F.cross_entropy(logits, Yb)

                # ---- backward, by hand (exercises 2 and 3 inlined) ----
                Yb_onehot = F.one_hot(Yb, num_classes=volc_dim).float()
                dlogits = (F.softmax(logits, 1) - Yb_onehot) / n

                dW2 = h.T @ dlogits
                db2 = dlogits.sum(dim=0)
                dh = dlogits @ W2.T

                dhpreact = (1 - h**2) * dh
                dbngain = (bnraw * dhpreact).sum(dim=0, keepdim=True)
                dbnbias = dhpreact.sum(dim=0, keepdim=True)
                dbnraw = bngain * dhpreact

                dhprebn = bnvar_inv * (
                    dbnraw
                    - 1/n * dbnraw.sum(dim=0, keepdim=True)
                    - 1/(n-1) * bnraw * (bnraw * dbnraw).sum(dim=0, keepdim=True)
                )

                dW1 = embcat.T @ dhprebn
                db1 = dhprebn.sum(dim=0)
                dembcat = dhprebn @ W1.T
                demb = dembcat.view(-1, block_size, embd_dim)

                xbonehot = F.one_hot(Xb, num_classes=volc_dim).float()
                dC = torch.einsum("bce,bcv->ve", demb, xbonehot)

                # ---- update ----
                lr0 = lr if i < lr_decay_step else lr2
                dparams = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]
                for p, dp in zip(parameters, dparams):
                    p += -lr0 * dp

                lossi.append(loss.item())
                if i % print_every == 0:
                    print(f"{i:6d}  |  loss = {loss.item():.4f}")

        print(f"final minibatch loss = {loss.item():.4f}")

        # ------------------------------------------------------------------
        # Calibrate BatchNorm at the end of training.
        #
        # Training used per-minibatch statistics; inference has no batch to
        # average over, so the mean and variance are measured once over the
        # whole training set.  (batchnorm.py keeps running averages instead --
        # same idea, paid for during training rather than after.)
        # ------------------------------------------------------------------

        with torch.no_grad():
            embcat_all = C[Xdata].view(Xdata.shape[0], -1)
            hprebn_all = embcat_all @ W1 + b1
            bnmean = hprebn_all.mean(0, keepdim=True)
            bnvar = hprebn_all.var(0, keepdim=True, unbiased=True)

        @torch.no_grad()
        def split_loss(split):
            x, y = model.Xdic[split], model.Ydic[split]
            embcat_s = C[x].view(x.shape[0], -1)
            hpreact_s = bngain * (embcat_s @ W1 + b1 - bnmean) * (bnvar + 1e-5) ** -0.5 + bnbias
            logits_s = (torch.tanh(hpreact_s)) @ W2 + b2
            print(f"{split:5s} loss {F.cross_entropy(logits_s, y).item():.4f}")

        split_loss("train")
        split_loss("dev")
