# makemore — character-level language models from scratch

Character-level models of names, each one earning its keep against the last.
Following Andrej Karpathy's makemore lectures, written from memory rather than
copied.

| model | context | NLL | perplexity |
| --- | --- | --- | --- |
| uniform over 27 characters | none | 3.2958 | 27.0 |
| bigram, counted | 1 char | 2.4546 | 11.6 |
| bigram, gradient descent | 1 char | ≈ 2.45 | ≈ 11.6 |
| MLP with embeddings | `block_size` (3 by default) | see below | |
| MLP + BatchNorm, √fan_in init | `block_size` (3 by default) | fill in after running | |

Perplexity reads as an effective number of choices: one character of context
narrows the next character from 27 possibilities to about 12. The loss cannot
reach zero — its floor is the entropy of the names themselves.

---

# 1. Bigram

Predict each character from the one before it. With `.` as the start/end token
the vocabulary is $V = 27$, the corpus gives $N = 228{,}146$ character pairs, and
the model is a single $27 \times 27$ table of conditional probabilities.

Run with `python bigram.py`.

## Two routes to the same table

**Counting.** Tally every observed pair into a $27 \times 27$ matrix, add one to
every entry, normalise each row. Closed form, no optimisation.

**Gradient descent.** A single weight matrix $W$ of the same shape, trained on
the negative log-likelihood

$$\mathcal{L} = -\frac{1}{N}\sum_{\alpha=1}^{N} \log P_{\alpha, y_\alpha},
\qquad P_{\alpha j} = \frac{\exp Z_{\alpha j}}{\sum_k \exp Z_{\alpha k}},
\qquad Z_{\alpha j} = \sum_i X_{\alpha i} W_{ij}$$

with $\alpha$ indexing training examples and $i,j,k$ indexing characters.

The second route converges to the first. That is the point: the neural network
does not learn something cleverer than counting, it *rediscovers* counting — and
unlike counting it keeps working when the context grows beyond one character.

![bigram counts and training curve](bigram.png)

## Smoothing is regularisation

Add-one smoothing in the count model and the L2 penalty `0.01 * (W**2).mean()` in
the trained model do the same job. Both pull the distribution toward uniform and
prevent any pair from being assigned zero probability. Remove the L2 term and the
trained model matches the *unsmoothed* counts instead. The two knobs are the same
knob.

---

# 2. MLP with character embeddings

Following Bengio et al. 2003. Run with `python example_MLP.py`.

`block_size` characters of context instead of one — three by default, but it is a
constructor argument and the whole model follows from it. A lookup table over
$b$ characters would need $27^{\,b}$ rows, which is already $\approx 19{,}700$ at
$b = 3$ and hopeless beyond: most rows would never be observed in 32k names.

Embeddings avoid that. Each character maps to a learned vector of dimension
$d$ = `embd_dim`, the $b$ context vectors are concatenated into one of length
$b\,d$, and a hidden layer of size `h_dim` reads the result:

$$X \;(N \times b) \;\xrightarrow{\;C\;}\; (N \times b \times d) \;\xrightarrow{\text{reshape}}\; (N \times bd) \;\xrightarrow{\;W_1, b_1,\ \tanh\;}\; (N \times h) \;\xrightarrow{\;W_2, b_2\;}\; (N \times 27)$$

with $C$ of shape $(27, d)$, $W_1$ of shape $(bd, h)$ and $W_2$ of shape
$(h, 27)$. Parameter count grows *linearly* in $b$, where the lookup table grew
exponentially — that is the whole trick. Characters that behave alike end up with
nearby embeddings, so the model generalises to contexts it has never seen, which
is exactly what the count table cannot do.

![MLP training curve against both baselines](mlp_loss.png)

## Structure

`MLP.py` holds the class; `example_MLP.py` is one configured experiment.

| method | |
| --- | --- |
| `makeXY()` | build $X$ of shape $(N, b)$ and $Y$ of shape $(N,)$, context reset per name |
| `build_data()` | shuffle into 80/10/10 train / dev / test |
| `init_params()` | $C$, $W_1$, $b_1$, $W_2$, $b_2$ |
| `forward()` | one pass, shared by training and evaluation so the two cannot drift |
| `evo_loss()` | minibatch SGD, returns the loss history |
| `evaluate()` | whole-split loss under `no_grad` |

## Seeding

`seed` is a constructor argument; `seed=None` disables seeding entirely and every
run differs. `split_seed` is kept separate, so the same train/dev/test partition
can be held fixed while the initialisation varies — that is how you tell whether
a result is the model or one lucky draw. Weight initialisation and minibatch
sampling draw from independent generators, so changing the number of training
steps cannot change the starting weights.

## Two things left deliberately wrong

The weights are initialised with plain `torch.randn`, unscaled. makemore 3 opens
by showing why that is wrong and what to divide by; seeing it fail first is the
exercise.

There is no learning-rate schedule. The rate is constant for the whole run.

---

# 3. BatchNorm and activation diagnostics

Following makemore 3. Run with `python batchnorm.py`.

The MLP above trains, but only because it is shallow. This step builds the
tools that show *why* it would fail deeper, then the layer that makes depth
safe.

## Module-ized layers

The monolithic forward pass becomes composable classes — `Layer`,
`BatchNorm1d`, `Tanh` — each holding its parameters and its output. This is
`torch.nn.Module` rebuilt by hand: the network is a list, the forward pass is
a loop.

## Initialization: divide by √fan_in

A pre-activation is a sum of `fan_in` independent terms. For zero-mean
$x_i$ and $W_{ij}$, variances multiply and then add:

$$\mathrm{Var}\Big(\sum_i x_i W_{ij}\Big) = n_{\text{in}}\,\sigma_w^2$$

so plain `randn` weights give the sum a width of $\sqrt{n_{\text{in}}}$ —
random-walk scaling. tanh then saturates at $\pm 1$, and saturated units pass
almost no gradient. Dividing the weights by $\sqrt{n_{\text{in}}}$ restores
unit variance at every layer. (Note that ordinary error propagation *misses*
this: it linearizes around the mean, and here the means are zero — the
variance product term it drops is the whole effect.)

## BatchNorm

`BatchNorm1d` standardizes each hidden unit over the minibatch, then lets the
network undo it where useful via learned `gamma`, `beta`. Training uses batch
statistics; inference uses running averages accumulated with momentum — the
train/inference asymmetry is the subtle part, and it is why the class carries
`running_mean` / `running_var` alongside the learned parameters.

## Diagnostics

`batchnorm.py` ends with the lecture's diagnostic panel, written before
watching his version: tanh activation histograms with the saturation
fraction (|h| > 0.97), gradient histograms per layer, and the loss curve.

![BatchNorm diagnostics](batchnorm_diag.png)

## A bug worth remembering

The first version of this network excluded the embedding matrix $C$ from the
parameter list. Nothing crashed — the loss still fell, the samples still
looked name-like — but the embeddings stayed frozen at their random initial
values for the entire run. Silent failure is the default failure mode of
neural networks; the diagnostics above exist because errors here do not
announce themselves.

---

## Files

| File | |
| --- | --- |
| `bigram.py` | both bigram models, sampling, figures |
| `MLP.py` | the MLP class |
| `example_MLP.py` | one configured MLP experiment, with baselines and diagnostics |
| `batchnorm.py` | module-ized layers, BatchNorm, activation/gradient diagnostics |
| `init_scaling.py` | deep tanh stack, no BatchNorm: break the init (10×, 0.1×), watch the histograms fail |
| `names.txt` | 32k names, one per line |
| `bigram.png`, `mlp_loss.png`, `batchnorm_diag.png` | generated figures |

## Notes

`bigram.py` computes `exp` → normalise → `log` explicitly rather than calling
`F.cross_entropy`. That is deliberate — the manual version is what the derivation
looks like — but it is not the form to use in real code: `logits.exp()` overflows
once any logit exceeds about 88 in float32, and the fused version has a cleaner
backward pass, $\partial\mathcal{L}/\partial Z = (P - Y)/N$. `MLP.py` uses the
fused version.

The one-hot encoding in `bigram.py` is likewise deliberate. `W[xs]` is identical
and allocates nothing, while the one-hot matrix costs about 25 MB; writing it out
makes visible that an embedding lookup *is* a matrix multiplication. `MLP.py` uses
the lookup.

## Next

makemore 4: the backward pass computed by hand for every parameter and checked
against autograd. Then makemore 5, WaveNet — hierarchical context instead of a
flat concatenation.
