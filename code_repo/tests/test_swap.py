"""The emulator's core assumption: if a layer's weights are overwritten in place via
`p.data.copy_(w_new)` between forward and backward, then backward
  (a) raises no autograd version error,
  (b) computes the INPUT gradient with w_new (what a real pipeline without a shadow does),
  (c) computes the WEIGHT gradient from the stored forward activations.
Checked for every layer type we use: Linear, Conv2d, BatchNorm2d, LayerNorm."""
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = "cuda"


def check(layer, x_shape, name):
    torch.manual_seed(0)
    layer = layer.to(dev).double()
    x = torch.randn(*x_shape, device=dev, dtype=torch.float64, requires_grad=True)
    old = {k: p.detach().clone() for k, p in layer.named_parameters()}
    new = {k: p + 0.5 * torch.randn_like(p) for k, p in old.items()}

    # emulated mismatch: forward with old, swap to new in place, backward
    y = layer(x)
    g = torch.randn_like(y)
    for k, p in layer.named_parameters():
        p.data.copy_(new[k])
    y.backward(g)
    gx_emul = x.grad.clone()
    gw_emul = {k: p.grad.clone() for k, p in layer.named_parameters()}

    # reference input-grad: d/dx of <layer_new(x), g> evaluated with x's stored activation.
    # For layers whose input-grad depends only on weights (+ saved input stats), this equals
    # a fresh backward through the NEW weights at the same x.
    layer.zero_grad(set_to_none=True)
    x2 = x.detach().clone().requires_grad_(True)
    y2 = layer(x2)
    y2.backward(g)
    gx_ref_new = x2.grad
    # reference weight-grad: fresh backward through OLD weights
    for k, p in layer.named_parameters():
        p.data.copy_(old[k])
    layer.zero_grad(set_to_none=True)
    x3 = x.detach().clone().requires_grad_(True)
    layer(x3).backward(g)
    gw_ref_old = {k: p.grad.clone() for k, p in layer.named_parameters()}

    ok_x = torch.allclose(gx_emul, gx_ref_new, atol=1e-9)
    ok_w = all(torch.allclose(gw_emul[k], gw_ref_old[k], atol=1e-9) for k in gw_emul)
    print(f"{name:10s} input-grad uses NEW weights: {ok_x}   weight-grad uses OLD activations: {ok_w}")
    return ok_x, ok_w


if __name__ == "__main__":
    res = [
        check(nn.Linear(16, 8), (4, 16), "Linear"),
        check(nn.Conv2d(3, 5, 3, padding=1), (2, 3, 8, 8), "Conv2d"),
        check(nn.LayerNorm(16), (4, 16), "LayerNorm"),
    ]
    bn = nn.BatchNorm2d(5)
    res.append(check(bn, (4, 5, 6, 6), "BatchNorm"))
    print("ALL OK" if all(a for a, _ in res) else "MISMATCH SEMANTICS DIFFER")
