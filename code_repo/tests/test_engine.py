"""V1: 'exact' mode == plain gradient accumulation.
V2: 'shadow' mode == closed-form 2BW update  W(t+1) = W(t) - lr * grad f(W(t-1)).
V2b: 'frozen' units never change and receive no gradient.
Float64 so equality is tight."""
import copy

import torch
import torch.nn as nn

from pipeemu.engine import PipelineEmulator

torch.manual_seed(0)
dev = "cuda"
D, M, STEPS, LR = 4, 4, 6, 0.1


def make():
    torch.manual_seed(1)
    units = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(4)]).to(dev).double()
    return units


def fwd(units, x):
    for u in units:
        x = u(x)
    return x


data = [[(torch.randn(5, 8, device=dev, dtype=torch.float64), torch.randn(5, 8, device=dev, dtype=torch.float64))
         for _ in range(M)] for _ in range(STEPS)]
loss_of = lambda units: (lambda mb: ((fwd(units, mb[0]) - mb[1]) ** 2).mean())


def emulate(mode_list):
    units = make()
    opt = torch.optim.SGD(units.parameters(), lr=LR)
    emu = PipelineEmulator([list(u.parameters()) for u in units], [1, 2, 3, 4], D, M, opt)
    emu.set_modes(mode_list)
    for batch in data:
        emu.step(batch, loss_of(units))
    return units


def flat(units):
    return torch.cat([p.detach().flatten() for p in units.parameters()])


# ---- V1: plain gradient accumulation
ref = make()
opt = torch.optim.SGD(ref.parameters(), lr=LR)
for batch in data:
    for mb in batch:
        (loss_of(ref)(mb) / M).backward()
    opt.step(); opt.zero_grad()
v1 = (flat(emulate(["exact"] * 4)) - flat(ref)).abs().max().item()
print(f"V1 exact vs grad-accumulation   max|diff| = {v1:.2e}")

# ---- V2: closed-form 2BW
ref = make()
w_prev = copy.deepcopy(ref)
for batch in data:
    grads = [torch.zeros_like(p) for p in ref.parameters()]
    for mb in batch:
        w_prev.zero_grad()
        (loss_of(w_prev)(mb) / M).backward()           # gradient at W(t-1)
        for g, p in zip(grads, w_prev.parameters()):
            g += p.grad
    snapshot = copy.deepcopy(ref)                       # W(t) becomes next W(t-1)
    with torch.no_grad():
        for p, g in zip(ref.parameters(), grads):
            p -= LR * g
    w_prev = snapshot
v2 = (flat(emulate(["shadow"] * 4)) - flat(ref)).abs().max().item()
print(f"V2 shadow vs closed-form 2BW     max|diff| = {v2:.2e}")

# ---- V2b: frozen unit untouched
units = make()
before = [p.detach().clone() for p in units[1].parameters()]
opt = torch.optim.SGD(units.parameters(), lr=LR)
emu = PipelineEmulator([list(u.parameters()) for u in units], [1, 2, 3, 4], D, M, opt)
emu.set_modes(["noshadow", "frozen", "shadow", "noshadow"])
for batch in data:
    emu.step(batch, loss_of(units))
v2b = max((a - b).abs().max().item() for a, b in zip(before, units[1].parameters()))
print(f"V2b frozen unit drift            max|diff| = {v2b:.2e}")
print(f"1BW mismatch bookkeeping: {emu.stats}  (expect units on stages<d to mismatch for j<=d-s)")
ok = v1 < 1e-12 and v2 < 1e-12 and v2b == 0
print("PASS" if ok else "FAIL")
