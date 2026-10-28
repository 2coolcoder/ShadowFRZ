"""Models expressed as a sequence of units (the granularity for stages, freezing, shadows).

Interface used by the trainers, profiler and emulator:
    net.units                       nn.ModuleList
    net.prepare(batch) -> (x, ctx)  first unit input and side info (labels, masks)
    net.unit_forward(i, x, ctx)     output of unit i; the LAST unit returns the scalar loss
    net.loss(batch, on_unit=None)   full forward; on_unit(i) is called before unit i runs
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitNet(nn.Module):
    def loss(self, batch, on_unit=None):
        x, ctx = self.prepare(batch)
        for i in range(len(self.units)):
            if on_unit is not None:
                on_unit(i)
            x = self.unit_forward(i, x, ctx)
        return x

    def unit_params(self):
        return [list(u.parameters()) for u in self.units]


# ============================================================================ ResNets (CIFAR)
class BasicBlock(nn.Module):
    """He et al. CIFAR basic block; projection (1x1 conv + BN) shortcut when shape changes."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, planes, stride):
        super().__init__()
        cout = planes * 4
        self.c1 = nn.Conv2d(cin, planes, 1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.b2 = nn.BatchNorm2d(planes)
        self.c3 = nn.Conv2d(planes, cout, 1, bias=False)
        self.b3 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = F.relu(self.b2(self.c2(y)))
        y = self.b3(self.c3(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class Head(nn.Module):
    def __init__(self, cin, classes):
        super().__init__()
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


class ResNetUnits(UnitNet):
    def __init__(self, kind, classes, zero_init_residual=False):
        super().__init__()
        units = []
        if kind in ("resnet56", "resnet110"):
            n = {"resnet56": 9, "resnet110": 18}[kind]
            units.append(nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()))
            cin = 16
            for cout, stride in ((16, 1), (32, 2), (64, 2)):
                for b in range(n):
                    units.append(BasicBlock(cin, cout, stride if b == 0 else 1))
                    cin = cout
        elif kind == "resnet50":
            # CIFAR variant: 3x3 stem, no max-pool (as in common CIFAR ResNet-50 code)
            units.append(nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()))
            cin = 64
            for planes, nblk, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
                for b in range(nblk):
                    units.append(Bottleneck(cin, planes, stride if b == 0 else 1))
                    cin = planes * 4
        else:
            raise ValueError(kind)
        units.append(Head(cin, classes))
        self.units = nn.ModuleList(units)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        if zero_init_residual:      # each residual block starts as the identity (Goyal et al. 2017)
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.zeros_(m.b3.weight)
                elif isinstance(m, BasicBlock):
                    nn.init.zeros_(m.b2.weight)

    def prepare(self, batch):
        return batch[0], batch[1]

    def unit_forward(self, i, x, ctx):
        x = self.units[i](x)
        if i == len(self.units) - 1:
            return F.cross_entropy(x, ctx)
        return x

    def logits(self, x):
        for u in self.units:
            x = u(x)
        return x


# ============================================================================ BERT
BERT_SIZES = {          # Turc et al. 2019 compact BERTs + the two public sizes
    "mini":   dict(layers=4, hidden=256, heads=4),
    "small":  dict(layers=4, hidden=512, heads=8),
    "medium": dict(layers=8, hidden=512, heads=8),
    "base":   dict(layers=12, hidden=768, heads=12),
    "large":  dict(layers=24, hidden=1024, heads=16),
}


def bert_config(size, vocab=30522, max_pos=512):
    from transformers import BertConfig
    s = BERT_SIZES[size]
    return BertConfig(vocab_size=vocab, hidden_size=s["hidden"], num_hidden_layers=s["layers"],
                      num_attention_heads=s["heads"], intermediate_size=4 * s["hidden"],
                      max_position_embeddings=max_pos, tie_word_embeddings=False,
                      attn_implementation="sdpa")


def _layer_out(o):
    return o[0] if isinstance(o, (tuple, list)) else o


class BertMLMUnits(UnitNet):
    """Units: [embeddings, layer_1 .. layer_L, MLM head]. Word embeddings and the MLM
    decoder are UNTIED (separate parameters): tying would put one parameter on the first
    and last pipeline stage. Applied to every variant, so comparisons stay like-for-like."""

    def __init__(self, size):
        super().__init__()
        from transformers import BertForMaskedLM
        cfg = bert_config(size)
        m = BertForMaskedLM(cfg)
        self.cfg = cfg
        self.units = nn.ModuleList([m.bert.embeddings, *m.bert.encoder.layer, m.cls])

    def prepare(self, batch):
        ids, labels = batch
        return ids, labels

    def unit_forward(self, i, x, ctx):
        last = len(self.units) - 1
        if i == 0:
            return self.units[0](input_ids=x)
        if i < last:
            return _layer_out(self.units[i](x))
        # MLM head only at masked positions: exact for the loss, ~7x less head compute/memory
        sel = ctx != -100
        logits = self.units[last](x[sel])
        return F.cross_entropy(logits, ctx[sel])


class BertClsUnits(UnitNet):
    """Sequence classification from a public checkpoint. Units: [embeddings, layers, head]
    where head = pooler + dropout + classifier. Padding mask passed through ctx."""

    def __init__(self, name, num_labels):
        super().__init__()
        from transformers import BertForSequenceClassification
        m = BertForSequenceClassification.from_pretrained(name, num_labels=num_labels,
                                                          attn_implementation="sdpa")
        self.cfg = m.config
        head = nn.ModuleDict({"pooler": m.bert.pooler, "dropout": m.dropout, "classifier": m.classifier})
        self.units = nn.ModuleList([m.bert.embeddings, *m.bert.encoder.layer, head])

    def prepare(self, batch):
        ids, tt, mask, labels = batch
        ext = (1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        return (ids, tt), (ext, labels)

    def _head_logits(self, x):
        h = self.units[-1]
        return h["classifier"](h["dropout"](h["pooler"](x)))

    def unit_forward(self, i, x, ctx):
        ext, labels = ctx
        last = len(self.units) - 1
        if i == 0:
            return self.units[0](input_ids=x[0], token_type_ids=x[1])
        if i < last:
            return _layer_out(self.units[i](x, attention_mask=ext))
        return F.cross_entropy(self._head_logits(x), labels)

    def logits(self, batch):
        x, (ext, _) = self.prepare(batch)
        for i in range(len(self.units) - 1):
            x = self.unit_forward(i, x, (ext, None))
        return self._head_logits(x)


def count_params(net):
    return sum(p.numel() for p in net.parameters())


# ============================================================================ Pre-LN BERT (Megatron-style)
class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class PreLNEmbeddings(nn.Module):
    def __init__(self, vocab, hidden, max_pos=512, types=2, p=0.1):
        super().__init__()
        self.word = nn.Embedding(vocab, hidden)
        self.pos = nn.Embedding(max_pos, hidden)
        self.tok_type = nn.Embedding(types, hidden)
        self.drop = nn.Dropout(p)

    def forward(self, ids, token_type_ids=None):
        pos = torch.arange(ids.shape[1], device=ids.device)
        h = self.word(ids) + self.pos(pos)[None]
        h = h + self.tok_type(token_type_ids if token_type_ids is not None else torch.zeros_like(ids))
        return self.drop(h)


class PreLNLayer(nn.Module):
    """h = h + Drop(Attn(LN(h)));  h = h + Drop(MLP(LN(h)))   (Megatron-LM BERT layer)"""

    def __init__(self, hidden, heads, p=0.1):
        super().__init__()
        self.heads, self.p = heads, p
        self.ln1 = nn.LayerNorm(hidden, eps=1e-12)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden, eps=1e-12)
        self.fc1 = nn.Linear(hidden, 4 * hidden)
        self.fc2 = nn.Linear(4 * hidden, hidden)
        self.drop = nn.Dropout(p)

    def forward(self, h, attn_mask=None):
        B, T, H = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, T, 3, self.heads, H // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        h = h + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, H)))
        return h + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(h)))))


class PreLNMLMHead(nn.Module):
    def __init__(self, hidden, vocab):
        super().__init__()
        self.final_ln = nn.LayerNorm(hidden, eps=1e-12)       # Megatron's final layernorm
        self.dense = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden, eps=1e-12)
        self.decoder = nn.Linear(hidden, vocab)

    def forward(self, h):
        return self.decoder(self.ln(F.gelu(self.dense(self.final_ln(h)))))


def _bert_init(module, layers):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    for m in module.modules():                                 # Megatron scaled init of output projections
        if isinstance(m, PreLNLayer):
            for lin in (m.proj, m.fc2):
                nn.init.normal_(lin.weight, 0.0, 0.02 / math.sqrt(2 * layers))


class BertPreLNMLMUnits(UnitNet):
    """Units: [embeddings, layer_1 .. layer_L, head(final LN + MLM transform + untied decoder)]."""

    def __init__(self, size, vocab=30522):
        super().__init__()
        s = BERT_SIZES[size]
        self.cfg = _Cfg(hidden_size=s["hidden"], num_hidden_layers=s["layers"], num_attention_heads=s["heads"], vocab_size=vocab)
        self.units = nn.ModuleList([PreLNEmbeddings(vocab, s["hidden"]),
                                    *[PreLNLayer(s["hidden"], s["heads"]) for _ in range(s["layers"])],
                                    PreLNMLMHead(s["hidden"], vocab)])
        _bert_init(self, s["layers"])

    def prepare(self, batch):
        return batch[0], batch[1]

    def unit_forward(self, i, x, ctx):
        last = len(self.units) - 1
        if i == 0:
            return self.units[0](x)
        if i < last:
            return self.units[i](x)
        sel = ctx != -100
        return F.cross_entropy(self.units[last](x[sel]), ctx[sel])


class PreLNClsHead(nn.Module):
    def __init__(self, hidden, num_labels, p=0.1):
        super().__init__()
        self.final_ln = nn.LayerNorm(hidden, eps=1e-12)
        self.pooler = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(p)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, h):
        return self.classifier(self.drop(torch.tanh(self.pooler(self.final_ln(h)[:, 0]))))


class BertPreLNClsUnits(UnitNet):
    """Sequence classifier built from a Pre-LN MLM checkpoint (encoder + final LN reused)."""

    def __init__(self, size, num_labels, mlm_state=None):
        super().__init__()
        mlm = BertPreLNMLMUnits(size)
        if mlm_state is not None:
            mlm.load_state_dict(mlm_state)
        s = BERT_SIZES[size]
        head = PreLNClsHead(s["hidden"], num_labels)
        nn.init.normal_(head.pooler.weight, 0, 0.02); nn.init.zeros_(head.pooler.bias)
        nn.init.normal_(head.classifier.weight, 0, 0.02); nn.init.zeros_(head.classifier.bias)
        head.final_ln.load_state_dict(mlm.units[-1].final_ln.state_dict())
        self.cfg = mlm.cfg
        self.units = nn.ModuleList([*mlm.units[:-1], head])

    def prepare(self, batch):
        ids, tt, mask, labels = batch
        ext = (1.0 - mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        return (ids, tt), (ext, labels)

    def unit_forward(self, i, x, ctx):
        ext, labels = ctx
        last = len(self.units) - 1
        if i == 0:
            return self.units[0](x[0], x[1])
        if i < last:
            return self.units[i](x, ext)
        return F.cross_entropy(self.units[last](x), labels)

    def logits(self, batch):
        x, (ext, _) = self.prepare(batch)
        for i in range(len(self.units) - 1):
            x = self.unit_forward(i, x, (ext, None))
        return self.units[-1](x)
