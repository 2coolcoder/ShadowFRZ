"""Check the emulator's closed-form version rules against the simulated 1F1B schedule."""
from pipeemu.schedule import straddlers_1bw, time_per_batch, version_table


def check_1bw(d, m, nb=4):
    t = version_table(d, m, nb, "1bw")
    closed = straddlers_1bw(d, m)
    bad = 0
    for s in range(1, d + 1):
        for k in range(m + 1, m * nb + 1):           # skip batch 0 (no previous version yet)
            b, j = (k - 1) // m, (k - 1) % m + 1
            vf, vb = t[(s, "F", k)], t[(s, "B", k)]
            expect_f = b - 1 if closed[(s, j)] else b
            if vb != b or vf != expect_f:
                bad += 1
    return bad


def check_2bw(d, m, nb=4):
    t = version_table(d, m, nb, "2bw")
    return sum(t[(s, "F", k)] != t[(s, "B", k)] for s in range(1, d + 1) for k in range(1, m * nb + 1))


if __name__ == "__main__":
    for d in (2, 4, 8):
        for r in (1, 2, 4):
            m = d * r
            print(f"d={d} m={m}: 1BW rule mismatches={check_1bw(d, m)}  2BW F!=B={check_2bw(d, m)}")
    # flush vs no-flush time per batch for a balanced pipeline (F=1,B=2 per stage)
    for d, m in ((4, 4), (4, 16), (8, 8)):
        tf = time_per_batch([1] * d, [2] * d, m, flush=True)
        tn = time_per_batch([1] * d, [2] * d, m, flush=False)
        print(f"d={d} m={m}: flush {tf:.1f}  no-flush {tn:.1f}  ideal {3*m}")
