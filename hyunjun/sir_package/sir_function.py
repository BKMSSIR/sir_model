def _func_sus(N, sus, inf, beta):
    return -beta * sus * inf


def _func_inf(N, sus, inf, beta, gamma):
    return beta * sus * inf - gamma * inf


def _func_rec(N, inf, gamma):
    return gamma * inf


def _rk4(N, sus, inf, rec, fs, fi, fr, beta, gamma, hs):

    s1 = fs(N, sus, inf, beta) * hs
    i1 = fi(N, sus, inf, beta, gamma) * hs
    r1 = fr(N, inf, gamma) * hs

    s_k = sus + s1 * 0.5
    i_k = inf + i1 * 0.5
    r_k = rec + r1 * 0.5

    s2 = fs(N, s_k, i_k, beta) * hs
    i2 = fi(N, s_k, i_k, beta, gamma) * hs
    r2 = fr(N, i_k, gamma) * hs

    s_k = sus + s2 * 0.5
    i_k = inf + i2 * 0.5
    r_k = rec + r2 * 0.5

    s3 = fs(N, s_k, i_k, beta) * hs
    i3 = fi(N, s_k, i_k, beta, gamma) * hs
    r3 = fr(N, i_k, gamma) * hs

    s_k = sus + s3
    i_k = inf + i3
    r_k = rec + r3

    s4 = fs(N, s_k, i_k, beta) * hs
    i4 = fi(N, s_k, i_k, beta, gamma) * hs
    r4 = fr(N, i_k, gamma) * hs

    sus = sus + (s1 + 2 * (s2 + s3) + s4) / 6
    inf = inf + (i1 + 2 * (i2 + i3) + i4) / 6
    rec = rec + (r1 + 2 * (r2 + r3) + r4) / 6

    return sus, inf, rec


def SIR(N, b0, beta, gamma, hs):
    sus = float(N-1)/N - b0
    inf = float(1)/N + b0
    rec = 0.

    sus_plots, inf_plots, rec_plots = [], [], []

    for _ in range(10000):
        sus_plots.append(sus)
        inf_plots.append(inf)
        rec_plots.append(rec)

        sus, inf, rec = _rk4(N, sus, inf, rec, _func_sus, _func_inf, _func_rec, beta, gamma, hs)

    return sus_plots, inf_plots, rec_plots

