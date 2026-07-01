# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pulp",
#     "networkx",
#     "scipy",
#     "numpy",
#     "matplotlib",
#     "wigglystuff",
#     "anywidget",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(
    width="full",
    app_title="PuE Übung 9: Integer Programming",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pulp as pl
    return mo, np, pl, plt


@app.cell(hide_code=True)
def solver(pl):
    # ─────────────────────────────────────────────────────────────────────
    # Hintergrund-Solver — löst PuLP-Modelle via scipy/HiGHS (auch im Browser).
    # CBC läuft in Pyodide (Website-WASM) NICHT, deshalb dieser Shim.
    # NEU ggü. UE 08: liest die Variablen-Kategorie (Integer/Binary) aus und
    # übergibt einen `integrality`-Vektor an HiGHS → echte (M)IP-Lösungen.
    # Für reine LP-Relaxationen (B&B-Walkthrough) `relax=True` setzen.
    # ─────────────────────────────────────────────────────────────────────
    import numpy as _np
    from scipy.optimize import linprog

    def _solve_prob(prob, relax=False):
        variables = prob.variables()
        n = len(variables)
        idx = {v.name: i for i, v in enumerate(variables)}
        c = _np.zeros(n)
        if prob.objective is not None:
            for v, coef in prob.objective.items():
                c[idx[v.name]] = coef
        if prob.sense == pl.LpMaximize:
            c = -c
        bounds = [(v.lowBound, v.upBound) for v in variables]
        # Ganzzahligkeit: 0 = kontinuierlich, 1 = ganzzahlig (HiGHS).
        integrality = _np.zeros(n)
        if not relax:
            for v in variables:
                if v.cat in (pl.LpInteger,) or getattr(v, "cat", "") in (
                    "Integer", "Binary"):
                    integrality[idx[v.name]] = 1
        A_ub_rows, b_ub_vals, A_eq_rows, b_eq_vals = [], [], [], []
        for _name, con in prob.constraints.items():
            row = _np.zeros(n)
            for v, coef in con.items():
                row[idx[v.name]] = coef
            rhs = -con.constant
            if con.sense == pl.LpConstraintLE:
                A_ub_rows.append(row); b_ub_vals.append(rhs)
            elif con.sense == pl.LpConstraintGE:
                A_ub_rows.append(-row); b_ub_vals.append(-rhs)
            else:
                A_eq_rows.append(row); b_eq_vals.append(rhs)
        result = linprog(
            c, A_ub=_np.array(A_ub_rows) if A_ub_rows else None,
            b_ub=_np.array(b_ub_vals) if b_ub_vals else None,
            A_eq=_np.array(A_eq_rows) if A_eq_rows else None,
            b_eq=_np.array(b_eq_vals) if b_eq_vals else None,
            bounds=bounds,
            integrality=(None if relax else integrality),
            method="highs")
        prob.status = 1 if result.success else -1
        if result.success:
            for i, v in enumerate(variables):
                v.varValue = float(result.x[i])
        return prob.status

    class _ScipyHiGHSSolver(pl.LpSolver):
        name = "SCIPY_HIGHS"

        def available(self):
            return True

        def actualSolve(self, lp, **kwargs):
            return _solve_prob(lp)

    _scipy_default = _ScipyHiGHSSolver()
    pl.LpSolverDefault = _scipy_default
    pl.pulp.LpSolverDefault = _scipy_default
    pl.apis.LpSolverDefault = _scipy_default
    return


@app.cell(hide_code=True)
def colors():
    IMSBlue = "#023B88"
    IMSOrange = "#D87237"
    farbe_lp = "#9FBEE6"        # LP-Relaxation / Zulässigkeitsbereich
    farbe_incumbent = "#A7D3A6"  # ganzzahlig / Incumbent
    farbe_prune = "#E6A6A6"      # geprunt
    farbe_aktiv = "#F3C9A8"      # aktiver Knoten / Bereich
    farbe_infeasible = "#C9CED6"  # unzulässig (grau)
    farbe_superseded = "#D7DCE2"  # überholter Incumbent (blasses grau)
    return (
        IMSBlue,
        IMSOrange,
        farbe_aktiv,
        farbe_incumbent,
        farbe_infeasible,
        farbe_lp,
        farbe_prune,
        farbe_superseded,
    )


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # PuE — Übung 9: Diskrete Optimierung / Integer Programming

    **Planen und Entscheiden · SS 2026**

    Heute geht es um **ganzzahlige Optimierung**: wie man Entscheidungen mit
    Binär- und Ganzzahlvariablen als (M)IP **formuliert** (Text → Modell) und wie
    der **Branch-and-Bound**-Algorithmus solche Probleme löst.

    ---

    ## Fahrplan

    | Phase | Inhalt |
    |------|--------|
    | **Wiederholung** | IP/MIP-Standardform · B&B in 3 Schritten · Constraint-Muster · $\sum$/$\forall$ |
    | **Geführt 1** | Branch-and-Bound **grafisch** an einem 2-Variablen-IP |
    | **Geführt 2** | Kraftwerkseinsatz als MIP modellieren (Fixkosten · Mindestlast · Entweder-oder) |
    | **Spielwiese** | Eigenes Mini-IP bauen und durch B&B geführt werden |
    | **Selbständig** | 4 Textaufgaben: Projektauswahl · Druckerei · Filialnetz · Stundenplan |
    | **Besprechung** | Trigger-Wörter · Mustererkennung |
    """)
    return


@app.cell(hide_code=True)
def wdh_header(mo):
    mo.md(r"""
    # 1 · Wiederholung

    ## Integer Programming (IP / MIP) — Standardform

    $$
    \begin{aligned}
    \max \quad & c^\top x \\
    \text{s.t.} \quad & Ax \le b \\
    & x \ge 0 \\
    & x_i \in \mathbb{Z} && \forall i \in \mathcal{I}
    \end{aligned}
    $$

    | Variablentyp | Schreibweise | Wofür? |
    |---|---|---|
    | **Binär** | $x \in \{0,1\}$ | Ja/Nein-Entscheidung (auswählen, öffnen) |
    | **Ganzzahlig** | $x \in \mathbb{Z}_{\ge 0}$ | Stückzahlen (unteilbare Einheiten) |
    | **Kontinuierlich** | $x \in \mathbb{R}_{\ge 0}$ | Mengen (Liter, kg, Stunden) |

    Sind **alle** Variablen ganzzahlig → **IP**; nur **einige** → **MIP**
    (gemischt-ganzzahlig).
    """)
    return


@app.cell(hide_code=True)
def wdh_bnb(mo):
    mo.md(r"""
    ## Branch-and-Bound in 3 Schritten

    1. **Relaxieren** — Ganzzahligkeit weglassen, LP lösen. Das LP-Optimum ist
       eine **Schranke**: bei Maximierung eine **obere Schranke** ($UB$).
    2. **Verzweigen (Branch)** — ist ein $x_i^\star$ fraktional, teile in zwei
       Teilprobleme: $x_i \le \lfloor x_i^\star \rfloor$ **oder**
       $x_i \ge \lceil x_i^\star \rceil$.
    3. **Abschneiden (Bound/Prune)** — einen Ast verwerfen, wenn
       - das LP **unzulässig** ist, **oder**
       - die LP-Lösung **ganzzahlig** ist (Kandidat für die beste Lösung =
         **Incumbent**), **oder**
       - die Schranke **nicht besser** als der Incumbent ist
         ($UB \le$ Incumbent).

    > **Merke:** B&B ist *systematisches Durchsuchen mit mathematischen
    > Schranken* — es probiert **nicht** alle $2^n$ Kombinationen durch.
    """)
    return


@app.cell(hide_code=True)
def wdh_patterns(mo):
    mo.md(r"""
    ## Constraint-Muster-Bibliothek (Deutsch → Mathematik)

    Die halbe Miete in der Klausur: **Trigger-Wörter** erkennen und in die
    passende Nebenbedingung übersetzen. $x_i \in \{0,1\}$.

    | Formulierung im Text | Nebenbedingung |
    |---|---|
    | „**mindestens** $k$ …" | $\sum_i x_i \ge k$ |
    | „**höchstens** $k$ …" | $\sum_i x_i \le k$ |
    | „**genau** $k$ …" | $\sum_i x_i = k$ |
    | „**falls** $A$, **dann** $B$" | $x_A \le x_B$ |
    | „**falls** $A$, **dann** $B$ **und** $C$" | $x_A \le x_B,\;\; x_A \le x_C$ |
    | „$A$ **oder** $B$ (nicht beide)" | $x_A + x_B \le 1$ |
    | „$A$ **oder** $B$ (mind. eins)" | $x_A + x_B \ge 1$ |
    | „**entweder** $A$ **oder** $B$" | $x_A + x_B = 1$ |
    | „$A$ und $B$ schließen sich aus" | $x_A + x_B \le 1$ |
    | „**wenn** $y=1$, **dann** $q \le U$" (Aktivierung) | $0 \le q \le U\,y$ |
    | „**wenn** $y=0$, **dann** $x \le s$" (Big-$M$) | $x \le s + M\,y$ |

    > **Klausur-Tipp:** „falls A dann nicht B" ist **dasselbe** wie
    > „A und B schließen sich aus" → $x_A + x_B \le 1$.
    """)
    return


@app.cell(hide_code=True)
def bnb_engine(
    IMSBlue,
    IMSOrange,
    farbe_aktiv,
    farbe_incumbent,
    farbe_infeasible,
    farbe_lp,
    farbe_prune,
    farbe_superseded,
    np,
    plt,
):
    from matplotlib.patches import FancyBboxPatch
    from scipy.optimize import linprog as _linprog

    def bnb_lp_relax(c, A, b, lo, hi):
        res = _linprog(-np.array(c, float), A_ub=np.array(A, float),
                       b_ub=np.array(b, float), bounds=list(zip(lo, hi)),
                       method="highs")
        if not res.success:
            return None, None
        return res.x, float(np.array(c) @ res.x)

    def _is_int(v, tol=1e-6):
        return abs(v - round(v)) < tol

    def bnb_run(c, A, b, lo0=(0, 0), hi0=(None, None), tol=1e-6):
        """Vollständiges Branch-and-Bound für ein 2-Variablen-Max-IP.
        Liefert (Knotenliste in Erzeugungsreihenfolge, Incumbent)."""
        nodes = []
        inc = {"z": -np.inf, "x": None}
        cnt = [0]

        def rec(lo, hi, parent, label):
            nid = cnt[0]; cnt[0] += 1
            x, z = bnb_lp_relax(c, A, b, lo, hi)
            nd = {"id": nid, "parent": parent, "label": label, "x": x,
                  "ub": z, "lo": list(lo), "hi": list(hi)}
            if x is None:
                nd["status"] = "infeasible"; nodes.append(nd); return
            if z <= inc["z"] + tol:
                nd["status"] = "pruned"; nodes.append(nd); return
            frac = [(i, xi) for i, xi in enumerate(x) if not _is_int(xi, tol)]
            if not frac:
                nd["status"] = "integer"
                if z > inc["z"] + tol:
                    inc["z"] = z; inc["x"] = [round(v) for v in x]
                    nd["incumbent"] = True
                nodes.append(nd); return
            nd["status"] = "fractional"; nodes.append(nd)
            i, xi = max(frac, key=lambda t: abs(t[1] - round(t[1])))
            f = int(np.floor(xi)); cl = int(np.ceil(xi))
            _sub = ["x₁", "x₂"][i]
            lo_d, hi_d = list(lo), list(hi); hi_d[i] = f
            rec(lo_d, hi_d, nid, f"{_sub} ≤ {f}")
            lo_u, hi_u = list(lo), list(hi); lo_u[i] = cl
            rec(lo_u, hi_u, nid, f"{_sub} ≥ {cl}")

        rec(list(lo0), list(hi0), None, "Wurzel")
        return nodes, inc

    _COL = {"fractional": farbe_lp, "integer": farbe_incumbent,
            "pruned": farbe_prune, "infeasible": farbe_infeasible}

    def _tree_pos(nodes):
        kids = {}
        for n in nodes:
            kids.setdefault(n["parent"], []).append(n["id"])
        by_id = {n["id"]: n for n in nodes}
        depth = {}
        for n in nodes:
            d = 0; p = n["parent"]
            while p is not None:
                d += 1; p = by_id[p]["parent"]
            depth[n["id"]] = d
        xs = {}; leaf = [0.0]

        def lay(i):
            ch = kids.get(i, [])
            if not ch:
                xs[i] = leaf[0]; leaf[0] += 1.0; return xs[i]
            cx = [lay(c) for c in ch]; xs[i] = sum(cx) / len(cx); return xs[i]

        for r in [n["id"] for n in nodes if n["parent"] is None]:
            lay(r)
        return {i: (xs[i], -depth[i]) for i in xs}

    def _clip(poly, a, c):
        # Sutherland-Hodgman: schneidet konvexes Polygon mit Halbebene a·x ≤ c.
        out = []
        n = len(poly)
        for i in range(n):
            p = poly[i]; q = poly[(i + 1) % n]
            dp = a[0] * p[0] + a[1] * p[1] - c
            dq = a[0] * q[0] + a[1] * q[1] - c
            if dp <= 1e-9:
                out.append(p)
            if (dp < -1e-9 < dq) or (dq < -1e-9 < dp):
                t = dp / (dp - dq)
                out.append((p[0] + t * (q[0] - p[0]),
                            p[1] + t * (q[1] - p[1])))
        return out

    def _node_poly(nd, A, b, xmax, ymax):
        poly = [(0, 0), (xmax, 0), (xmax, ymax), (0, ymax)]
        half = list(zip([list(r) for r in A], b))
        lo, hi = nd["lo"], nd["hi"]
        if hi[0] is not None:
            half.append(([1, 0], hi[0]))
        if hi[1] is not None:
            half.append(([0, 1], hi[1]))
        if lo[0] > 0:
            half.append(([-1, 0], -lo[0]))
        if lo[1] > 0:
            half.append(([0, -1], -lo[1]))
        for a, c in half:
            poly = _clip(poly, a, c)
            if not poly:
                break
        return poly

    def _feasible(i, j, A, b):
        return all(r[0] * i + r[1] * j <= bb + 1e-9 for r, bb in zip(A, b))

    def _isoline(ax, c, xmax, k, color, lw, label=None):
        # Eine Iso-Zielfunktionslinie c·x = k durch den aktuellen Punkt.
        xs = np.linspace(0, xmax, 50)
        if abs(c[1]) > 1e-9:
            ax.plot(xs, (k - c[0] * xs) / c[1], color=color, ls="--",
                    lw=lw, zorder=4, label=label)

    def _faint_lattice(ax, xmax, ymax):
        # Ganzzahliges Gitter (noch unbewertet) — Startbild des Aufbaus.
        for i in range(int(xmax) + 1):
            for j in range(int(ymax) + 1):
                ax.plot(i, j, "o", ms=5, color="#c8cdd5", zorder=2,
                        clip_on=False)

    def _draw_lattice(ax, A, b, xmax, ymax):
        # Ganzzahlige Punkte über den GANZEN Graphen: zulässig grün, sonst offen.
        for i in range(int(xmax) + 1):
            for j in range(int(ymax) + 1):
                if _feasible(i, j, A, b):
                    ax.plot(i, j, "o", ms=7, color=farbe_incumbent,
                            mec="#2f6b2f", mew=1.0, zorder=5, clip_on=False)
                else:
                    ax.plot(i, j, "o", ms=7, mfc="white", mec="#9aa3ad",
                            mew=1.0, zorder=5, clip_on=False)

    def _draw_constraints(ax, A, b, xmax):
        xs = np.linspace(0, xmax, 200)
        for row, bb in zip(A, b):
            a1, a2 = row
            if abs(a2) < 1e-9:
                ax.axvline(bb / a1, color=IMSBlue, lw=1.1, alpha=0.6)
            else:
                ax.plot(xs, (bb - a1 * xs) / a2, color=IMSBlue, lw=1.1,
                        alpha=0.6)

    def _best_upto(nodes, upto):
        # Bester ganzzahliger Knoten (Incumbent) unter den Knoten 0..upto.
        best = -1e18; bid = None
        for n in nodes[:upto + 1]:
            if n["status"] == "integer" and n["ub"] > best:
                best = n["ub"]; bid = n["id"]
        return bid, best

    _PURPLE = "#6A4FB0"   # aktueller Punkt & seine Iso-Linie (≠ Orange = Schnitte)
    _GREEN = "#2f6b2f"    # beste gefundene Lösung & ihre Iso-Linie

    def bnb_draw_plot(prob, nodes, step, frames, figsize=(7.0, 6.6)):
        """EIN Plot für Aufbau + Branch-and-Bound (gesteuert über einen Slider).
        step < n_build  → Aufbauphase; danach Frames (Knoten „visit"/„promote")."""
        A = prob["A"]; b = prob["b"]; c = prob["c"]
        xmax = prob["xmax"]; ymax = prob["ymax"]
        n_build = prob.get("n_build", 4)
        base = _node_poly({"lo": [0, 0], "hi": [None, None]}, A, b, xmax, ymax)
        fig, ax = plt.subplots(figsize=figsize)

        if step < n_build:
            # ── Aufbauphase: von leer schrittweise aufbauen ──
            _faint_lattice(ax, xmax, ymax)
            _titles = [
                "Aufbau 0: leeres Koordinatensystem mit ganzzahligem Gitter",
                "Aufbau 1: Restriktionen → Zulässigkeitsbereich (LP-Relaxation)",
                "Aufbau 2: zulässige (grün) vs. unzulässige (offen) Punkte",
                "Aufbau 3: LP-Optimum auf der höchsten Iso-Linie — fraktional!"]
            if step >= 1:
                _draw_constraints(ax, A, b, xmax)
                if base:
                    ax.fill([p[0] for p in base], [p[1] for p in base],
                            color=farbe_lp, alpha=0.40, zorder=1)
            if step >= 2:
                _draw_lattice(ax, A, b, xmax, ymax)
            if step >= 3:
                lp_opt = nodes[0]["x"]; zo = nodes[0]["ub"]
                _isoline(ax, c, xmax, zo, _PURPLE, 2.0)
                ax.plot(lp_opt[0], lp_opt[1], "*", ms=22, color=_PURPLE,
                        mec="white", mew=1.2, zorder=7, clip_on=False)
                _nrm = (c[0] ** 2 + c[1] ** 2) ** 0.5
                ax.annotate("", xy=(0.55 + 1.0 * c[0] / _nrm,
                                    0.55 + 1.0 * c[1] / _nrm),
                            xytext=(0.55, 0.55),
                            arrowprops=dict(arrowstyle="-|>", color=_PURPLE,
                                            lw=2.2))
                ax.text(0.45 + 1.2 * c[0] / _nrm, 0.4 + 1.2 * c[1] / _nrm,
                        "$z$ steigt", color=_PURPLE, fontsize=10,
                        fontweight="bold")
            title = _titles[min(step, 3)]
        else:
            # ── Branch-and-Bound: aktueller Knoten ──
            if base:
                ax.fill([p[0] for p in base], [p[1] for p in base],
                        color="#EEF2F8", zorder=0)
            _draw_constraints(ax, A, b, xmax)
            _draw_lattice(ax, A, b, xmax, ymax)
            node_idx, phase = frames[step - n_build]
            nd = nodes[node_idx]
            _fill = {"fractional": farbe_aktiv, "integer": farbe_incumbent,
                     "pruned": farbe_prune, "infeasible": farbe_infeasible}
            poly = _node_poly(nd, A, b, xmax, ymax)
            if poly:
                ax.fill([p[0] for p in poly], [p[1] for p in poly],
                        color=_fill[nd["status"]], alpha=0.85, zorder=2)
                ax.plot([p[0] for p in poly] + [poly[0][0]],
                        [p[1] for p in poly] + [poly[0][1]],
                        color="#33425a", lw=1.4, zorder=3)
            # Verzweigungs-Schnitte (zusätzliche Ungleichungen) — ORANGE
            lo, hi = nd["lo"], nd["hi"]
            if hi[0] is not None:
                ax.axvline(hi[0], color=IMSOrange, ls="--", lw=1.7, zorder=4)
            if lo[0] > 0:
                ax.axvline(lo[0], color=IMSOrange, ls="--", lw=1.7, zorder=4)
            if hi[1] is not None:
                ax.axhline(hi[1], color=IMSOrange, ls="--", lw=1.7, zorder=4)
            if lo[1] > 0:
                ax.axhline(lo[1], color=IMSOrange, ls="--", lw=1.7, zorder=4)
            # GRÜNE Incumbent-Iso: bei „visit" eines Integer-Knotens noch OHNE
            # den aktuellen (man sieht erst die Differenz zum Kandidaten);
            # bei „promote" springt sie auf den neuen Wert.
            _binc_upto = (node_idx - 1 if (phase == "visit"
                          and nd["status"] == "integer") else node_idx)
            bid, best = _best_upto(nodes, _binc_upto)
            if bid is not None:
                _isoline(ax, c, xmax, best, _GREEN, 2.2,
                         label=f"beste Lösung z={best:.0f}")
            if nd["x"] is None:
                # Unzulässig → Pfeil auf den Punkt AUSSERHALB (Schnitt-Ecke)
                _tx = lo[0] if lo[0] > 0 else (
                    hi[0] if hi[0] is not None else xmax * 0.5)
                _ty = lo[1] if lo[1] > 0 else (
                    hi[1] if hi[1] is not None else ymax * 0.5)
                ax.annotate("∅ unzulässig", xy=(_tx, _ty),
                            xytext=(max(_tx - 1.7, 0.1), min(_ty + 0.7, ymax)),
                            fontsize=12, fontweight="bold", color="#B0473F",
                            ha="center", zorder=8,
                            arrowprops=dict(arrowstyle="->", color="#B0473F",
                                            lw=1.8))
            elif phase == "promote":
                # Kandidat wird zur Lösung → grüner Stern auf der grünen Linie.
                ax.plot(nd["x"][0], nd["x"][1], "*", ms=24, color=_GREEN,
                        mec="white", mew=1.4, zorder=8, clip_on=False)
            else:
                # visit → LILA aktuelle Iso-Linie + Punkt (≠ Orange der Schnitte)
                _isoline(ax, c, xmax, nd["ub"], _PURPLE, 1.9,
                         label=f"aktuell z={nd['ub']:.2f}")
                ax.plot(nd["x"][0], nd["x"][1], "*", ms=20, color=_PURPLE,
                        mec="white", mew=1.1, zorder=7, clip_on=False)
            if bid is not None or (phase == "visit" and nd["x"] is not None):
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            _ub = "—" if nd["ub"] is None else f"{nd['ub']:.2f}"
            if phase == "promote":
                _suf = "neuer Incumbent ★"
            elif nd["status"] == "integer":
                _suf = "ganzzahlig — Kandidat"
            else:
                _suf = {"fractional": "fraktional", "pruned": "Schranke ✗",
                        "infeasible": "unzulässig ✗"}[nd["status"]]
            title = f"K{node_idx} · {nd['label']} · UB={_ub} · {_suf}"

        ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title(title, fontsize=11)
        fig.tight_layout()
        return fig

    def bnb_draw_tree(nodes, node_idx, phase, figsize=(7.0, 6.4)):
        pos = _tree_pos(nodes)
        # „bester bisher": bei visit eines Integer-Knotens noch ohne ihn.
        _binc_upto = (node_idx - 1 if (phase == "visit"
                      and nodes[node_idx]["status"] == "integer") else node_idx)
        binc, _ = _best_upto(nodes, _binc_upto)
        fig, ax = plt.subplots(figsize=figsize)
        for n in nodes[:node_idx + 1]:
            if n["parent"] is not None and n["parent"] <= node_idx:
                x0, y0 = pos[n["parent"]]; x1, y1 = pos[n["id"]]
                ax.plot([x0, x1], [y0, y1], "-", color="#8895a7", lw=1.2,
                        zorder=1)
        for n in nodes[:node_idx + 1]:
            x, y = pos[n["id"]]; s = n["status"]
            if n["id"] == node_idx and phase == "visit" and s == "integer":
                # Kandidat gefunden, aber noch nicht als Incumbent bestätigt.
                col = farbe_aktiv; mark = "✓ Kandidat"
            elif s == "integer":
                # Nur der AKTUELL beste Integer-Knoten ist grün. Wird er von
                # einem besseren überholt, ist er „gebounded" (rot, wie K3) —
                # so bleibt am Ende genau EIN grüner Knoten.
                if n["id"] == binc:
                    col = farbe_incumbent; mark = "✓ ★ best"
                else:
                    col = farbe_prune; mark = "✗ überholt"
            elif s == "pruned":
                col = farbe_prune; mark = "✗ Schr."
            elif s == "infeasible":
                col = farbe_infeasible; mark = "✗ unzul."
            else:
                col = farbe_lp; mark = ""
            ub = "—" if n["ub"] is None else f"{n['ub']:.1f}"
            ax.add_patch(FancyBboxPatch(
                (x - 0.47, y - 0.25), 0.94, 0.50,
                boxstyle="round,pad=0.02,rounding_size=0.07",
                fc=col, ec="#33425a", lw=1.1, zorder=2))
            ax.text(x, y + 0.09, f"K{n['id']}", ha="center", va="center",
                    fontsize=8, zorder=3)
            ax.text(x, y - 0.11, f"{n['label']}\nUB={ub} {mark}",
                    ha="center", va="center", fontsize=6, zorder=3)
        xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
        ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
        ax.axis("off"); fig.tight_layout()
        return fig
    return bnb_draw_plot, bnb_draw_tree, bnb_run


@app.cell(hide_code=True)
def gt1_data(bnb_run):
    gt1_prob = {"c": [5, 4], "A": [[1, 2], [3, 2]], "b": [7, 12],
                "xmax": 4.4, "ymax": 3.8, "n_build": 4}
    gt1_nodes, gt1_inc = bnb_run(gt1_prob["c"], gt1_prob["A"], gt1_prob["b"])
    gt1_build_narr = {
        0: "**Aufbau 0.** Wir starten mit dem leeren Koordinatensystem. Die "
           "grauen Punkte sind alle möglichen **ganzzahligen** Kombinationen "
           "$(x_1, x_2)$.",
        1: "**Aufbau 1.** Die beiden Restriktionen spannen den "
           "**Zulässigkeitsbereich** der LP-Relaxation auf (blaue Fläche).",
        2: "**Aufbau 2.** Jetzt bewerten wir die Gitterpunkte: **grün = "
           "zulässig**, **offen = unzulässig** (außerhalb). Die optimale "
           "ganzzahlige Lösung muss einer der **grünen** Punkte sein.",
        3: "**Aufbau 3.** Das **LP-Optimum** $(2.5,\\;2.25)$ liegt auf der "
           "**höchsten Iso-Linie** ($z=21.5$) — aber es ist **fraktional**. "
           "In Pfeilrichtung steigt $z$. Wir brauchen **Branch-and-Bound**, "
           "um die beste *ganzzahlige* Lösung zu finden.",
    }
    gt1_narr = {
        0: "**K0 — Wurzel.** LP-Relaxation: $x^\\star=(2.5,\\;2.25)$, "
           "$UB=21.5$ (obere Schranke). $x_1=2.5$ ist fraktional "
           "$\\Rightarrow$ verzweige auf $x_1$: $x_1\\le2$ **oder** $x_1\\ge3$.",
        1: "**K1 — $x_1\\le2$.** LP-Lösung $(2,\\;2.5)$, $UB=20$. Jetzt ist "
           "$x_2$ fraktional $\\Rightarrow$ verzweige auf $x_2$: "
           "$x_2\\le2$ **oder** $x_2\\ge3$.",
        2: "**K2 — $x_1\\le2,\\,x_2\\le2$.** LP-Lösung $(2,\\;2)$ ist "
           "**ganzzahlig** mit $z=18$ — der **erste Kandidat** (lila). Noch "
           "gibt es keinen Incumbent zum Vergleich.",
        3: "**K3 — $x_1\\le2,\\,x_2\\ge3$.** LP-Lösung $(1,\\;3)$, $UB=17$. "
           "$17 \\le 18$ (Incumbent) $\\Rightarrow$ **abschneiden durch "
           "Schranke** ✗. (Selbst dieser ganzzahlige Punkt bringt keine "
           "Verbesserung — **Bound-Pruning**.)",
        4: "**K4 — $x_1\\ge3$.** LP-Lösung $(3,\\;1.5)$, $UB=21>18$ — hier "
           "kann es noch besser werden! $x_2$ fraktional $\\Rightarrow$ "
           "verzweige auf $x_2$: $x_2\\le1$ **oder** $x_2\\ge2$.",
        5: "**K5 — $x_1\\ge3,\\,x_2\\le1$.** LP-Lösung $(3.33,\\;1)$, "
           "$UB=20.67$. $x_1$ fraktional $\\Rightarrow$ verzweige auf $x_1$: "
           "$x_1\\le3$ **oder** $x_1\\ge4$.",
        6: "**K6 — $\\dots,\\,x_1\\le3$** ($x_1=3,\\,x_2=1$). Ganzzahlig mit "
           "$z=19$: die **lila Kandidaten-Linie (19)** liegt **über** der "
           "grünen Incumbent-Linie (18) → **besser!** (Differenz sichtbar.)",
        7: "**K7 — $\\dots,\\,x_1\\ge4$** ($x_1=4,\\,x_2=0$). Ganzzahlig mit "
           "$z=20$ — die lila Linie (20) liegt **über** der grünen (19) → "
           "noch besser!",
        8: "**K8 — $x_1\\ge3,\\,x_2\\ge2$.** Das LP ist **unzulässig** — kein "
           "Punkt erfüllt $3x_1+2x_2\\le12$ zusammen mit $x_1\\ge3, x_2\\ge2$ "
           "$\\Rightarrow$ **abschneiden, unzulässig** ✗. Optimum: "
           "$(4,\\;0)$, $z=20$.",
    }
    gt1_promote_narr = {
        2: "**Incumbent gesetzt.** Der Kandidat $z=18$ wird zum **Incumbent** "
           "— die grüne Iso-Linie steht jetzt bei $18$, der Punkt $(2,2)$ "
           "wird grün markiert.",
        6: "**Incumbent-Update.** Die grüne Linie **springt von 18 auf 19**, "
           "der neue beste Punkt $(3,1)$ wird grün. K2 ist damit **überholt**.",
        7: "**Incumbent-Update.** Die grüne Linie **springt auf 20**, der "
           "Punkt $(4,0)$ wird grün — das ist bereits das **Optimum**.",
    }
    gt1_frames = []
    for _n in gt1_nodes:
        gt1_frames.append((_n["id"], "visit"))
        if _n["status"] == "integer":
            gt1_frames.append((_n["id"], "promote"))
    return (gt1_build_narr, gt1_frames, gt1_narr, gt1_nodes,
            gt1_promote_narr, gt1_prob)


@app.cell(hide_code=True)
def gt1_controls(gt1_frames, gt1_prob, mo):
    # EIN Slider: erst Aufbau (0 .. n_build-1), dann B&B-Frames
    # (jeder Knoten „visit"; verbessernde Integer-Knoten zusätzlich „promote").
    _total = gt1_prob["n_build"] + len(gt1_frames) - 1
    gt1_step = mo.ui.slider(
        start=0, stop=_total, step=1, value=0,
        label="Schritt", show_value=True)
    return (gt1_step,)


@app.cell(hide_code=True)
def gt1_view(
    bnb_draw_plot,
    bnb_draw_tree,
    gt1_build_narr,
    gt1_frames,
    gt1_narr,
    gt1_nodes,
    gt1_promote_narr,
    gt1_prob,
    gt1_step,
    mo,
):
    _intro = mo.md(r"""
    ## 2 · Geführte Übung 1 — Branch-and-Bound grafisch

    Eine Werkstatt fertigt zwei **unteilbare** Produkttypen (nur ganze Stück) und
    maximiert den Deckungsbeitrag. Mit **einem Regler** bauen wir erst die
    LP-Lösung auf und durchlaufen dann Branch-and-Bound — alles im **selben
    Graphen** (links), der Suchbaum wächst rechts mit.

    $$
    \begin{aligned}
    \max \quad & z = 5x_1 + 4x_2 \\
    \text{u.d.N.} \quad & x_1 + 2x_2 \le 7 \\
    & 3x_1 + 2x_2 \le 12 \\
    & x_1,\, x_2 \in \mathbb{Z}_{\ge 0}
    \end{aligned}
    $$
    """)

    _nb = gt1_prob["n_build"]
    _step = int(gt1_step.value)
    _plot = bnb_draw_plot(gt1_prob, gt1_nodes, _step, gt1_frames)

    if _step < _nb:
        _txt = gt1_build_narr.get(_step, "")
        _visual = mo.hstack([_plot], justify="center")
    else:
        _node_idx, _phase = gt1_frames[_step - _nb]
        _txt = (gt1_promote_narr.get(_node_idx, "") if _phase == "promote"
                else gt1_narr.get(_node_idx, ""))
        _visual = mo.hstack(
            [_plot, bnb_draw_tree(gt1_nodes, _node_idx, _phase)],
            justify="center", align="center")

    _solution = mo.accordion({
        "🔑 Zusammenfassung & PuLP (erst selbst durchklicken!)": mo.md(r"""
    | K | Bedingung | LP-Lösung | UB | Status |
    |---|---|---|---|---|
    | K0 | Wurzel | $(2.5,\,2.25)$ | 21.5 | fraktional → split $x_1$ |
    | K1 | $x_1\le2$ | $(2,\,2.5)$ | 20.0 | fraktional → split $x_2$ |
    | K2 | $x_1\le2,x_2\le2$ | $(2,\,2)$ | 18.0 | ganzzahlig → Incumbent 18 |
    | K3 | $x_1\le2,x_2\ge3$ | $(1,\,3)$ | 17.0 | **Schranke** ✗ ($17\le18$) |
    | K4 | $x_1\ge3$ | $(3,\,1.5)$ | 21.0 | fraktional → split $x_2$ |
    | K5 | $x_1\ge3,x_2\le1$ | $(3.33,\,1)$ | 20.67 | fraktional → split $x_1$ |
    | K6 | $\dots,x_1\le3$ | $(3,\,1)$ | 19.0 | ganzzahlig → Incumbent 19 |
    | K7 | $\dots,x_1\ge4$ | $(4,\,0)$ | 20.0 | ganzzahlig → **Optimum 20** ★ |
    | K8 | $x_1\ge3,x_2\ge2$ | — | — | **unzulässig** ✗ |

    **Optimum: $x_1=4,\,x_2=0$, $z=20$.** Der Incumbent verbessert sich
    $18 \to 19 \to 20$ — ältere ganzzahlige Lösungen werden **überholt**, am
    Ende bleibt im Baum genau **ein grüner** Knoten (K7). Zwei Arten des
    Abschneidens: K3 per **Schranke** ($UB \le$ Incumbent), K8 wegen
    **Unzulässigkeit** (leeres LP).

    ```python
    from pulp import LpProblem, LpVariable, LpMaximize, value
    m  = LpProblem("werkstatt", LpMaximize)
    x1 = LpVariable("x1", lowBound=0, cat="Integer")
    x2 = LpVariable("x2", lowBound=0, cat="Integer")
    m += 5*x1 + 4*x2
    m += 1*x1 + 2*x2 <= 7
    m += 3*x1 + 2*x2 <= 12
    m.solve()        # -> x1 = 4, x2 = 0, z = 20
    ```
    """)
    })

    mo.vstack([
        _intro,
        gt1_step,
        mo.md(_txt),
        _visual,
        _solution,
    ])
    return


@app.cell(hide_code=True)
def gt2_intro(mo):
    mo.md(r"""
    # 3 · Geführte Übung 2 — Kraftwerkseinsatz

    > **Die Stadtwerke** müssen für die morgige Spitzenstunde eine Last von
    > **100 MW** bereitstellen. Dafür stehen drei Kraftwerke zur Verfügung. Für
    > jedes Werk ist zu entscheiden, **ob es überhaupt angefahren wird** und, falls
    > ja, **mit welcher Leistung** es fährt.
    >
    > Beim Anfahren eines Werks entstehen einmalige **Anfahrkosten**; jede
    > erzeugte MW kostet zusätzlich einen werksspezifischen **variablen Preis**.
    > Aus technischen Gründen lässt sich ein **laufendes** Werk nicht beliebig
    > schwach fahren — es gibt eine **Mindestlast** — und natürlich auch eine
    > **Höchstlast**. Ist ein Werk aus, liefert es 0 MW und verursacht keine
    > Anfahrkosten.
    >
    > Die beiden Altblöcke **A und B** hängen am selben Netzanschluss; deshalb darf
    > **höchstens einer** der beiden laufen.

    | Kraftwerk | var. Kosten (€/MW) | Anfahrkosten (€) | Mindestlast (MW) | Höchstlast (MW) |
    |---|:-:|:-:|:-:|:-:|
    | A (alt) | 8 | 200 | 20 | 60 |
    | B (neu) | 6 | 150 | 15 | 50 |
    | C | 7 | 250 | 25 | 70 |

    **Aufgabe.** Modellieren Sie den kostenminimalen Einsatzplan als MIP
    (Variablen, Zielfunktion, Nebenbedingungen) und bestimmen Sie die Lösung.
    """)
    return


@app.cell(hide_code=True)
def gt2_model(mo, pl):
    _P = ["A", "B", "C"]
    _D = 100
    _qmax = {"A": 60, "B": 50, "C": 70}
    _qmin = {"A": 20, "B": 15, "C": 25}
    _v = {"A": 8, "B": 6, "C": 7}
    _f = {"A": 200, "B": 150, "C": 250}
    _m = pl.LpProblem("kraftwerk", pl.LpMinimize)
    _q = {p: pl.LpVariable(f"q_{p}", lowBound=0) for p in _P}
    _y = {p: pl.LpVariable(f"y_{p}", cat="Binary") for p in _P}
    _m += pl.lpSum(_v[p] * _q[p] + _f[p] * _y[p] for p in _P)
    _m += pl.lpSum(_q[p] for p in _P) >= _D
    for p in _P:
        _m += _q[p] <= _qmax[p] * _y[p]
        _m += _q[p] >= _qmin[p] * _y[p]
    _m += _y["A"] + _y["B"] <= 1
    _m.solve()
    _rows = "\n".join(
        f"| {p} | {'**an**' if _y[p].varValue > 0.5 else 'aus'} | "
        f"{_q[p].varValue:.0f} MW |" for p in _P)
    mo.accordion({
        "🔑 Lösung Geführt 2": mo.md(rf"""
    **Variablen.** Für jedes Werk $p$: Binärvariable $y_p\in\{{0,1\}}$ (1 = Werk
    läuft) und Last $q_p \ge 0$ (MW).

    **Zielfunktion** — variable Kosten plus Anfahrkosten nur bei laufendem Werk:

    $$ \min \sum_p \big( v_p\,q_p + f_p\,y_p \big) $$

    **Nebenbedingungen.**

    - **Nachfrage decken:** $\;\sum_p q_p \ge D = 100$
    - **An/Aus koppelt Last (Höchstlast + Fixkosten-Trigger):**
      $\;q_p \le q^{{\max}}_p\,y_p$. Ist $y_p=0$, erzwingt das $q_p=0$; ist $y_p=1$,
      darf das Werk bis zur Höchstlast fahren.
    - **Mindestlast, falls in Betrieb:** $\;q_p \ge q^{{\min}}_p\,y_p$. Greift nur
      bei $y_p=1$ (sonst $q_p \ge 0$).
    - **Entweder-oder (A/B teilen den Netzanschluss):** $\;y_A + y_B \le 1$

    Zusammen:

    $$
    \begin{{aligned}}
    \min \quad & \sum_p \big( v_p\,q_p + f_p\,y_p \big) \\
    \text{{s.t.}} \quad & \sum_p q_p \ge D \\
    & q^{{\min}}_p\,y_p \le q_p \le q^{{\max}}_p\,y_p && \forall p \\
    & y_A + y_B \le 1 \\
    & q_p \ge 0,\;\; y_p \in \{{0,1\}} && \forall p
    \end{{aligned}}
    $$

    Im Browser via HiGHS gelöst:

    | Kraftwerk | Status | Last |
    |---|:-:|:-:|
    {_rows}

    **Gesamtkosten: {pl.value(_m.objective):.0f} €.** Intuition: das günstige Werk
    B (6 €/MW) läuft voll, C ergänzt den Rest — A bleibt aus (teurer **und** durch
    den Ausschluss mit B blockiert).
    """),
    })
    return


@app.cell(hide_code=True)
def selbst_header(mo):
    mo.md(r"""
    # 4 · Selbständige Aufgaben

    Jetzt selbst formulieren — **Textaufgaben im Klausurformat.** Erst auf Papier
    versuchen, dann die Lösung im Akkordeon aufklappen. Die Aufgaben werden von
    1 nach 4 **schwerer**.
    """)
    return


@app.cell(hide_code=True)
def a1_task(mo):
    mo.md(r"""
    ## Aufgabe 1 — Projektauswahl in der Forschung

    Ein **Pharmaunternehmen** prüft sieben Forschungsprojekte. Jedes Projekt, das
    gestartet wird, verspricht einen erwarteten **Kapitalwert** (Mio €) und bindet
    dafür zwei knappe Ressourcen: **Forschungsbudget** (Mio €) und **Laborzeit**
    (Personenmonate). Ein Projekt kann nur **ganz oder gar nicht** gestartet werden.

    | Projekt | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
    |---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    | Kapitalwert (Mio €) | 40 | 25 | 30 | 18 | 22 | 15 | 20 |
    | Budget (Mio €) | 3 | 2 | 4 | 1 | 2 | 1 | 2 |
    | Laborzeit (PM) | 2 | 1 | 3 | 2 | 2 | 1 | 1 |

    Insgesamt stehen **10 Mio € Budget** und **9 Personenmonate Laborzeit** zur
    Verfügung. Gesucht ist die **wertmaximale** Projektauswahl.

    **(a)** Formulieren Sie das Grundproblem als IP (Variablen, Ziel, NB).

    **(b)** Das Management nennt zusätzliche Wünsche. Formalisieren Sie jeden
    **einzeln** als Nebenbedingung:

    1. Aus strategischen Gründen sollen **mindestens drei** Projekte starten.
    2. Die Projekte 1 und 4 nutzen dasselbe Patent — wird **Projekt 1** gestartet,
       darf **Projekt 4 nicht** starten.
    3. Projekt 3 ist nur sinnvoll mit Vorarbeiten: startet **Projekt 3**, müssen
       **6 und 7** ebenfalls starten.
    4. Von den drei Konkurrenzprojekten **2, 3 und 5** dürfen **höchstens zwei**
       gleichzeitig laufen.
    """)
    return


@app.cell(hide_code=True)
def a1_solution(mo, pl):
    def _demo():
        g = {1: 40, 2: 25, 3: 30, 4: 18, 5: 22, 6: 15, 7: 20}
        a = {1: (3, 2), 2: (2, 1), 3: (4, 3), 4: (1, 2),
             5: (2, 2), 6: (1, 1), 7: (2, 1)}
        b = (10, 9)
        m = pl.LpProblem("capbudget", pl.LpMaximize)
        x = {i: pl.LpVariable(f"x_{i}", cat="Binary") for i in g}
        m += pl.lpSum(g[i] * x[i] for i in g)
        for j in (0, 1):
            m += pl.lpSum(a[i][j] * x[i] for i in g) <= b[j]
        m.solve()
        return [i for i in g if x[i].varValue > 0.5], pl.value(m.objective)

    _ch, _z = _demo()
    mo.accordion({
        "🔑 Lösung Aufgabe 1": mo.md(rf"""
    **(a) Grundmodell.** Binärvariable $x_i\in\{{0,1\}}$: 1, falls Projekt $i$ gewählt.

    $$
    \begin{{aligned}}
    \max \quad & \sum_{{i\in I}} g_i x_i \\
    \text{{s.t.}} \quad & \sum_{{i\in I}} a_{{ij}}\,x_i \le b_j && \forall j\in J \\
    & x_i \in \{{0,1\}} && \forall i\in I
    \end{{aligned}}
    $$

    **(b) Zusatzrestriktionen** (Muster aus der Cheatsheet):

    1. „mindestens drei Projekte": $\displaystyle\sum_{{i\in I}} x_i \ge 3$
    2. „falls 1 dann nicht 4" = Ausschluss: $x_1 + x_4 \le 1$
    3. „falls 3 dann 6 und 7": $x_3 \le x_6$ **und** $x_3 \le x_7$
    4. „von $\{{2,3,5\}}$ höchstens 2": $x_2 + x_3 + x_5 \le 2$

    ---

    **Konkrete Mini-Instanz** ($n=7$, 2 Rohstoffe, ohne Zusatz-NB), per Solver:
    gewählt werden Projekte **{_ch}** mit Gesamtgewinn **{_z:.0f}**.
    """),
    })
    return


@app.cell(hide_code=True)
def a2_task(mo):
    mo.md(r"""
    ## Aufgabe 2 — Auftragsplanung in einer Druckerei

    Eine **Druckerei** hat heute fünf Aufträge von Werbekunden in der Warteschlange
    und möchte sie **so schnell wie möglich alle fertigstellen** (die letzte
    Fertigstellung soll möglichst früh liegen). Jeder Auftrag durchläuft je nach
    Motiv eine **vorgegebene Reihenfolge** von Druckschritten auf einer **grünen**,
    **blauen** und **gelben** Presse. Von jeder Farbpresse gibt es im Betrieb
    **genau eine**, und sie kann **nur einen Auftrag zur Zeit** bearbeiten. Ein
    einmal begonnener Druckschritt wird ohne Unterbrechung zu Ende geführt.
    Sequenzen und Bearbeitungsdauern (in Minuten):

    | Job | Sequenz | Dauern |
    |---|---|---|
    | 1 | blau → gelb | $p_{1,\text{blau}}=10,\;p_{1,\text{gelb}}=5$ |
    | 2 | blau → grün → gelb | $p_{2,\text{blau}}=5,\;p_{2,\text{grün}}=15,\;p_{2,\text{gelb}}=10$ |
    | 3 | grün → blau → gelb | $p_{3,\text{grün}}=5,\;p_{3,\text{blau}}=5,\;p_{3,\text{gelb}}=10$ |
    | 4 | gelb → grün | $p_{4,\text{gelb}}=10,\;p_{4,\text{grün}}=5$ |
    | 5 | grün → gelb → blau | $p_{5,\text{grün}}=15,\;p_{5,\text{gelb}}=5,\;p_{5,\text{blau}}=5$ |

    **Aufgabe.** Modellieren Sie als MIP mit dem Ziel, die **Gesamtdauer**
    (Makespan $c_{\max}$) zu minimieren. *Tipp:* Pro Presse kann immer nur **ein**
    Job gleichzeitig laufen — das verlangt eine **Big-$M$**-Reihenfolgelogik.
    """)
    return


@app.cell(hide_code=True)
def a2_solution(mo, pl, plt):
    _seq = {1: ["blau", "gelb"], 2: ["blau", "gruen", "gelb"],
            3: ["gruen", "blau", "gelb"], 4: ["gelb", "gruen"],
            5: ["gruen", "gelb", "blau"]}
    _p = {(1, "blau"): 10, (1, "gelb"): 5, (2, "blau"): 5, (2, "gruen"): 15,
          (2, "gelb"): 10, (3, "gruen"): 5, (3, "blau"): 5, (3, "gelb"): 10,
          (4, "gelb"): 10, (4, "gruen"): 5, (5, "gruen"): 15, (5, "gelb"): 5,
          (5, "blau"): 5}
    _jobs = list(_seq)
    _presses = ["gruen", "blau", "gelb"]
    _M = sum(_p.values())
    _m = pl.LpProblem("druckerei", pl.LpMinimize)
    _s = {(i, k): pl.LpVariable(f"s_{i}_{k}", lowBound=0)
          for i in _jobs for k in _seq[i]}
    _e = {(i, k): pl.LpVariable(f"e_{i}_{k}", lowBound=0)
          for i in _jobs for k in _seq[i]}
    _cmax = pl.LpVariable("cmax", lowBound=0)
    _x = {}
    for k in _presses:
        _u = [i for i in _jobs if k in _seq[i]]
        for _a in range(len(_u)):
            for _bk in range(_a + 1, len(_u)):
                _x[(_u[_a], _u[_bk], k)] = pl.LpVariable(
                    f"x_{_u[_a]}_{_u[_bk]}_{k}", cat="Binary")
    _m += _cmax
    for i in _jobs:
        for k in _seq[i]:
            _m += _e[(i, k)] == _s[(i, k)] + _p[(i, k)]
            _m += _cmax >= _e[(i, k)]
        for _a in range(len(_seq[i]) - 1):
            _m += _s[(i, _seq[i][_a + 1])] >= _e[(i, _seq[i][_a])]
    for (i, j, k), _xv in _x.items():
        _m += _e[(j, k)] <= _s[(i, k)] + _xv * _M
        _m += _e[(i, k)] <= _s[(j, k)] + (1 - _xv) * _M
    _m.solve()
    _farben = {"gruen": "#5BA85B", "blau": "#3B6FB0", "gelb": "#E0B33A"}
    _lane = {"gruen": 2, "blau": 1, "gelb": 0}
    _fig, _ax = plt.subplots(figsize=(8.5, 2.8))
    for (i, k), _sv in _s.items():
        _ax.barh(_lane[k], _p[(i, k)], left=_sv.varValue, height=0.6,
                 color=_farben[k], edgecolor="white")
        _ax.text(_sv.varValue + _p[(i, k)] / 2, _lane[k], f"J{i}",
                 ha="center", va="center", color="white", fontsize=8,
                 fontweight="bold")
    _ax.set_yticks([0, 1, 2])
    _ax.set_yticklabels(["gelb", "blau", "grün"])
    _ax.set_xlabel("Zeit")
    _ax.set_xlim(0, _cmax.varValue + 2)
    _ax.set_title(f"Optimaler Belegungsplan · Makespan = {_cmax.varValue:.0f}")
    _fig.tight_layout()
    mo.accordion({
        "🔑 Lösung Aufgabe 2": mo.vstack([mo.md(r"""
    **Entscheidungsvariablen — woher?** Drei Fragen: **(1) Wann beginnt jeder
    Schritt?** → kontinuierliche **Startzeit** $s_{ik}\ge 0$ (Endzeit
    $e_{ik}=s_{ik}+p_{ik}$ als Hilfsvariable). **(2) Was minimieren?** → **Makespan**
    $c_{\max}\ge 0$. **(3) Reihenfolge zweier Jobs auf einer Presse?** →
    Ja/Nein-Entscheidung, nicht über Zeiten ausdrückbar → **Binärvariable**
    $x_{ijk}\in\{0,1\}$ ($=1$, falls $i$ vor $j$) je Paar $i<j$.
    *(Faustregel: Zeiten/Mengen kontinuierlich, Entweder-oder/Reihenfolge binär.)*
    **Mengen:** $J$ Jobs, $K_i$ vorgegebene Pressenfolge von Job $i$,
    $J_k=\{i\in J:\, k\in K_i\}$ Jobs auf Presse $k$; $p_{ik}$ Bearbeitungsdauer,
    $M=\sum p_{ik}$.

    $$
    \begin{aligned}
    \min \quad & c_{\max} \\
    \text{s.t.} \quad & e_{ik} = s_{ik} + p_{ik} && \forall\, i\in J,\ k\in K_i \\
    & c_{\max} \ge e_{ik} && \forall\, i\in J,\ k\in K_i \\
    & s_{i,k_2} \ge e_{i,k_1} && \forall\, i\in J,\ (k_1\!\to\! k_2)\in K_i \\
    & e_{jk} \le s_{ik} + x_{ijk}\,M && \forall\, k\in K,\ i<j\in J_k \\
    & e_{ik} \le s_{jk} + (1 - x_{ijk})\,M && \forall\, k\in K,\ i<j\in J_k \\
    & s_{ik}, e_{ik}, c_{\max} \ge 0,\;\; x_{ijk} \in \{0,1\}
    \end{aligned}
    $$

    **Nebenbedingungen — woher?**

    - **Endzeit:** Operation belegt die Presse ununterbrochen $p_{ik}$ lang →
      $e_{ik}=s_{ik}+p_{ik}$.
    - **Makespan:** $c_{\max}\ge e_{ik}$ für alle Operationen; Minimierung drückt
      $c_{\max}$ auf die größte Endzeit.
    - **Reihenfolge im Job:** folgt $k_2$ auf $k_1$ in der *vorgegebenen Sequenz*
      des Jobs, dann $s_{i,k_2}\ge e_{i,k_1}$. Diese Reihenfolge ist *gegeben* (Teil
      der Aufgabendaten), keine Entscheidung.
    - **Maschinenkonflikt (Big-$M$):** $i,j$ dürfen sich auf Presse $k$ nicht
      überlappen. $x_{ijk}=1$ macht die erste Ungleichung durch $M$ wirkungslos und
      hält $e_{ik}\le s_{jk}$ aktiv ($i$ vor $j$); $x_{ijk}=0$ umgekehrt — so gilt
      stets **genau eine** Reihenfolge, mit $M=\sum p_{ik}$ groß genug, dass die
      andere nie bindet.
    - **Warum nur $i<j$?** Ein *einziges* $x_{ijk}$ pro *ungeordnetem* Paar legt die
      Richtung schon fest. „Für alle $i,j$" würde jedes Paar doppelt zählen (man
      bräuchte zusätzlich $x_{jik}$ und die Kopplung $x_{ijk}+x_{jik}=1$) und
      erzeugte redundante Bedingungen — daher genau ein Paar $i<j$.
    """), _fig]),
    })
    return


@app.cell(hide_code=True)
def a3_task(mo):
    mo.md(r"""
    ## Aufgabe 3 — Filialnetz einer Supermarktkette

    Eine **Supermarktkette** will in einer Stadt mit **7 Stadtteilen** vertreten
    sein. Dafür hat sie **5 mögliche Grundstücke** für neue Filialen im Blick.
    Jede Filiale versorgt aufgrund ihrer Lage einen bestimmten Kranz an Stadtteilen
    (die fußläufig bzw. mit kurzer Fahrt erreichbar sind). **Jeder Stadtteil** muss
    von **mindestens einer** Filiale abgedeckt werden; die **Bau- und
    Eröffnungskosten** sollen dabei so gering wie möglich sein.

    Die folgende Tabelle zeigt, welche Stadtteile ein Grundstück abdeckt (×) und was
    die Eröffnung dort kostet:

    | Stadtteil \ Grundstück | 1 | 2 | 3 | 4 | 5 |
    |---|:-:|:-:|:-:|:-:|:-:|
    | 1 | × |   | × |   |   |
    | 2 | × | × |   | × | × |
    | 3 |   | × |   | × |   |
    | 4 |   |   | × |   | × |
    | 5 | × | × |   |   |   |
    | 6 |   |   | × |   | × |
    | 7 |   |   |   | × | × |
    | **Kosten** (1000 €) | 450 | 650 | 550 | 500 | 525 |

    **(a)** Formulieren Sie die kostenminimale, vollständige Abdeckung als IP.

    **(b)** Ergänzen Sie folgende Vorgaben der Geschäftsführung als
    Nebenbedingungen: es sollen **mindestens 2 Filialen** eröffnet werden; die
    Grundstücke **2 und 5** schließen sich aus (zu nah beieinander); und **falls
    Grundstück 1** eröffnet wird, dürfen **2 und 3 nicht beide** zugleich öffnen.
    """)
    return


@app.cell(hide_code=True)
def a3_solution(mo, pl):
    def _demo():
        cost = {1: 450, 2: 650, 3: 550, 4: 500, 5: 525}
        cover = {1: [1, 3], 2: [1, 2, 4, 5], 3: [2, 4], 4: [3, 5],
                 5: [1, 2], 6: [3, 5], 7: [4, 5]}
        m = pl.LpProblem("setcover", pl.LpMinimize)
        x = {i: pl.LpVariable(f"x_{i}", cat="Binary") for i in cost}
        m += pl.lpSum(cost[i] * x[i] for i in cost)
        for _sts in cover.values():
            m += pl.lpSum(x[i] for i in _sts) >= 1
        m += pl.lpSum(x[i] for i in cost) >= 2
        m += x[2] + x[5] <= 1
        m += x[1] + x[2] + x[3] <= 2
        m.solve()
        return [i for i in cost if x[i].varValue > 0.5], pl.value(m.objective)

    _open, _c = _demo()
    mo.accordion({
        "🔑 Lösung Aufgabe 3": mo.md(rf"""
    **(a) Grundmodell (Set-Covering).** $x_i\in\{{0,1\}}$: 1, falls auf Grundstück
    $i$ eine Filiale eröffnet wird. $c_i$ = Eröffnungskosten, $d_{{ij}}=1$, falls
    Grundstück $i$ den Stadtteil $j$ abdeckt (× in der Tabelle).

    $$
    \begin{{aligned}}
    \min \quad & \sum_i c_i x_i \\
    \text{{s.t.}} \quad & \sum_i d_{{ij}}\,x_i \ge 1 && \forall \text{{ Stadtteil }} j \\
    & x_i \in \{{0,1\}} && \forall i
    \end{{aligned}}
    $$

    Die Überdeckungs-NB sagt: **jeder** Stadtteil wird von **mindestens einer**
    eröffneten Filiale bedient.

    **(b) Zusatzrestriktionen.**

    - „mindestens 2 eröffnen": $\sum_i x_i \ge 2$
    - „2 und 5 schließen sich aus": $x_2 + x_5 \le 1$
    - „falls 1 öffnet, nicht 2 und 3 zugleich": $x_1 + x_2 + x_3 \le 2$

    ---

    **Per Solver:** eröffnet werden die Grundstücke **{_open}**, Gesamtkosten
    **{_c:.0f}** (× 1000 €).
    """),
    })
    return


@app.cell(hide_code=True)
def a4_task(mo):
    mo.md(r"""
    ## Aufgabe 4 — Personalplanung an einer Schule · Bonus

    Eine **Schule** stellt den Stundenplan für einen Vormittag auf und möchte dabei
    mit **möglichst wenigen Lehrkräften** auskommen, da jede zusätzlich eingesetzte
    Lehrkraft Kosten verursacht. Der Unterricht verteilt sich auf **drei Räume**
    ($r=1,\dots,3$) und **sechs Schulstunden** ($h=1,\dots,6$). Es gibt **acht
    Fächer** ($s=1,\dots,8$) und **vier Jahrgangsstufen** ($b=1,\dots,4$); zur
    Verfügung steht ein Pool von bis zu **100 Lehrkräften** ($i=1,\dots,100$).

    Welche Fächer welche Jahrgangsstufe braucht, gibt der Lehrplan vor: $a_{s,b}=1$,
    falls Fach $s$ in Jahrgang $b$ unterrichtet wird (sonst 0). Jede solche
    Fach-Jahrgang-Einheit ist **genau einmal** zu halten. Nicht jede Lehrkraft darf
    jedes Fach unterrichten: $\beta_{i,s}=1$, falls Lehrkraft $i$ das Fach $s$
    unterrichten darf (sonst 0).

    Alle Lehrkräfte sind zu allen Stunden verfügbar, und jedes Fach kann in jedem
    Raum unterrichtet werden. Pro Raum und Stunde findet höchstens eine Stunde
    statt, und weder eine Lehrkraft noch eine Jahrgangsstufe kann zur selben Stunde
    an zwei Orten sein.
    """)
    return


@app.cell(hide_code=True)
def a4_solution(mo):
    mo.accordion({
        "🔑 Lösung Aufgabe 4 (Bonus)": mo.md(r"""
    **Variablen.**
    $x_{s,b,h,r,i}\in\{0,1\}$: 1, falls Fach $s$ der Jahrgangsstufe $b$ in Stunde $h$
    in Raum $r$ von Lehrkraft $i$ unterrichtet wird.
    $y_i\in\{0,1\}$: 1, falls Lehrkraft $i$ überhaupt eingesetzt wird.

    **Modell.**

    $$
    \begin{aligned}
    \min \quad & \sum_{i} y_i \\
    \text{s.t.} \quad & \sum_{h}\sum_{r}\sum_{i} x_{s,b,h,r,i} = a_{s,b} && \forall s, b \\
    & \sum_{s}\sum_{b}\sum_{i} x_{s,b,h,r,i} \le 1 && \forall r, h \\
    & \sum_{s,b,r} x_{s,b,h,r,i} \le 1 && \forall i, h \\
    & \sum_{s,r,i} x_{s,b,h,r,i} \le 1 && \forall b, h \\
    & x_{s,b,h,r,i} \le \beta_{i,s} && \forall s,b,h,r,i \\
    & \sum_{s,b,h,r} x_{s,b,h,r,i} \le M\, y_i && \forall i \\
    & x_{s,b,h,r,i} \in \{0,1\},\;\; y_i \in \{0,1\}
    \end{aligned}
    $$

    Zeile für Zeile: jede Fach-Jahrgang-Einheit **genau einmal**; pro **Raum &
    Stunde** höchstens eine Einheit; jede **Lehrkraft** und jede **Jahrgangsstufe**
    nicht doppelt zur selben Stunde; nur **qualifizierte** Lehrkräfte; und die
    Big-$M$-Kopplung an $y_i$.

    Der Kniff ist die letzte **Big-$M$**-Bedingung: sobald **irgendeine** Einheit von
    Lehrkraft $i$ unterrichtet wird, wird $y_i=1$ erzwungen — und $y_i$ wird im Ziel
    bestraft. So minimiert das Modell die Zahl eingesetzter Lehrkräfte.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
