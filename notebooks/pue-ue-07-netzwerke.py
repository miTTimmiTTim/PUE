# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pulp",
#     "networkx",
#     "scipy",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(
    width="full",
    app_title="PuE Übung 7: Netzwerke",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import networkx as nx
    import numpy as np
    import pulp as pl
    return mo, np, nx, pl


@app.cell(hide_code=True)
def solver(pl):
    # ─────────────────────────────────────────────────────────────────────
    # Hintergrund-Solver — löst PuLP-Modelle via scipy/HiGHS.
    # Funktioniert auch im Browser (Pyodide/WASM), wo CBC nicht verfügbar ist.
    # Setzt LpSolverDefault, sodass plain `modell.solve()` den Shim nutzt.
    # Studierende sehen davon nichts.
    # ─────────────────────────────────────────────────────────────────────
    import numpy as _np
    from scipy.optimize import linprog

    def _solve_prob(prob):
        variables = prob.variables()
        n = len(variables)
        idx = {v.name: i for i, v in enumerate(variables)}

        c = _np.zeros(n)
        if prob.objective is not None:
            for v, coef in prob.objective.items():
                c[idx[v.name]] = coef
        is_max = (prob.sense == pl.LpMaximize)
        if is_max:
            c = -c

        bounds = [(v.lowBound, v.upBound) for v in variables]

        A_ub_rows, b_ub_vals = [], []
        A_eq_rows, b_eq_vals = [], []
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

        A_ub = _np.array(A_ub_rows) if A_ub_rows else None
        b_ub = _np.array(b_ub_vals) if b_ub_vals else None
        A_eq = _np.array(A_eq_rows) if A_eq_rows else None
        b_eq = _np.array(b_eq_vals) if b_eq_vals else None

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")
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
    solver_ready = True  # Sentinel: erzwingt Reihenfolge solver → Engine
    return (solver_ready,)


@app.cell(hide_code=True)
def colors():
    IMSBlue = "#023B88"
    IMSOrange = "#D87237"
    farbe_quelle = "#A9D18E"      # Angebot / Quelle (grün)
    farbe_senke = "#E8A6A0"       # Nachfrage / Senke (rot)
    farbe_zwischen = "#9FBEE6"    # Zwischenknoten (blau)
    farbe_terminal = "#F2C14E"    # Quelle/Senke im Max-Flow (gold)
    return (
        IMSBlue, IMSOrange, farbe_quelle, farbe_senke, farbe_terminal, farbe_zwischen,
    )


@app.cell(hide_code=True)
def engine(
    IMSBlue,
    IMSOrange,
    farbe_quelle,
    farbe_senke,
    farbe_terminal,
    farbe_zwischen,
    nx,
    pl,
    solver_ready,
):
    # ─────────────────────────────────────────────────────────────────────
    # Netzwerk-Engine: Graph aus Spezifikation bauen, lösen, zeichnen.
    # Eine Spec ist ein dict:
    #   kind   : "maxflow" | "transport"
    #   pos    : {knoten: (x, y)}
    #   edges  : [(u, v, wert[, yield])]   wert = Kapazität (maxflow) bzw. Kosten (transport)
    #   src/snk: Quelle/Senke (nur maxflow)
    #   balance: {knoten: bilanz}          (+Angebot / -Nachfrage; nur transport)
    #   colors : {knoten: farbe}           (optional)
    # ─────────────────────────────────────────────────────────────────────
    _ = solver_ready  # Reihenfolge erzwingen
    BIG = 10**6

    def build_graph(spec, overrides=None):
        """nx.DiGraph aus Spec bauen (Kantenwerte ggf. via overrides ersetzt)."""
        overrides = overrides or {}
        G = nx.DiGraph()
        for k, v in spec.get("pos", {}).items():
            G.add_node(k)
        wkey = "cap" if spec["kind"] == "maxflow" else "cost"
        for e in spec["edges"]:
            u, v, w = e[0], e[1], e[2]
            yfac = e[3] if len(e) > 3 else 1.0
            w = overrides.get((u, v), w)
            G.add_edge(u, v, **{wkey: w, "yield": yfac})
        if spec["kind"] == "transport":
            for nd, b in spec.get("balance", {}).items():
                G.nodes[nd]["balance"] = b
        return G

    def solve_spec(spec, overrides=None):
        """Graph bauen + lösen. Setzt flow/utilization auf die Kanten.
        Rückgabe: (G, zielwert)."""
        G = build_graph(spec, overrides)
        if spec["kind"] == "maxflow":
            src, snk = spec["src"], spec["snk"]
            H = G.copy()
            H.add_edge(snk, src, cap=BIG, ret=True)
            m = pl.LpProblem("MaxFlow", pl.LpMaximize)
            x = {(u, v): pl.LpVariable(f"x_{u}_{v}".replace(" ", ""),
                                       lowBound=0, upBound=d["cap"])
                 for u, v, d in H.edges(data=True)}
            m += x[(snk, src)]
            for nd in H.nodes():
                m += (pl.lpSum(x[(i, nd)] for i in H.predecessors(nd))
                      == pl.lpSum(x[(nd, j)] for j in H.successors(nd)))
            m.solve()
            obj = pl.value(m.objective) or 0.0
            # Fluss auf der künstlichen Rückkante (Senke→Quelle) = Gesamtdurchfluss.
            G.graph["return_flow"] = x[(snk, src)].varValue or 0.0
            for u, v in G.edges():
                f = x[(u, v)].varValue or 0.0
                G[u][v]["flow"] = f
                cap = G[u][v]["cap"]
                G[u][v]["utilization"] = (f / cap) if cap else 0.0
            return G, obj
        else:
            m = pl.LpProblem("Transport", pl.LpMinimize)
            x = {(u, v): pl.LpVariable(f"x_{u}_{v}".replace(" ", ""), lowBound=0)
                 for u, v in G.edges()}
            m += pl.lpSum(G[u][v]["cost"] * x[(u, v)] for u, v in G.edges())
            for nd in G.nodes():
                inflow = pl.lpSum(x[(i, nd)] * G[i][nd]["yield"]
                                  for i in G.predecessors(nd))
                outflow = pl.lpSum(x[(nd, j)] for j in G.successors(nd))
                m += inflow + G.nodes[nd].get("balance", 0) >= outflow
            m.solve()
            obj = pl.value(m.objective) or 0.0
            for u, v in G.edges():
                G[u][v]["flow"] = x[(u, v)].varValue or 0.0
            return G, obj

    def _node_colors(spec, G):
        if "colors" in spec:
            return [spec["colors"].get(n, farbe_zwischen) for n in G.nodes()]
        if spec["kind"] == "maxflow":
            return [farbe_terminal if n in (spec["src"], spec["snk"])
                    else farbe_zwischen for n in G.nodes()]
        cols = []
        for n in G.nodes():
            b = G.nodes[n].get("balance", 0)
            cols.append(farbe_quelle if b > 0 else farbe_senke if b < 0
                        else farbe_zwischen)
        return cols

    def draw_spec(spec, G, flows=False, title="", figsize=(8.5, 5.5),
                  show_return=False):
        """Netz zeichnen. flows=True → optimale Flüsse (Dicke+Farbe) einblenden.
        show_return=True → künstliche Rückkante Senke→Quelle (nur Max-Flow)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, PathPatch
        from matplotlib.path import Path
        from matplotlib.colors import Normalize
        import matplotlib as mpl

        pos = spec["pos"]
        kind = spec["kind"]
        wkey = "cap" if kind == "maxflow" else "cost"
        fig, ax = plt.subplots(figsize=figsize)

        normal = [(u, v) for u, v in G.edges()]
        ncolors = _node_colors(spec, G)

        # ── Flusswerte / Auslastung ──────────────────────────────────────
        flow_vals = {}
        if flows:
            for u, v in G.edges():
                f = G[u][v].get("flow", 0.0)
                if f and f > 1e-6:
                    flow_vals[(u, v)] = f
        scale = (5.5 / max(flow_vals.values())) if flow_vals else 1.0
        util_cmap = mpl.colormaps.get_cmap("RdYlGn_r")
        unorm = Normalize(vmin=0.0, vmax=1.0)

        # ── Knoten + Basis-Kanten ────────────────────────────────────────
        nx.draw_networkx_nodes(G, pos, node_color=ncolors, node_size=1500,
                               edgecolors="#33425a", linewidths=1.2, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=normal, edge_color="#9aa3ad",
                               width=1.4, arrows=True, arrowsize=15,
                               connectionstyle="arc3,rad=0.08",
                               node_size=1500, ax=ax)
        _num = spec.get("num", {})
        _nl = chr(10)
        _labels = {n: (f"{str(n).replace('_', _nl)}{_nl}({_num[n]})"
                       if n in _num else str(n).replace("_", _nl))
                   for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=_labels, font_size=8.5, ax=ax)

        # ── Fluss-Overlay ────────────────────────────────────────────────
        if flow_vals:
            fe = list(flow_vals.keys())
            widths = [max(1.5, scale * flow_vals[e]) for e in fe]
            ecol = [util_cmap(unorm(G[e[0]][e[1]].get("utilization", 0.6)))
                    if kind == "maxflow" else IMSOrange for e in fe]
            nx.draw_networkx_edges(G, pos, edgelist=fe, width=widths,
                                   edge_color=ecol, arrows=True, arrowsize=16,
                                   connectionstyle="arc3,rad=0.08",
                                   node_size=1500, ax=ax)

        # ── Kantenbeschriftungen ─────────────────────────────────────────
        elabels = {}
        for u, v in G.edges():
            base = G[u][v][wkey]
            txt = (f"{base:g}" if kind == "maxflow" else
                   (f"{base:g}".replace(".", ",")))
            if flows and (u, v) in flow_vals:
                f = flow_vals[(u, v)]
                if kind == "maxflow":
                    txt = f"{f:g}/{base:g}"
                else:
                    txt = f"c={txt} | x={f:g}"
            elabels[(u, v)] = txt
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=elabels, font_size=7.5, label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
            connectionstyle="arc3,rad=0.08", ax=ax)

        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        ax.margins(0.12)
        ax.axis("off")

        # ── Rückkante (Max-Flow): Senke → Quelle, unbeschränkt ────────────
        # Nach margins(), damit die set_ylim-Erweiterung nicht überschrieben wird.
        # Eckige (rechtwinklige) Führung unter dem Netz: runter → quer → rauf.
        if show_return and kind == "maxflow":
            s, t = spec["src"], spec["snk"]
            xs, y_s = pos[s]
            xt, y_t = pos[t]
            y_lo = min(p[1] for p in pos.values())
            y_bot = y_lo - 0.55                  # Höhe des Quersegments
            verts = [(xt, y_t), (xt, y_bot), (xs, y_bot), (xs, y_s)]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            arr = FancyArrowPatch(
                path=Path(verts, codes), arrowstyle="-|>", mutation_scale=16,
                lw=2.0, ls="--", color=IMSBlue, zorder=1,
                shrinkA=2, shrinkB=16, joinstyle="miter", capstyle="butt")
            ax.add_patch(arr)
            rf = G.graph.get("return_flow", 0.0)
            _num = spec.get("num", {})
            if t in _num and s in _num:
                _rk = f"Rückkante  $X_{{{_num[t]}{_num[s]}}}={rf:g}$  (unbeschränkt)"
            else:
                _rk = f"Rückkante  {t} → {s} = {rf:g}  (unbeschränkt)"
            ax.text((xs + xt) / 2, y_bot - 0.12, _rk,
                    ha="center", va="top", fontsize=8, style="italic",
                    color=IMSBlue,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=IMSBlue, ls="--", lw=0.8, alpha=0.92))
            _cur = ax.get_ylim()
            ax.set_ylim(min(_cur[0], y_bot - 0.9), _cur[1])

        fig.tight_layout()
        return fig

    return build_graph, draw_spec, solve_spec


@app.cell(hide_code=True)
def specs(
    farbe_quelle,
    farbe_senke,
    farbe_terminal,
    farbe_zwischen,
):
    # ── Pipeline (Northwest Petroleum) — Max-Flow, exakt wie VL 08 ────────
    SPEC_PIPELINE = {
        "kind": "maxflow", "src": "Ölfeld", "snk": "Raffinerie",
        "num": {"Ölfeld": 1, "Station 1": 2, "Station 2": 3,
                "Station 3": 4, "Station 4": 5, "Raffinerie": 6},
        "pos": {"Ölfeld": (0, 0), "Station 1": (1, 1.1), "Station 2": (1, -1.1),
                "Station 3": (2, 1.1), "Station 4": (2, -1.1), "Raffinerie": (3, 0)},
        "edges": [("Ölfeld", "Station 1", 6), ("Ölfeld", "Station 2", 4),
                  ("Station 1", "Station 3", 3), ("Station 1", "Station 4", 2),
                  ("Station 2", "Station 3", 2), ("Station 2", "Station 4", 5),
                  ("Station 3", "Raffinerie", 6), ("Station 4", "Raffinerie", 4)],
    }

    # ── Café-Evakuierung — Max-Flow (Kapazitäten = Personen/Minute) ───────
    def cafe_spec(t=1.0):
        return {
            "kind": "maxflow", "src": "Café", "snk": "Draußen",
            "pos": {"Café": (0, 1), "Hauptraum": (1.4, 1), "Empore": (1.4, 2.4),
                    "Hinterzimmer": (1.4, -0.4), "Treppe": (2.8, 2.4),
                    "Flur N": (2.8, 1), "Flur S": (2.8, -0.4),
                    "Ausgang N": (4.2, 1.5), "Ausgang S": (4.2, -0.4),
                    "Draußen": (5.6, 0.6)},
            "edges": [("Café", "Hauptraum", 200), ("Café", "Empore", 70),
                      ("Café", "Hinterzimmer", 60),
                      ("Empore", "Treppe", 45 * t), ("Treppe", "Flur N", 45 * t),
                      ("Hauptraum", "Flur N", 90 * t), ("Hauptraum", "Flur S", 50 * t),
                      ("Hinterzimmer", "Flur S", 40 * t),
                      ("Flur N", "Ausgang N", 80 * t), ("Flur S", "Ausgang S", 60 * t),
                      ("Ausgang N", "Draußen", 80 * t), ("Ausgang S", "Draußen", 60 * t)],
        }

    # ── BMC (Bavarian Motor Company) — Transport/Transshipment, wie VL 08 ─
    SPEC_BMC = {
        "kind": "transport",
        "num": {"Newark": 1, "Boston": 2, "Columbus": 3, "Richmond": 4,
                "Atlanta": 5, "Mobile": 6, "Jacksonville": 7},
        "pos": {"Boston": (1, 3), "Newark": (2.6, 3.4), "Richmond": (2, 1.5),
                "Columbus": (0, 2), "Mobile": (0, 0), "Atlanta": (1, 1),
                "Jacksonville": (3, 0)},
        "balance": {"Newark": 200, "Boston": -100, "Richmond": -80,
                    "Columbus": -60, "Atlanta": -170, "Mobile": -70,
                    "Jacksonville": 300},
        "edges": [("Newark", "Boston", 30), ("Newark", "Richmond", 40),
                  ("Boston", "Columbus", 50), ("Columbus", "Atlanta", 35),
                  ("Atlanta", "Columbus", 40), ("Atlanta", "Richmond", 30),
                  ("Atlanta", "Mobile", 35), ("Mobile", "Atlanta", 25),
                  ("Jacksonville", "Richmond", 50), ("Jacksonville", "Atlanta", 45),
                  ("Jacksonville", "Mobile", 50)],
    }

    # ── Bier-Logistik (A3) — 2 Brauereien → 3 Festzelte, balanciert ───────
    SPEC_TRANS3 = {
        "kind": "transport",
        "pos": {"Brauerei 1": (0, 1.2), "Brauerei 2": (0, -1.2),
                "Zelt A": (3, 2), "Zelt B": (3, 0), "Zelt C": (3, -2)},
        "balance": {"Brauerei 1": 100, "Brauerei 2": 80,
                    "Zelt A": -60, "Zelt B": -70, "Zelt C": -50},
        "edges": [("Brauerei 1", "Zelt A", 4), ("Brauerei 1", "Zelt B", 6),
                  ("Brauerei 1", "Zelt C", 8), ("Brauerei 2", "Zelt A", 5),
                  ("Brauerei 2", "Zelt B", 3), ("Brauerei 2", "Zelt C", 7)],
    }

    # ── Coffee Supply Network (selbständig) — mehrstufiger Min-Kosten-Fluss ─
    _scm_green = ["BR_North", "BR_South", "CO_Huila", "ET_Sidamo"]
    _scm_cafe = ["Cafe_WZ_Altstadt", "Cafe_WZ_Campus", "Cafe_NUC_Burg",
                 "Cafe_NUC_Innenstadt"]
    SPEC_COFFEE = {
        "kind": "transport",
        "pos": {
            "BR_North": (0, 3), "BR_South": (0, 1), "CO_Huila": (0, -1),
            "ET_Sidamo": (0, -3), "BR_Mill": (1.5, 2), "CO_Mill": (1.5, -0.5),
            "ET_Mill": (1.5, -2.5), "Santos_Port": (3, 2),
            "Cartagena_Port": (3, -0.5), "Djibouti_Port": (3, -2.5),
            "Hamburg_Port": (4.5, 1), "Trieste_Port": (4.5, -2),
            "Roastery_WZ": (6, 1), "Roastery_NUC": (6, -2),
            "Cafe_WZ_Altstadt": (7.6, 2.5), "Cafe_WZ_Campus": (7.6, 0.5),
            "Cafe_NUC_Burg": (7.6, -1.5), "Cafe_NUC_Innenstadt": (7.6, -3.2)},
        "balance": {"BR_North": 4000, "BR_South": 4000, "CO_Huila": 3000,
                    "ET_Sidamo": 3000, "Cafe_WZ_Altstadt": -2000,
                    "Cafe_WZ_Campus": -1500, "Cafe_NUC_Burg": -2500,
                    "Cafe_NUC_Innenstadt": -3000},
        "edges": [
            ("BR_North", "BR_Mill", 0.10), ("BR_South", "BR_Mill", 0.12),
            ("CO_Huila", "CO_Mill", 0.11), ("ET_Sidamo", "ET_Mill", 0.15),
            ("BR_Mill", "Santos_Port", 0.15), ("CO_Mill", "Cartagena_Port", 0.10),
            ("ET_Mill", "Djibouti_Port", 0.25),
            ("Santos_Port", "Hamburg_Port", 0.50), ("Santos_Port", "Trieste_Port", 0.55),
            ("Cartagena_Port", "Hamburg_Port", 0.55), ("Cartagena_Port", "Trieste_Port", 0.60),
            ("Djibouti_Port", "Hamburg_Port", 0.65), ("Djibouti_Port", "Trieste_Port", 0.50),
            ("Hamburg_Port", "Roastery_WZ", 0.30, 0.9), ("Hamburg_Port", "Roastery_NUC", 0.35, 0.9),
            ("Trieste_Port", "Roastery_WZ", 0.40, 0.9), ("Trieste_Port", "Roastery_NUC", 0.25, 0.9),
            ("Roastery_WZ", "Cafe_WZ_Altstadt", 0.10), ("Roastery_WZ", "Cafe_WZ_Campus", 0.12),
            ("Roastery_WZ", "Cafe_NUC_Burg", 0.35), ("Roastery_WZ", "Cafe_NUC_Innenstadt", 0.38),
            ("Roastery_NUC", "Cafe_WZ_Altstadt", 0.32), ("Roastery_NUC", "Cafe_WZ_Campus", 0.30),
            ("Roastery_NUC", "Cafe_NUC_Burg", 0.14), ("Roastery_NUC", "Cafe_NUC_Innenstadt", 0.12)],
        "colors": {
            **{n: farbe_quelle for n in _scm_green},
            **{n: farbe_zwischen for n in ["BR_Mill", "CO_Mill", "ET_Mill"]},
            **{n: farbe_terminal for n in ["Santos_Port", "Cartagena_Port",
                                           "Djibouti_Port", "Hamburg_Port", "Trieste_Port"]},
            **{n: "#C8A2D8" for n in ["Roastery_WZ", "Roastery_NUC"]},
            **{n: farbe_senke for n in _scm_cafe}},
    }
    return (
        SPEC_BMC, SPEC_COFFEE, SPEC_PIPELINE, SPEC_TRANS3, cafe_spec,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Übung 7 · Netzwerke

    **Planung und Entscheidung — SS 2026 · Begleitnotebook zu VL 08**

    Sehr viele Planungsprobleme sind **Netzwerke**: Knoten (Orte, Lager, Räume)
    verbunden durch Kanten (Wege, Leitungen) mit Attributen wie **Kosten** oder
    **Kapazität**. Das Schöne: Sie folgen alle **demselben LP-Muster** —
    *eine Variable je Kante, eine Bilanz je Knoten.*

    > **Klausurfokus:** Sie sollen ein Netz **als LP formulieren** können
    > (Entscheidungsvariablen, Zielfunktion, Knotenbedingungen). Die Aufgaben
    > hier sind **Textaufgaben** in genau diesem Format. Die interaktiven Plots
    > stecken in den **Lösungen** zum Selbstkontrollieren — und in der **Sandbox**
    > zum freien Experimentieren.

    **Ablauf (90 Min.):**

    1. **Wiederholung** — Begriffe, das Grundmuster, $\sum$/$\forall$ richtig lesen
    2. **Geführte Übung 1** — Transportproblem (BMC): Netz zeichnen + LP formulieren
    3. **Geführte Übung 2** — Maximaler Fluss (Pipeline), Rückkanten-Trick
    4. **Sandbox** — Kosten/Kapazitäten verstellen und die Lösung beobachten
    5. **Spickzettel** — Netzwerk $\leftrightarrow$ LP auf einen Blick
    6. **Selbständige Aufgaben** — Transport (Bier), Fernbus-Umsatz & Bonus: Lieferkette
    7. **Besprechung & Ausblick**
    """)
    return


@app.cell(hide_code=True)
def _(IMSBlue, IMSOrange, farbe_zwischen, mo):
    # ── Wiederholung 1.1 — alles auf EINEM Slide: Begriffe | Graph | Listen
    #   Knoten 1,2,3 · Kanten 2->1 (c=1,u=5), 1->3 (c=3,u=2), 3->1 (c=2,u=4).
    #   Im Graph nur c beschriftet; u (Kapazitaet) steht in der Kantenliste.
    def _draw_repr():
        import io
        import re
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        pos = {"1": (0.0, 2.0), "2": (2.4, 2.0), "3": (0.0, 0.0)}
        # (u, v, rad, c): 1->3 woelbt links, 3->1 woelbt rechts -> getrennt
        for _u, _v, _rad, _c in [("2", "1", 0.0, 1), ("1", "3", 0.28, 3),
                                 ("3", "1", 0.28, 2)]:
            x1, y1 = pos[_u]
            x2, y2 = pos[_v]
            ax.add_patch(FancyArrowPatch(
                (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={_rad}",
                arrowstyle="-|>", mutation_scale=16, lw=2.0, color="#5b6b7d",
                shrinkA=16, shrinkB=16, zorder=3))
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx, dy = x2 - x1, y2 - y1
            L = (dx * dx + dy * dy) ** 0.5
            ox, oy = dy / L, -dx / L          # senkrecht zur Woelbungsseite
            off = abs(_rad) * L * 1.15 + 0.16
            ax.text(mx + ox * off, my + oy * off, f"{_c}", fontsize=13,
                    ha="center", va="center", fontweight="bold", color="#33425a",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                              alpha=0.9), zorder=7)
        for _n, (x, y) in pos.items():
            ax.scatter([x], [y], s=1800, color=farbe_zwischen,
                       edgecolors="#33425a", lw=1.6, zorder=5)
            ax.text(x, y, _n, ha="center", va="center", fontsize=16,
                    fontweight="bold", zorder=6)

        def _co(text, target, xytext, ec):
            ax.annotate(text, xy=target, xytext=xytext, fontsize=10.5,
                        ha="center", va="center", color="#33425a", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ec,
                                  lw=1.2),
                        arrowprops=dict(arrowstyle="->", color=ec, lw=1.3,
                                        connectionstyle="arc3,rad=0.15"))
        _co("Knoten $i\\in V$", pos["2"], (3.0, 3.0), IMSBlue)
        _co("Kante $\\langle i,j\\rangle\\in\\mathcal{L}$\ngerichtet: $i\\to j$",
            (1.2, 2.0), (0.9, 3.15), IMSOrange)
        _co("Kantengewicht:\nKosten $c_{ij}$ / Kapazität $u_{ij}$", (-0.42, 1.0),
            (-1.6, 1.7), IMSOrange)
        _co("Hin- & Rückkante:\nzwei Pfeile", (0.42, 1.0), (1.9, 0.35), IMSBlue)
        ax.set_xlim(-2.5, 4.0)
        ax.set_ylim(-0.7, 3.7)
        ax.axis("off")
        # Responsiv als SVG einbetten -> fuellt die Spaltenbreite (kein Leerraum)
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg = buf.getvalue()
        svg = svg[svg.index("<svg"):]
        svg = re.sub(r'\s(?:width|height)="[^"]*"', "", svg, count=2)
        svg = svg.replace(
            "<svg ", '<svg style="width:100%;height:auto;display:block" ', 1)
        return mo.Html(svg)

    _begriffe = mo.md(r"""
    ## 1.1 · Begriffe

    Ein **Netzwerk** (Graph) besteht aus

    - **Knoten** $i\in V$ — Orte, optional mit Attributen
      (Angebot $+$, Nachfrage $-$).
    - **Kanten** $\langle i,j\rangle\in\mathcal{L}$ — gerichtete Verbindung
      $i\to j$ mit Attributen: **Kosten** $c_{ij}$ (Transport) und/oder
      **Kapazität** $u_{ij}$ — die *maximale* Menge je Kante (Max-Flow).

    Graphen sind **gerichtet** oder **ungerichtet**. Gespeichert als
    **Kantenliste** (Zeile je Kante) oder **Distanzmatrix**
    ($\infty$ = keine Verbindung).

    **Merksatz:** *Pro Kante eine Variable, pro Knoten eine Bilanz.*
    """)

    _distanzmatrix = mo.md(r"""
    **Distanzmatrix** *(Zeile $\to$ Spalte; $\infty$ = keine Kante)*

    | $i\,\backslash\,j$ | 1 | 2 | 3 |
    |:---:|:---:|:---:|:---:|
    | **1** | 0 | $\infty$ | 3 |
    | **2** | 1 | 0 | $\infty$ |
    | **3** | 2 | $\infty$ | 0 |

    Kompakt für **dichte** Graphen — je Attribut aber eine eigene Matrix.
    """)

    _kantenliste = mo.md(r"""
    **Kantenliste** *(eine Zeile je Kante)*

    | von $i$ | nach $j$ | Kosten $c_{ij}$ | Kapazität $u_{ij}$ |
    |:---:|:---:|:---:|:---:|
    | 2 | 1 | 1 | 5 |
    | 1 | 3 | 3 | 2 |
    | 3 | 1 | 2 | 4 |

    $3$ Kanten $\Rightarrow$ $3$ Variablen $X_{ij}$. Jede **Spalte** ist ein
    Kanten-Attribut ($c$ für Transport, $u$ für Max-Flow, …).
    """)

    _vergleich = mo.md(r"""
    **Zwei Modelltypen — dasselbe Muster:**

    | | **Transportproblem** | **Maximaler Fluss** |
    |---|---|---|
    | Frage | günstigste Verteilung? | maximaler Durchsatz? |
    | Variable je Kante | $X_{ij}$ = Menge $i\to j$ | $X_{ij}$ = Fluss $i\to j$ |
    | Ziel | $\min \sum c_{ij}X_{ij}$ | $\max$ Gesamtfluss |
    | je Knoten | Bilanz: Zufluss $+$ Angebot $\ge$ Abfluss $+$ Nachfrage | Erhaltung: Abfluss $\le$ Zufluss |
    | je Kante | $X_{ij}\ge 0$ | $0\le X_{ij}\le u_{ij}$ |
    """)

    mo.vstack([
        mo.md(r"""
    ---
    # Teil 1 · Wiederholung
    """),
        mo.hstack([
            mo.vstack([_begriffe, _vergleich], gap=0.6),
            _draw_repr(),
            mo.vstack([_distanzmatrix, _kantenliste], gap=0.6),
        ], widths=[0.40, 0.34, 0.26], gap=1.2, align="start"),
    ], gap=0.6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1.2 · $\sum$ und $\forall$ richtig lesen

    Netzwerk-LPs haben **viele** Variablen und Bedingungen. Statt alles
    auszuschreiben, kürzen wir mit **Summe** $\sum$ und **„für alle"** $\forall$ ab.
    Genau hier verrutscht in Klausuren oft etwas — daher Schritt für Schritt:

    - $\displaystyle\sum_{j} X_{ij}$ heißt **„addiere $X_{ij}$ über alle $j$"**.
      Der Index **unter** dem $\sum$ ($j$) **läuft**; was **nicht** drunter steht
      ($i$) bleibt **fest**.
    - $\forall\, i$ heißt **„für jedes $i$ gilt diese Zeile separat"** — eine
      Bedingung **je** $i$.

    Verstellen Sie unten $m$ (Quellen) und $n$ (Senken) und vergleichen Sie:
    **links** wächst die kompakte Form **nicht**, **rechts** explodiert die
    ausgeschriebene Form (bei größeren $m,n$ mit $\dots$ / $\vdots$ abgekürzt).
    *Das* ist der Grund für $\sum$ und $\forall$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sf_m = mo.ui.slider(start=1, stop=5, step=1, value=2,
                        label="$m$ — Anzahl Quellen", show_value=True)
    sf_n = mo.ui.slider(start=1, stop=5, step=1, value=3,
                        label="$n$ — Anzahl Senken", show_value=True)
    return sf_m, sf_n


@app.cell(hide_code=True)
def _(IMSBlue, IMSOrange, farbe_quelle, farbe_senke, mo, sf_m, sf_n):
    m = int(sf_m.value)
    n = int(sf_n.value)
    ndv = m * n
    ncon = m + n
    # Voll ausgeschrieben nur solange kompakt genug; sonst mit \dots / \vdots kürzen.
    expand = (ndv <= 6) and (ncon <= 5)

    # ── konkreter Graph: m Werke → n Märkte, Summen farblich sichtbar ─────
    def _draw_bipartite():
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.5, 3.9))
        ax.axis("off")

        def ys(k):
            return [0.0] if k == 1 else [(k - 1) / 2.0 - t for t in range(k)]
        Ly, Ry = ys(m), ys(n)
        Lpos = {i: (0.0, Ly[i - 1]) for i in range(1, m + 1)}
        Rpos = {j: (2.2, Ry[j - 1]) for j in range(1, n + 1)}
        GRAY, PURPLE = "#c4cad2", "#8E44AD"

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if i == 1 and j == 1:
                    col, lw, z = PURPLE, 2.8, 4
                elif i == 1:
                    col, lw, z = IMSOrange, 2.4, 3
                elif j == 1:
                    col, lw, z = IMSBlue, 2.4, 3
                else:
                    col, lw, z = GRAY, 1.0, 2
                ax.annotate("", xy=Rpos[j], xytext=Lpos[i],
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                            shrinkA=11, shrinkB=11), zorder=z)
        for i in range(1, m + 1):
            x, y = Lpos[i]
            ax.scatter([x], [y], s=640, color=farbe_quelle, edgecolors="#33425a",
                       lw=1.2, zorder=5)
            ax.text(x, y, f"W{i}", ha="center", va="center", fontsize=9,
                    fontweight="bold", zorder=6)
        for j in range(1, n + 1):
            x, y = Rpos[j]
            ax.scatter([x], [y], s=640, color=farbe_senke, edgecolors="#33425a",
                       lw=1.2, zorder=5)
            ax.text(x, y, f"M{j}", ha="center", va="center", fontsize=9,
                    fontweight="bold", zorder=6)
        _hi = max(max(Ly), max(Ry))
        _lo = min(min(Ly), min(Ry))
        ax.text(0.0, _hi + 0.7, "Werke\n(Angebot $a_i$)", ha="center", va="bottom",
                fontsize=8.5, color="#2C8F4E", fontweight="bold")
        ax.text(2.2, _hi + 0.7, "Märkte\n(Nachfrage $b_j$)", ha="center", va="bottom",
                fontsize=8.5, color="#C0392B", fontweight="bold")
        ax.set_xlim(-0.8, 3.0)
        ax.set_ylim(_lo - 0.7, _hi + 1.7)
        fig.tight_layout()
        return fig

    _problem = mo.md(rf"""
    ### Konkret: Werk-Markt-Transport
    $m={m}$ **Werke** beliefern $n={n}$ **Märkte**. $X_{{ij}}=$ Liefermenge von Werk $i$
    zu Markt $j$ (Kosten $c_{{ij}}$). **Jede Linie im Graph ist eine Variable $X_{{ij}}$**
    — hier {ndv} Stück.

    - 🟧 $\sum_{{j=1}}^{{{n}}} X_{{1j}}$: alle Kanten, die **W1 verlassen** — Abfluss (Angebotszeile $i{{=}}1$).
    - 🟦 $\sum_{{i=1}}^{{{m}}} X_{{i1}}$: alle Kanten, die in **M1 münden** — Zufluss (Nachfragezeile $j{{=}}1$).
    - 🟪 $X_{{11}}$ steht in **beiden** Summen.
    """)

    def _terms(items, head=2):
        if expand or len(items) <= head + 1:
            return " + ".join(items)
        return f"{items[0]} + {items[1]} + \\dots + {items[-1]}"

    def _block(rows_items, rel, rhs_list, maxrows=3):
        bodies = [f"{_terms(it)} {rel} {rhs}"
                  for it, rhs in zip(rows_items, rhs_list)]
        if expand or len(bodies) <= maxrows + 1:
            return bodies
        return [bodies[0], bodies[1], r"\vdots", bodies[-1]]

    def _aligned(lines):
        # Einheitlich GENAU EIN & je Zeile → saubere Ausrichtung (Label | Ausdruck).
        body = r" \\[0.3em] ".join(f"{lab} & {expr}" for lab, expr in lines)
        return r"\begin{aligned}" + body + r"\end{aligned}"

    # ── Kompakt (mit den echten Werten von m, n) ─────────────────────────
    _compact_lp = _aligned([
        (r"\min", rf"\sum_{{i=1}}^{{{m}}}\sum_{{j=1}}^{{{n}}} c_{{ij}}\,X_{{ij}}"),
        (r"\text{u.d.N.}",
         rf"\sum_{{j=1}}^{{{n}}} X_{{ij}} \ge a_i \quad \forall\, i \in \{{1,\dots,{m}\}}"),
        ("", rf"\sum_{{i=1}}^{{{m}}} X_{{ij}} = b_j \quad \forall\, j \in \{{1,\dots,{n}\}}"),
        ("", r"X_{ij} \ge 0 \quad \forall\, i, j"),
    ])

    # ── Ausgeschrieben (gleiche &-Struktur → nichts verrutscht) ──────────
    _obj = _terms([f"c_{{{i}{j}}}X_{{{i}{j}}}"
                   for i in range(1, m + 1) for j in range(1, n + 1)])
    _sup = _block([[f"X_{{{i}{j}}}" for j in range(1, n + 1)]
                   for i in range(1, m + 1)],
                  r"\ge", [f"a_{{{i}}}" for i in range(1, m + 1)])
    _dem = _block([[f"X_{{{i}{j}}}" for i in range(1, m + 1)]
                   for j in range(1, n + 1)],
                  "=", [f"b_{{{j}}}" for j in range(1, n + 1)])
    _lines = [(r"\min", _obj), (r"\text{u.d.N.}", _sup[0])]
    _lines += [("", b) for b in _sup[1:]]
    _lines += [("", b) for b in _dem]
    _lines += [("", r"X_{ij} \ge 0")]
    _explicit_lp = _aligned(_lines)

    _compact = mo.md(rf"""
    ### Kompakt — mit $\sum$ und $\forall$
    *(wächst **nicht** — hier mit $m={m},\ n={n}$ eingesetzt)*

    $$ {_compact_lp} $$

    $i$ **fest**, $j$ läuft → Summe; $\forall\, i$ erzeugt **{m}** Angebotszeilen.
    """)

    _explicit = mo.md(rf"""
    ### Ausgeschrieben — ohne Abkürzung
    *({ndv} Variablen, {ncon} Bedingungen{"" if expand else r" — gekürzt mit $\dots$ / $\vdots$"})*

    $$ {_explicit_lp} $$

    Jede $\forall$-Zeile wird **eine eigene** Bedingung. Bei $m=5,\ n=5$ wären das
    **25** Variablen und **10** Bedingungen — **deshalb** $\sum$ und $\forall$.
    """)

    mo.vstack([
        mo.hstack([sf_m, sf_n], justify="start", gap=1.5),
        mo.md(rf"**Aktuell:** $m={m}$ Quellen, $n={n}$ Senken $\Rightarrow$ "
              rf"$m\cdot n = {ndv}$ Variablen, $m+n = {ncon}$ Knotenbedingungen."),
        mo.hstack([
            mo.vstack([_problem, mo.as_html(_draw_bipartite())]),
            _compact,
            _explicit,
        ], widths=[0.36, 0.30, 0.34], gap=1.2, align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    _knoten = mo.md(r"""
    **Knotenliste** *(Bilanz $b_i$: Werk $+$, Händler $-$)*

    | Knoten | $b_i$ |
    |:--|--:|
    | Newark *(Werk)* | $+200$ |
    | Jacksonville *(Werk)* | $+300$ |
    | Boston | $-100$ |
    | Columbus | $-60$ |
    | Richmond | $-80$ |
    | Atlanta | $-170$ |
    | Mobile | $-70$ |
    """)
    _kanten = mo.md(r"""
    **Kantenliste** *(Kosten $c_{ij}$ in \$ je Fahrzeug)*

    | von | nach | $c_{ij}$ |
    |:--|:--|--:|
    | Newark | Boston | 30 |
    | Newark | Richmond | 40 |
    | Boston | Columbus | 50 |
    | Columbus | Atlanta | 35 |
    | Atlanta | Columbus | 40 |
    | Atlanta | Richmond | 30 |
    | Atlanta | Mobile | 35 |
    | Mobile | Atlanta | 25 |
    | Jacksonville | Richmond | 50 |
    | Jacksonville | Atlanta | 45 |
    | Jacksonville | Mobile | 50 |
    """)
    mo.vstack([
        mo.md(r"""
        ---
        # Teil 2 · Geführte Übung 1 — Transportproblem (BMC)

        Die **Bavarian Motor Company** liefert Fahrzeuge von zwei Werken (**Newark**,
        **Jacksonville**) zu fünf Händlern. Zwischenknoten dürfen **durchgeleitet**
        werden (Transshipment). Gegeben ist das Netz **nur als Listen** — Sie
        zeichnen es selbst und formulieren das LP.
        """),
        mo.hstack([_knoten, _kanten], widths=[0.42, 0.58], gap=1.2, align="start"),
        mo.md(r"""
        ### a) Netz zeichnen
        Skizzieren Sie das Transportnetz: **Werke** als Quellen ($+$, grün),
        **Händler** als Senken ($-$, rot), je Kantenzeile ein Pfeil mit Kosten.
        *(In der Lösung steht das fertige Netz zum Abgleich.)*

        ### b) Als LP formulieren  ·  *Kernaufgabe*

        Formulieren Sie das vollständige LP:

        - **Entscheidungsvariablen** $X_{ij}$
        - **Zielfunktion**
        - **Nebenbedingungen** (Knotenbilanzen + Nichtnegativität)

        > Erst selbst zeichnen & formulieren, dann die Lösung aufklappen.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(SPEC_BMC, build_graph, draw_spec, mo, solve_spec):
    def _bmc_loesung():
        fig_a = draw_spec(SPEC_BMC, build_graph(SPEC_BMC), flows=False,
                          title="a) Netz — Werke (+, grün), Händler (−, rot)",
                          figsize=(8.5, 5.2))
        G, z = solve_spec(SPEC_BMC)
        fig_b = draw_spec(SPEC_BMC, G, flows=True,
                          title=f"b) optimale Belieferung — Gesamtkosten = {z:.0f} \\$",
                          figsize=(8.5, 5.2))
        return fig_a, fig_b, z

    _fig_a, _fig_b, _z = _bmc_loesung()
    mo.accordion({
        "💡 Lösung zu Aufgabe 1 (a + b)": mo.vstack([
            mo.md(r"**a) Netz:** jede Kantenzeile wird ein Pfeil von $i$ nach $j$."),
            mo.as_html(_fig_a),
            mo.md(rf"""
        **Entscheidungsvariablen:** eine je Kante $\Rightarrow$ **11** Variablen
        $X_{{ij}} \ge 0$ (Fahrzeuge $i\to j$; z. B. $X_{{56}}=$ Atlanta$\to$Mobile).

        **Zielfunktion** (Gesamtkosten):
        $$\min\; 30X_{{12}}+40X_{{14}}+50X_{{23}}+35X_{{35}}+40X_{{53}}+30X_{{54}}
        +35X_{{56}}+25X_{{65}}+50X_{{74}}+45X_{{75}}+50X_{{76}}$$

        **Nebenbedingungen** (Abfluss $+$ Nachfrage $\le$ Zufluss $+$ Angebot):
        $$\begin{{aligned}}
        X_{{12}}+X_{{14}} &\le 200 & (\text{{Knoten 1, Newark}})\\
        X_{{23}}+100 &\le X_{{12}} & (\text{{Knoten 2, Boston}})\\
        X_{{35}}+60 &\le X_{{23}}+X_{{53}} & (\text{{Knoten 3, Columbus}})\\
        80 &\le X_{{14}}+X_{{54}}+X_{{74}} & (\text{{Knoten 4, Richmond}})\\
        X_{{53}}+X_{{54}}+X_{{56}}+170 &\le X_{{35}}+X_{{65}}+X_{{75}} & (\text{{Knoten 5, Atlanta}})\\
        X_{{65}}+70 &\le X_{{56}}+X_{{76}} & (\text{{Knoten 6, Mobile}})\\
        X_{{74}}+X_{{75}}+X_{{76}} &\le 300 & (\text{{Knoten 7, Jacksonville}})\\
        X_{{ij}} &\ge 0 & (\text{{Nichtnegativität}})
        \end{{aligned}}$$

        Ungleichungen, weil das Gesamtangebot ($500$) die Nachfrage ($480$) übersteigt.

        **Optimale Lösung:** Gesamtkosten **$z^* = {_z:.0f}$ \$** mit
        $X_{{12}}=120,\ X_{{14}}=80,\ X_{{23}}=20,\ X_{{53}}=40,\ X_{{75}}=210,\ X_{{76}}=70$
        (Newark $\to$ Boston \& Richmond, Jacksonville $\to$ Atlanta \& Mobile,
        Columbus über Boston/Atlanta).
        """),
            mo.as_html(_fig_b),
        ])
    })
    return


@app.cell(hide_code=True)
def _(SPEC_PIPELINE, build_graph, draw_spec, mo):
    _fig = draw_spec(SPEC_PIPELINE, build_graph(SPEC_PIPELINE), flows=False,
                     title="Pipeline — Kantenzahlen = Durchflusskapazitäten",
                     figsize=(8, 4.6))
    mo.vstack([
        mo.md(r"""
        ---
        # Teil 3 · Geführte Übung 2 — Maximaler Fluss (Pipeline)

        Vom **Ölfeld** soll möglichst viel Öl zur **Raffinerie** fließen. Jede Kante
        hat eine **Kapazität**; an jeder Station gilt **Erhaltung** (Abfluss $\le$ Zufluss).

        **Rückkanten-Trick (VL 08):** Wir ergänzen eine künstliche **Rückkante**
        von der Senke zur Quelle, $X_{61}$, *ohne* Kapazitätsgrenze.
        Der Fluss auf ihr ist der **Gesamtdurchfluss** — und genau den maximieren wir.
        """),
        mo.as_html(_fig),
        mo.md(r"""
        ### Aufgabe 2 — Max-Flow als LP

        **2a)** Formulieren Sie das LP mit dem Rückkanten-Trick:

        - **Entscheidungsvariablen** $X_{ij}$
        - **Zielfunktion**
        - **Nebenbedingungen** (Flusserhaltung je Station + Kapazitätsschranken
          $0 \le X_{ij} \le u_{ij}$)

        **Verständnisfragen:**

        **2b)** Warum maximieren wir den Fluss auf der **Rückkante**
        $X_{61}$ — was leistet sie im Modell?

        **2c)** Ein **Engpass** ist eine Menge gesättigter Kanten. Warum bringt
        mehr Kapazität auf einer **nicht** gesättigten Kante *keinen* höheren
        Gesamtfluss?
        """),
    ])
    return


@app.cell(hide_code=True)
def _(SPEC_PIPELINE, draw_spec, mo, solve_spec):
    def _pipe_loesung():
        G, z = solve_spec(SPEC_PIPELINE)
        fig = draw_spec(SPEC_PIPELINE, G, flows=True,
                        title=f"Pipeline — maximaler Fluss = {z:.0f} (Beschriftung: Fluss/Kapazität)",
                        figsize=(8, 4.6), show_return=True)
        _num = SPEC_PIPELINE["num"]
        sat = [f"X_{{{_num[u]}{_num[v]}}}" for u, v in G.edges()
               if G[u][v]["cap"] and abs(G[u][v]["flow"] - G[u][v]["cap"]) < 1e-6]
        return fig, z, sat

    _fig, _z, _sat = _pipe_loesung()
    mo.accordion({
        "💡 Lösung zu Aufgabe 2 (zum Aufklappen)": mo.vstack([
            mo.md(rf"""
        **2a)** LP mit Rückkante $X_{{61}}$:

        $$\max \; X_{{61}} \quad (\text{{Gesamtdurchfluss}})$$

        **Flusserhaltung** (Abfluss $\le$ Zufluss) — alle Knoten:
        $$\begin{{aligned}}
        \text{{Knoten 1 (Ölfeld):}}\quad & X_{{12}}+X_{{13}} \le X_{{61}}\\
        \text{{Knoten 2 (St. 1):}}\quad & X_{{24}}+X_{{25}} \le X_{{12}}\\
        \text{{Knoten 3 (St. 2):}}\quad & X_{{34}}+X_{{35}} \le X_{{13}}\\
        \text{{Knoten 4 (St. 3):}}\quad & X_{{46}} \le X_{{24}}+X_{{34}}\\
        \text{{Knoten 5 (St. 4):}}\quad & X_{{56}} \le X_{{25}}+X_{{35}}\\
        \text{{Knoten 6 (Raff.):}}\quad & X_{{61}} \le X_{{46}}+X_{{56}}
        \end{{aligned}}$$

        **Kapazitäten** $0 \le X_{{ij}} \le u_{{ij}}$ (Rückkante $X_{{61}}$ unbeschränkt):
        $$\begin{{aligned}}
        0 \le X_{{12}} \le 6,\quad & 0 \le X_{{13}} \le 4\\
        0 \le X_{{24}} \le 3,\quad & 0 \le X_{{25}} \le 2\\
        0 \le X_{{34}} \le 2,\quad & 0 \le X_{{35}} \le 5\\
        0 \le X_{{46}} \le 6,\quad & 0 \le X_{{56}} \le 4
        \end{{aligned}}$$

        **2b)** Die Rückkante schließt das Netz zum **Kreis**: weil an *jedem* Knoten
        Abfluss $\le$ Zufluss gilt (nichts versickert), läuft im Optimum der gesamte
        von der Quelle erzeugte Fluss über
        sie zurück. Ihr Wert ist damit genau der **Gesamtdurchfluss** — deshalb
        maximieren wir ihn. *(Maximaler Fluss $= {_z:.0f}$; gesättigt: ${", ".join(_sat)}$.)*

        **2c)** Eine **nicht** gesättigte Kante hat noch **freie** Kapazität — sie
        begrenzt den Fluss nicht. Der Gesamtfluss wird allein vom **Engpass** (dem
        Schnitt aus gesättigten Kanten) bestimmt; nur dort hilft mehr Kapazität.
        *(In der Sandbox unten ausprobieren!)*
        """),
            mo.as_html(_fig),
        ])
    })
    return


@app.cell(hide_code=True)
def _(SPEC_BMC, SPEC_PIPELINE, SPEC_TRANS3, cafe_spec, mo):
    sb_choice = mo.ui.dropdown(
        options={
            "Pipeline (Max-Flow)": "pipeline",
            "Café-Evakuierung (Max-Flow)": "cafe",
            "BMC-Transport": "bmc",
            "Bier → Festzelte (Transport)": "trans3",
        },
        value="Pipeline (Max-Flow)",
        label="**Netz wählen:**",
    )
    SB_SPECS = {
        "pipeline": SPEC_PIPELINE,
        "cafe": cafe_spec(2.0),
        "bmc": SPEC_BMC,
        "trans3": SPEC_TRANS3,
    }
    return SB_SPECS, sb_choice


@app.cell(hide_code=True)
def _(SB_SPECS, mo, sb_choice):
    _spec = SB_SPECS[sb_choice.value]
    _wkey = "cap" if _spec["kind"] == "maxflow" else "cost"
    _sliders = {}
    for _e in _spec["edges"]:
        _u, _v, _w = _e[0], _e[1], _e[2]
        if _spec["kind"] == "maxflow":
            _sliders[f"{_u}→{_v}"] = mo.ui.slider(
                start=0, stop=int(max(12, _w * 2)), step=1, value=int(_w),
                label=f"{_u}→{_v}", show_value=True)
        else:
            _hi = round(_w * 2 + (5 if _w >= 1 else 1), 2)
            _step = 1 if _w >= 1 else 0.05
            _sliders[f"{_u}→{_v}"] = mo.ui.slider(
                start=0, stop=_hi, step=_step, value=_w,
                label=f"{_u}→{_v}", show_value=True)
    sb_edges = mo.ui.dictionary(_sliders)
    return (sb_edges,)


@app.cell(hide_code=True)
def _(SB_SPECS, draw_spec, mo, sb_choice, sb_edges, solve_spec):
    def _sandbox():
        spec = SB_SPECS[sb_choice.value]
        overrides = {}
        for e in spec["edges"]:
            u, v = e[0], e[1]
            key = f"{u}→{v}"
            if key in sb_edges.value:
                overrides[(u, v)] = sb_edges.value[key]
        G, z = solve_spec(spec, overrides)
        einheit = "maximaler Fluss" if spec["kind"] == "maxflow" else "Gesamtkosten"
        fig = draw_spec(spec, G, flows=True,
                        title=f"Lösung — {einheit} = {z:g}",
                        figsize=(8.4, 5.2),
                        show_return=(spec["kind"] == "maxflow"))
        return spec, fig, z, einheit

    def _model_md(spec):
        nE = len(spec["edges"])
        nN = len(spec["pos"])
        if spec["kind"] == "maxflow":
            s, t = spec["src"], spec["snk"]
            return mo.md(rf"""
            ### Modell (Max-Flow)

            $$\max\ X_{{\text{{{t}}}\to\text{{{s}}}}}
            \quad (\text{{Gesamtfluss}})$$

            **u. d. N.**

            $$\begin{{aligned}}
            \sum_{{j=1}}^{{{nN}}} X_{{kj}} \le \sum_{{i=1}}^{{{nN}}} X_{{ik}}\ \ \forall k &\quad (\text{{Flusserhaltung}})\\
            0 \le X_{{ij}} \le u_{{ij}} &\quad (\text{{Kapazität je Kante}})
            \end{{aligned}}$$

            *{nE} Variablen · {nN} Knotengleichungen*
            """)
        return mo.md(rf"""
        ### Modell (Transport)

        $$\min\ \sum_{{i=1}}^{{{nN}}}\sum_{{j=1}}^{{{nN}}} c_{{ij}}\,X_{{ij}}
        \quad (\text{{Gesamtkosten}})$$

        **u. d. N.**

        $$\begin{{aligned}}
        \sum_{{i=1}}^{{{nN}}} X_{{ik}} + b_k \ge \sum_{{j=1}}^{{{nN}}} X_{{kj}}\ \ \forall k &\quad (\text{{Knotenbilanz}})\\
        X_{{ij}} \ge 0 &\quad (\text{{Nichtnegativität}})
        \end{{aligned}}$$

        *{nE} Variablen · {nN} Knotenbedingungen*
        """)

    _spec, _fig, _z, _einheit = _sandbox()
    mo.vstack([
        mo.md(r"""
        ---
        # Teil 4 · Sandbox — selbst experimentieren

        Netz wählen, **Kapazitäten** (Max-Flow) bzw. **Kosten** (Transport) je Kante
        verstellen — **Modell** und **Lösung** aktualisieren sich live.
        *Tipp:* Eine **nicht** gesättigte Kante zu erhöhen ändert den Max-Flow nicht;
        es zählt der **Engpass**. Bei Max-Flow zeigt der **gestrichelte blaue Bogen**
        unten die künstliche **Rückkante** (Senke → Quelle) — ihr Fluss ist der
        Gesamtdurchsatz, den wir maximieren.
        """),
        mo.hstack([sb_choice, mo.md(f"**{_einheit.capitalize()}:** {_z:g}")],
                  justify="start", gap=2.0, align="center"),
        mo.hstack([
            mo.vstack([mo.md("**Kanten verstellen:**"),
                       mo.vstack(list(sb_edges.elements.values()), gap=0.3,
                                 align="end")], align="end"),
            mo.as_html(_fig),
            _model_md(_spec),
        ], widths=[0.24, 0.48, 0.28], gap=1.0, align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Teil 5 · Spickzettel — Netzwerk $\leftrightarrow$ LP

    Alles, was Sie für die selbständigen Aufgaben (und die Klausur) brauchen:

    | Schritt | So gehen Sie vor |
    |---|---|
    | **1. Variablen** | Eine je Kante: $X_{ij}$ = Menge/Fluss von $i$ nach $j$, $X_{ij}\ge 0$ |
    | **2. Zielfunktion** | Transport: $\min \sum c_{ij}X_{ij}$ · Max-Flow: $\max X_{\text{Senke}\to\text{Quelle}}$ (Rückkante) |
    | **3. Knotenbedingung** | Transport: Zufluss $+$ Angebot $\ge$ Abfluss $+$ Nachfrage · Max-Flow: Abfluss $\le$ Zufluss |
    | **4. Kantengrenzen** | Max-Flow: $0\le X_{ij}\le u_{ij}$ |

    **Klausur-Tipps**

    - **Zähle zuerst die Kanten** → so viele Variablen $X_{ij}$ gibt es.
    - **$\sum_j X_{ij}$**: erster Index $i$ fest (Knoten), zweiter läuft → *Abfluss aus $i$*.
    - **$\forall i$**: eine Bedingung **je** Knoten — alle hinschreiben (bzw. mit $\forall$ kürzen).
    - **Max-Flow**: Rückkante nicht vergessen; sie trägt die Zielgröße.
    - **Engpass** = Menge gesättigter Kanten; nur dort hilft mehr Kapazität.
    """)
    return


@app.cell(hide_code=True)
def _(SPEC_TRANS3, build_graph, draw_spec, mo):
    _fig = draw_spec(SPEC_TRANS3, build_graph(SPEC_TRANS3), flows=False,
                     title="Bier-Logistik — Angebot (+), Nachfrage (−), Kanten = Stückkosten",
                     figsize=(7, 4.4))
    mo.vstack([
        mo.md(r"""
        ---
        # Teil 6 · Selbständige Aufgaben

        ## Aufgabe 3 — Bier-Logistik vollständig formulieren

        Zum **Würzburger Stadtfest** beliefern zwei Brauereien drei Festzelte mit
        Bier (in Hektolitern). **Vorrat:** Brauerei 1 $=100$, Brauerei 2 $=80$.
        **Bedarf:** Zelt A $=60$, Zelt B $=70$, Zelt C $=50$ (Summe $=180$,
        Vorrat $=180$ → ausgeglichen). Stückkosten je hl siehe Kanten.
        """),
        mo.as_html(_fig),
        mo.md(r"""
        **a)** Definieren Sie die **Entscheidungsvariablen** ($X_{ij}\ge 0$).
        **b)** Stellen Sie die **Zielfunktion** vollständig auf.
        **c)** Schreiben Sie **alle 5 Knotenbedingungen** (2 Angebot, 3 Nachfrage).
        """),
    ])
    return


@app.cell(hide_code=True)
def _(SPEC_TRANS3, draw_spec, mo, solve_spec):
    def _t3():
        G, z = solve_spec(SPEC_TRANS3)
        fig = draw_spec(SPEC_TRANS3, G, flows=True,
                        title=f"Aufgabe 3 — optimale Belieferung, Gesamtkosten = {z:.0f}",
                        figsize=(7, 4.4))
        return fig, z

    _fig, _z = _t3()
    mo.accordion({
        "💡 Lösung zu Aufgabe 3": mo.vstack([
            mo.md(r"""
        **a)** $X_{ij}$ = gelieferte Menge (hl) von Brauerei $i$ zu Zelt $j$,
        $i\in\{1,2\}$, $j\in\{A,B,C\}$ — also **6 Variablen**.

        **b)** $\min\; 4X_{1A} + 6X_{1B} + 8X_{1C} + 5X_{2A} + 3X_{2B} + 7X_{2C}
        \quad (\text{Transportkosten})$

        **c)** Je Knoten eine Bedingung:

        $$\begin{aligned}
        X_{1A}+X_{1B}+X_{1C} &\le 100 & (\text{Vorrat Brauerei 1})\\
        X_{2A}+X_{2B}+X_{2C} &\le 80 & (\text{Vorrat Brauerei 2})\\
        X_{1A}+X_{2A} &\ge 60 & (\text{Bedarf Zelt A})\\
        X_{1B}+X_{2B} &\ge 70 & (\text{Bedarf Zelt B})\\
        X_{1C}+X_{2C} &\ge 50 & (\text{Bedarf Zelt C})
        \end{aligned}$$
        """),
            mo.as_html(_fig),
        ])
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Aufgabe 4 — Fernbus-Umsatzmaximierung *(Transfer)*

    Ein Fernbus mit **$p=100$ Sitzplätzen** fährt die Städte **1 → 2 → 3 → 4 → 5**
    in fester Reihenfolge ab. An jeder Stadt können Fahrgäste **zusteigen** und an
    einer späteren Stadt wieder **aussteigen**. $b_{ij}$ ist die **Nachfrage**
    (Personen, die von $i$ nach $j$ wollen), $f_{ij}$ der **Ticketpreis**. Gesucht
    sind die Mitnahmemengen $x_{ij}$, die den **Umsatz maximieren**.

    | Verbindung $i\to j$ | $b_{ij}$ (Nachfrage) | $f_{ij}$ (Preis \$) |
    |:--:|:--:|:--:|
    | 1→2 | 40 | 15 |
    | 1→3 | 30 | 25 |
    | 1→4 | 20 | 32 |
    | 1→5 | 25 | 38 |
    | 2→3 | 35 | 14 |
    | 2→4 | 15 | 24 |
    | 2→5 | 20 | 30 |
    | 3→4 | 30 | 13 |
    | 3→5 | 25 | 22 |
    | 4→5 | 40 | 16 |

    **b1)** EV: $x_{ij}=$ mitgenommene Personen $i\to j$ ($i<j$). Wie **viele**
    Variablen sind das?

    **b2)** **Zielfunktion** (Umsatz) in Summenschreibweise.

    **b3)** **Nachfrage-Nebenbedingung**: es kann niemand über die Nachfrage hinaus
    mitgenommen werden.

    **b4)** **Kapazität pro Abschnitt:** Auf dem Abschnitt **2→3** sitzen *alle*
    Fahrgäste im Bus, die **bei $\le 2$ einsteigen und erst bei $>2$ aussteigen**.
    Schreiben Sie diese Kapazitätsbedingung auf.

    **b5)** *(kurz)* Warum bindet die Kapazität **pro Abschnitt** und **nicht pro
    Kante**? *(Denken Sie an einen Fahrgast 1→5.)*

    > Eine Kapazitäts-NB **je Abschnitt** $s\in\{1,2,3,4\}$ — nicht je Kante.
    """)
    return


@app.cell(hide_code=True)
def _(IMSBlue, IMSOrange, farbe_zwischen, mo, pl):
    _stops = [1, 2, 3, 4, 5]
    _conns = [(i, j) for i in _stops for j in _stops if i < j]
    _b = {(1, 2): 40, (1, 3): 30, (1, 4): 20, (1, 5): 25, (2, 3): 35,
          (2, 4): 15, (2, 5): 20, (3, 4): 30, (3, 5): 25, (4, 5): 40}
    _f = {(1, 2): 15, (1, 3): 25, (1, 4): 32, (1, 5): 38, (2, 3): 14,
          (2, 4): 24, (2, 5): 30, (3, 4): 13, (3, 5): 22, (4, 5): 16}
    _p = 100

    def _fernbus_solve():
        m = pl.LpProblem("Fernbus", pl.LpMaximize)
        x = {(i, j): pl.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=_b[(i, j)])
             for (i, j) in _conns}
        m += pl.lpSum(_f[(i, j)] * x[(i, j)] for (i, j) in _conns)
        for s in range(1, 5):
            m += (pl.lpSum(x[(i, j)] for (i, j) in _conns if i <= s < j) <= _p,
                  f"cap_{s}")
        m.solve()
        z = pl.value(m.objective) or 0.0
        loads = {s: sum((x[(i, j)].varValue or 0.0)
                        for (i, j) in _conns if i <= s < j)
                 for s in range(1, 5)}
        return z, loads

    def _fernbus_sketch():
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        fig, ax = plt.subplots(figsize=(8.6, 3.0))
        xs = [0, 1.7, 3.4, 5.1, 6.8]
        for s in range(4):
            ax.add_patch(FancyArrowPatch(
                (xs[s], 0), (xs[s + 1], 0), arrowstyle="-|>", mutation_scale=16,
                lw=2.4, color=IMSBlue, shrinkA=16, shrinkB=16, zorder=3))
            ax.text((xs[s] + xs[s + 1]) / 2, 0.22, f"Abschnitt {s + 1}",
                    ha="center", fontsize=8, color=IMSBlue)
        ax.add_patch(FancyArrowPatch(
            (xs[0], 0), (xs[4], 0), connectionstyle="arc3,rad=-0.38",
            arrowstyle="-|>", mutation_scale=13, lw=1.6, ls="--",
            color=IMSOrange, shrinkA=16, shrinkB=16, zorder=2))
        ax.text((xs[0] + xs[4]) / 2, 1.25,
                "Fahrgast 1→5 belegt ALLE vier Abschnitte", ha="center",
                fontsize=8.5, color=IMSOrange, style="italic")
        for k, xx in enumerate(xs, start=1):
            ax.scatter([xx], [0], s=1500, color=farbe_zwischen,
                       edgecolors="#33425a", lw=1.4, zorder=5)
            ax.text(xx, 0, str(k), ha="center", va="center", fontsize=13,
                    fontweight="bold", zorder=6)
        ax.set_xlim(-0.8, 7.6)
        ax.set_ylim(-0.7, 1.7)
        ax.axis("off")
        fig.tight_layout()
        return fig

    _z, _loads = _fernbus_solve()
    _load_md = " · ".join(f"Abschn. {s}: {_loads[s]:g}/{_p}" for s in range(1, 5))
    mo.accordion({
        "💡 Lösung zu Aufgabe 4": mo.vstack([
            mo.as_html(_fernbus_sketch()),
            mo.md(rf"""
        **b1)** Eine Variable je Verbindung $i<j$ über 5 Stops $\Rightarrow$
        $\binom{{5}}{{2}}=$ **10 Variablen**.

        **b2)** $\displaystyle \max \sum_{{i=1}}^{{4}}\sum_{{j=i+1}}^{{5}} f_{{ij}}\,x_{{ij}}$

        **b3)** Nachfrage: $\;x_{{ij}} \le b_{{ij}} \quad \forall\, i<j$

        **b4)** Je Abschnitt $s$ alle Fahrgäste mit $i\le s < j$:
        $$\sum_{{i=1}}^{{s}}\sum_{{j=s+1}}^{{5}} x_{{ij}} \le p
        \quad \forall\, s\in\{{1,2,3,4\}}$$
        Für Abschnitt **2→3** ($s=2$) ausgeschrieben:
        $$x_{{13}}+x_{{14}}+x_{{15}}+x_{{23}}+x_{{24}}+x_{{25}} \;\le\; p$$

        **b5)** Ein Fahrgast $1\to5$ sitzt auf **allen vier** Abschnitten im Bus und
        belegt dort je einen Sitz. Die Kapazität bezieht sich daher auf den
        **Querschnitt je Abschnitt**, nicht auf eine einzelne Kante.

        **Optimaler Umsatz:** $z^* = {_z:.0f}$\$. Auslastung je Abschnitt —
        {_load_md}. Abschnitte am Limit $p={_p}$ sind der **Engpass**: dort
        verdrängen umsatzstärkere Fahrgäste die schwächeren.
        """),
        ])
    })
    return


@app.cell(hide_code=True)
def _(SPEC_COFFEE, build_graph, draw_spec, mo):
    _fig = draw_spec(SPEC_COFFEE, build_graph(SPEC_COFFEE), flows=False,
                     title="Coffee Supply Network — Plantagen → Mühlen → Häfen → Röstereien → Cafés",
                     figsize=(12, 6))
    mo.vstack([
        mo.md(r"""
        ## Aufgabe 5 — Mehrstufiges Versorgungsnetz *(Bonus · Vertiefung — nur Notebook)*

        Eine Kaffee-Lieferkette: vier **Plantagen** (Angebot, grün) versorgen über
        **Mühlen → Exporthäfen → Importhäfen → Röstereien** vier **Cafés**
        (Nachfrage, rot). Kanten tragen **Kosten je kg**. Beim Rösten gibt es
        **Schwund**: auf den Kanten *Hafen → Rösterei* kommt nur ein Anteil
        (yield $=0{,}9$) an.
        """),
        mo.as_html(_fig),
        mo.md(r"""
        **a)** Warum ist das trotz fünf Stufen **dasselbe** LP-Muster wie in
        Aufgabe 3? Was ist die Entscheidungsvariable je Kante?

        **b)** Wie verändert der **yield-Faktor** $0{,}9$ die Knotenbilanz einer
        Rösterei? Schreiben Sie die Bilanz für **Roastery\_WZ** schematisch
        (Zufluss $\cdot\,0{,}9 \ge$ Abfluss).

        **c)** Stellen Sie das **vollständige Optimierungsproblem** auf:
        **Entscheidungsvariablen**, **Zielfunktion** und **Nebenbedingungen**
        (Knotenbilanz je Knoten mit yield $+$ Nichtnegativität) in $\sum$/$\forall$-Form.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(SPEC_COFFEE, draw_spec, mo, solve_spec):
    def _coffee():
        G, z = solve_spec(SPEC_COFFEE)
        fig = draw_spec(SPEC_COFFEE, G, flows=True,
                        title=f"Coffee Supply Network — optimale Flüsse, Gesamtkosten = {z:.0f}",
                        figsize=(12, 6))
        return fig, z

    _fig, _z = _coffee()
    mo.accordion({
        "💡 Lösung zu Aufgabe 5": mo.vstack([
            mo.md(rf"""
        **a)** Es bleibt: *eine Variable je Kante* $X_{{ij}}=$ transportierte kg von
        $i$ nach $j$, *eine Bilanz je Knoten*. Mehr Stufen = mehr Knoten/Kanten,
        nicht mehr Modell-Logik.

        **b)** Mit Ausbeute $y=0{{,}}9$ zählt am Knoten nur der *ankommende* Anteil:
        $$0{{,}}9\cdot\big(X_{{\text{{Ham,WZ}}}} + X_{{\text{{Tri,WZ}}}}\big)
        \;\ge\; X_{{\text{{WZ,Altstadt}}}} + X_{{\text{{WZ,Campus}}}} + \dots$$

        **c)** Vollständiges LP (Min-Kosten-Fluss mit Ausbeute $y_{{ij}}$):
        $$\begin{{aligned}}
        \min\ & \sum_{{i=1}}^{{N}}\sum_{{j=1}}^{{N}} c_{{ij}}\,X_{{ij}}
        & (\text{{Gesamtkosten}})\\
        \text{{u.d.N.}}\ & \sum_{{i=1}}^{{N}} y_{{ik}}\,X_{{ik}} + a_k \ge \sum_{{j=1}}^{{N}} X_{{kj}} + d_k
        \ \ \forall k & (\text{{Knotenbilanz}})\\
        & X_{{ij}} \ge 0 \ \ \forall\,i,j & (\text{{Nichtnegativität}})
        \end{{aligned}}$$
        mit $N$ = Anzahl Knoten (nicht vorhandene Kanten zählen als $X_{{ij}}=0$),
        $y_{{ij}}=0{{,}}9$ auf *Hafen$\to$Rösterei* (sonst $1$), Angebot $a_k$ an
        den Plantagen, Nachfrage $d_k$ an den Cafés (Gesamtkosten $z^* = {_z:.0f}$).
        """),
            mo.as_html(_fig),
        ])
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Teil 7 · Besprechung & Ausblick

    - **Ein Muster für alles:** Transport, kürzester Pfad, Zuordnung und Max-Flow sind
      Varianten desselben LPs — *eine Variable je Kante, eine Bilanz je Knoten*.
    - **Hand vs. Solver:** Sie formulieren das LP (Klausur); der Solver (PuLP) rechnet.
      In der Sandbox haben Sie gesehen, wie die Lösung auf Parameter reagiert.
    - **Ausblick VL 09 — Netzplantechnik (CPM):** dieselbe Netzwerklogik, aber die
      Ressource ist **Zeit** — früheste/späteste Termine, Puffer, kritischer Pfad.
    """)
    return


if __name__ == "__main__":
    app.run()
