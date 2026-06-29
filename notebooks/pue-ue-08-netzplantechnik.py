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
    app_title="PuE Übung 8: Netzplantechnik",
    layout_file="layouts/pue-ue-08-netzplantechnik.slides.json",
    css_file="custom.css",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import networkx as nx
    import numpy as np
    import pulp as pl
    from wigglystuff import PlaySlider
    return PlaySlider, mo, nx, pl


@app.cell(hide_code=True)
def fixed_edgedraw():
    # EdgeDraw-Variante mit fixierten Ankern: Start links, Ende rechts.
    # Wir patchen das mitgelieferte (Force-Layout-)JS, sodass Knoten namens
    # "Start"/"Ende" feste Positionen (fx/fy) bekommen — der Rest bleibt frei
    # zeichenbar. Funktioniert lokal wie im Browser (WASM).
    from pathlib import Path
    import anywidget
    import traitlets
    import wigglystuff

    _base = Path(wigglystuff.__file__).parent / "static"
    _src = (_base / "edgedraw.js").read_text()
    _needle = "({id:y,x:100,y:100})"
    _repl = ('({id:y,x:y==="Start"?70:y==="Ende"?530:300,y:200,'
             'fx:y==="Start"?70:(y==="Ende"?530:null),'
             'fy:(y==="Start"||y==="Ende")?200:null})')
    _js = _src.replace(_needle, _repl)  # bei abweichender Version: unverändert

    class FixedEdgeDraw(anywidget.AnyWidget):
        _esm = _js
        _css = (_base / "edgedraw.css").read_text()
        names = traitlets.List([]).tag(sync=True)
        links = traitlets.List([]).tag(sync=True)
        directed = traitlets.Bool(True).tag(sync=True)
        height = traitlets.Int(400).tag(sync=True)
        width = traitlets.Int(600).tag(sync=True)

    return (FixedEdgeDraw,)


@app.cell(hide_code=True)
def solver(pl):
    # ─────────────────────────────────────────────────────────────────────
    # Hintergrund-Solver — löst PuLP-Modelle via scipy/HiGHS (auch im Browser).
    # Wird in Teil 3 (CPM als LP) gebraucht; Studierende sehen davon nichts.
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
        if prob.sense == pl.LpMaximize:
            c = -c
        bounds = [(v.lowBound, v.upBound) for v in variables]
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
    solver_ready = True
    return (solver_ready,)


@app.cell(hide_code=True)
def colors():
    IMSBlue = "#023B88"
    IMSOrange = "#D87237"
    farbe_normal = "#9FBEE6"      # unkritische Aktivität
    farbe_krit = "#F3C9A8"        # kritische Aktivität
    farbe_dummy = "#d9dde2"       # Start/Ende-Dummy
    return IMSBlue, IMSOrange, farbe_dummy, farbe_krit, farbe_normal


@app.cell(hide_code=True)
def engine(
    IMSBlue,
    IMSOrange,
    farbe_dummy,
    farbe_krit,
    farbe_normal,
    nx,
    pl,
    solver_ready,
):
    # ─────────────────────────────────────────────────────────────────────
    # CPM-Engine. Eine Aktivitätsliste ist ein dict
    #     id -> {"name":, "p": Dauer, "pred": [Vorgänger]}.
    # Optional lag: {(i, j): d}  Mindestabstand auf Kante i→j.
    # Notation: FSZ Frühester Start · FEZ Frühestes Ende · SSZ Spätester Start
    #           · SEZ Spätestes Ende · TF Gesamtpuffer = SSZ−FSZ = SEZ−FEZ.
    # ─────────────────────────────────────────────────────────────────────
    _ = solver_ready

    def _topo(acts):
        G = nx.DiGraph(); G.add_nodes_from(acts)
        for j in acts:
            for i in acts[j]["pred"]:
                if i in acts:
                    G.add_edge(i, j)

        def key(n):
            try:
                return (0, int(n))
            except (TypeError, ValueError):
                return (1, str(n))
        return list(nx.lexicographical_topological_sort(G, key=key))

    def cpm_solve(acts, lag=None):
        """Vorwärts-/Rückwärtsrechnung mit optionalem Mindestabstand."""
        lag = lag or {}
        order = _topo(acts)
        succ = {i: [k for k in acts if i in acts[k]["pred"]] for i in acts}
        FSZ, FEZ = {}, {}
        for j in order:                                   # Vorwärtsrechnung
            FSZ[j] = max((FEZ[i] + lag.get((i, j), 0) for i in acts[j]["pred"]),
                         default=0)
            FEZ[j] = FSZ[j] + acts[j]["p"]
        T = max(FEZ.values()) if FEZ else 0               # Projektdauer
        SEZ, SSZ = {}, {}
        for j in reversed(order):                         # Rückwärtsrechnung
            SEZ[j] = min((SSZ[k] - lag.get((j, k), 0) for k in succ[j]),
                         default=T)
            SSZ[j] = SEZ[j] - acts[j]["p"]
        TF = {j: SSZ[j] - FSZ[j] for j in acts}
        crit = {j for j in acts if abs(TF[j]) < 1e-9}
        return dict(FSZ=FSZ, FEZ=FEZ, SSZ=SSZ, SEZ=SEZ, TF=TF,
                    crit=crit, T=T, order=order, succ=succ)

    def build_acts(names, links, durations):
        """EdgeDraw-Links ({source,target}) + Dauern → Aktivitätsliste."""
        pred = {n: [] for n in names}
        for ln in links:
            s, t = str(ln.get("source")), str(ln.get("target"))
            if s in pred and t in pred and s not in pred[t]:
                pred[t].append(s)
        return {n: {"name": "", "p": int(durations.get(n, 1)), "pred": pred[n]}
                for n in names}

    def _dg(acts):
        G = nx.DiGraph(); G.add_node("Start"); G.add_node("Ende")
        for j in acts:
            G.add_node(j)
        for j in acts:
            if acts[j]["pred"]:
                for i in acts[j]["pred"]:
                    G.add_edge(i, j)
            else:
                G.add_edge("Start", j)
        for j in acts:
            if not any(j in acts[k]["pred"] for k in acts):
                G.add_edge(j, "Ende")
        return G

    def _nk(n):
        try:
            return (0, int(n))
        except (TypeError, ValueError):
            return (1, str(n))

    def _suffix(j, acts):
        nm = acts[j].get("name", "")
        return f"{j}: {nm}" if nm else f"{j}"

    # Geometrie des Layer-Layouts (Datenkoordinaten). Spalten- und Zeilen-
    # abstand sind größer als die Box → Knoten überlappen nie. Lange Kanten
    # werden unter den Knoten (in einer freien Spur) herumgeführt.
    _XG, _YG, _W, _H = 3.6, 2.0, 2.5, 1.45

    def _layer_pos(acts):
        G = _dg(acts)
        gens = list(nx.topological_generations(G))
        layer = {n: i for i, g in enumerate(gens) for n in g}
        pos = {}
        for li, g in enumerate(gens):
            og = sorted(g, key=lambda n: (str(n) != "Start",
                                          str(n) == "Ende", _nk(n)))
            k = len(og)
            for idx, n in enumerate(og):
                pos[n] = (li * _XG, ((k - 1) / 2 - idx) * _YG)
        return G, layer, pos

    def _draw_net(acts, lag=None, *, visible=None, fwd=None, bwd=None,
                  show_crit=True, cur=None, show_ende=True, title="",
                  annot="", callouts=None, figsize=(9.5, 5.4)):
        """Kern-Zeichner (überlappungsfreies Layer-Layout). visible/fwd/bwd
        steuern, was sichtbar ist bzw. welche Zeiten schon eingetragen sind."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        from matplotlib.path import Path

        r = cpm_solve(acts, lag); lag = lag or {}
        G, layer, pos = _layer_pos(acts)
        if visible is None:
            visible = set(acts)
        fwd = set(acts) if fwd is None else fwd
        bwd = set(acts) if bwd is None else bwd

        FSZ, FEZ = dict(r["FSZ"]), dict(r["FEZ"])
        SSZ, SEZ = dict(r["SSZ"]), dict(r["SEZ"])
        T = r["T"]; crit = set(r["crit"])
        for d in (FSZ, FEZ, SSZ, SEZ):
            d["Start"] = 0
        FSZ["Ende"] = FEZ["Ende"] = SSZ["Ende"] = SEZ["Ende"] = T
        crit |= {"Start", "Ende"}

        nodes_vis = {"Start"} | set(visible)
        if show_ende:
            nodes_vis |= {"Ende"}
        draw_nodes = [n for n in G.nodes() if n in nodes_vis]
        draw_edges = [(u, v) for u, v in G.edges()
                      if u in nodes_vis and v in nodes_vis]

        def _col(n):
            if n in ("Start", "Ende"):
                return farbe_dummy
            return farbe_krit if (show_crit and n in crit) else farbe_normal

        def _lab(n):
            if n in ("Start", "Ende"):
                return f"{n}\n{FSZ[n]} | {FEZ[n]}"
            head = f"{_suffix(n, acts)} (p={acts[n]['p']})"
            l1 = f"\nFSZ {FSZ[n]} | FEZ {FEZ[n]}" if n in fwd else "\n·  |  ·"
            l2 = f"\nSSZ {SSZ[n]} | SEZ {SEZ[n]}" if n in bwd else "\n·  |  ·"
            return head + l1 + l2

        W, H = _W, _H
        ys = [p[1] for p in pos.values()] or [0]   # ganzer Graph → stabiler Ausschnitt
        ymin, ymax = min(ys), max(ys)
        lane = ymin - H - 0.7

        fig, ax = plt.subplots(figsize=figsize)

        def _clear(u, v, x1, y1, x2, y2):
            # Kreuzt die gerade Verbindung einen anderen sichtbaren Knoten?
            for w in draw_nodes:
                if w in (u, v):
                    continue
                xw, yw = pos[w]
                if x1 + W / 2 < xw < x2 - W / 2 and abs(x2 - x1) > 1e-9:
                    yl = y1 + (y2 - y1) * (xw - x1) / (x2 - x1)
                    if abs(yl - yw) < H / 2 + 0.2:
                        return False
            return True

        # ── Kanten: direkt (gerade), sonst — falls ein Knoten im Weg liegt —
        #    unten herum. Keine unnötigen Umwege.
        for u, v in draw_edges:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            on_cp = (show_crit and u in crit and v in crit
                     and FEZ[u] + lag.get((u, v), 0) == FSZ[v])
            ec = IMSOrange if on_cp else "#9aa3ad"
            lw = 2.6 if on_cp else 1.3
            if layer[v] - layer[u] <= 1 or _clear(u, v, x1, y1, x2, y2):
                pa = FancyArrowPatch((x1 + W / 2, y1), (x2 - W / 2, y2),
                                     arrowstyle="-|>", mutation_scale=13, lw=lw,
                                     color=ec, shrinkA=1, shrinkB=2, zorder=1)
                if lag.get((u, v), 0):
                    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18,
                            f"+{lag[(u, v)]}", color=IMSOrange, fontsize=8,
                            fontweight="bold", ha="center", zorder=4)
            else:
                xa, xb = x1 + W / 2 + 0.18, x2 - W / 2 - 0.18
                verts = [(x1 + W / 2, y1), (xa, y1), (xa, lane),
                         (xb, lane), (xb, y2), (x2 - W / 2, y2)]
                pa = FancyArrowPatch(
                    path=Path(verts, [Path.MOVETO] + [Path.LINETO] * 5),
                    arrowstyle="-|>", mutation_scale=13, lw=lw, color=ec,
                    shrinkA=1, shrinkB=2, zorder=1, joinstyle="round")
                if lag.get((u, v), 0):
                    ax.text((xa + xb) / 2, lane + 0.16, f"+{lag[(u, v)]}",
                            color=IMSOrange, fontsize=8, fontweight="bold",
                            ha="center", zorder=4)
            ax.add_patch(pa)
        # ── Knoten als Boxen (feste Größe in Datenkoordinaten) ──
        for n in draw_nodes:
            x, y = pos[n]
            ax.add_patch(FancyBboxPatch(
                (x - W / 2, y - H / 2), W, H,
                boxstyle="round,pad=0.02,rounding_size=0.14", fc=_col(n),
                ec=(IMSOrange if n == cur else "#33425a"),
                lw=(2.6 if n == cur else 1.1), zorder=2))
            ax.text(x, y, _lab(n), ha="center", va="center", fontsize=6.8,
                    zorder=3)
        # ── optionale Erklär-Callouts ──
        for txt, node, tx, ty, c, rad in (callouts or []):
            ax.annotate(txt, xy=pos[node], xytext=(tx, ty), fontsize=8,
                        color=c, ha="center", va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=c, lw=1.0),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.3,
                                        shrinkB=22,
                                        connectionstyle=f"arc3,rad={rad}"))
        nl = max(layer.values()) + 1
        ax.set_xlim(-W - 1.0, (nl - 1) * _XG + W + 0.6)
        ax.set_ylim(lane - 0.8, ymax + H / 2 + 1.7)
        ax.set_aspect("equal"); ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        if annot:
            ax.text(0.5, -0.02, annot, transform=ax.transAxes, ha="center",
                    va="top", fontsize=9.5, color="#33425a",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#fff6ef",
                              ec=IMSOrange, lw=1.0))
        fig.tight_layout()
        return fig

    def draw_network(acts, lag=None, title="", figsize=(9.5, 5.4)):
        return _draw_net(acts, lag, title=title, figsize=figsize)

    def draw_explain(acts, figsize=(7.2, 5.6)):
        """Teil-1-Erklärgrafik: Netz mit eingetragenen Zeiten + Callouts
        'Vorwärts = max' / 'Rückwärts = min' in den freien Rändern."""
        co = [("Vorwärts:\n$FSZ_j=\\max$ der Vorgänger", 4,
               7.2, 3.0, IMSBlue, -0.2),
              ("Rückwärts:\n$SEZ_j=\\min$ der Nachfolger", 2,
               0.6, -3.4, IMSOrange, -0.3)]
        return _draw_net(acts, callouts=co, figsize=figsize)

    def anim_state(acts, step):
        """Schritt → (Phase, sichtbare Knoten, fertige fwd/bwd, kritisch)."""
        order = _topo(acts); N = len(order); total = 3 * N + 1
        step = max(0, min(int(step), total))
        if step <= N:
            phase, k = "Aufbau", step
        elif step <= 2 * N:
            phase, k = "Vorwärts", step - N
        elif step <= 3 * N:
            phase, k = "Rückwärts", step - 2 * N
        else:
            phase, k = "Kritisch", 0
        rev = list(reversed(order))
        visible = set(order[:k]) if phase == "Aufbau" else set(order)
        fwd = (set(order) if phase in ("Rückwärts", "Kritisch")
               else set(order[:k]) if phase == "Vorwärts" else set())
        bwd = (set(order) if phase == "Kritisch"
               else set(rev[:k]) if phase == "Rückwärts" else set())
        cur = None
        if phase == "Aufbau" and k:
            cur = order[k - 1]
        elif phase == "Vorwärts" and k:
            cur = order[k - 1]
        elif phase == "Rückwärts" and k:
            cur = rev[k - 1]
        return dict(phase=phase, visible=visible, fwd=fwd, bwd=bwd,
                    crit=(phase == "Kritisch"), cur=cur, total=total)

    def draw_anim(acts, step, lag=None, figsize=(9.5, 5.6)):
        """Animations-Frame: zeichnet den Zustand bei `step` mit Erklärtext."""
        r = cpm_solve(acts, lag); lag = lag or {}
        st = anim_state(acts, step)
        cur = st["cur"]; ph = st["phase"]
        annot = {"Aufbau": "Netz aufbauen — Knoten und Vorgängerkanten erscheinen.",
                 "Vorwärts": "Vorwärtsrechnung läuft …",
                 "Rückwärts": "Rückwärtsrechnung läuft …",
                 "Kritisch": ""}[ph]
        if ph == "Aufbau" and cur is not None:
            pv = ",".join(str(p) for p in acts[cur]["pred"]) or "Start"
            annot = f"Aktivität {cur} (p={acts[cur]['p']}) — Vorgänger: {pv}"
        elif ph == "Vorwärts" and cur is not None:
            preds = acts[cur]["pred"]
            if preds:
                terms = ", ".join(f"FEZ_{i}+{lag.get((i,cur),0)}={r['FEZ'][i]+lag.get((i,cur),0)}"
                                  if lag.get((i, cur), 0) else f"FEZ_{i}={r['FEZ'][i]}"
                                  for i in preds)
                annot = (f"FSZ_{cur} = max({terms}) = {r['FSZ'][cur]}   →   "
                         f"FEZ_{cur} = {r['FSZ'][cur]}+{acts[cur]['p']} = {r['FEZ'][cur]}")
            else:
                annot = f"FSZ_{cur} = 0 (kein Vorgänger)   →   FEZ_{cur} = {r['FEZ'][cur]}"
        elif ph == "Rückwärts" and cur is not None:
            ss = r["succ"][cur]
            if ss:
                terms = ", ".join(f"SSZ_{k}={r['SSZ'][k]}" for k in ss)
                annot = (f"SEZ_{cur} = min({terms}) = {r['SEZ'][cur]}   →   "
                         f"SSZ_{cur} = {r['SEZ'][cur]}−{acts[cur]['p']} = {r['SSZ'][cur]}")
            else:
                annot = (f"SEZ_{cur} = Projektende = {r['T']}   →   "
                         f"SSZ_{cur} = {r['SSZ'][cur]}")
        elif ph == "Kritisch":
            cp = " → ".join(str(j) for j in r["order"] if j in r["crit"])
            annot = f"Fertig! Kritischer Pfad: {cp}   ·   Projektdauer = {r['T']}"
        title = f"Schritt {min(int(step), st['total'])}/{st['total']}  ·  Phase: {ph}"
        return _draw_net(acts, lag, visible=st["visible"], fwd=st["fwd"],
                         bwd=st["bwd"], show_crit=st["crit"], cur=cur,
                         show_ende=(ph != "Aufbau" or st["visible"] == set(acts)),
                         title=title, annot=annot, figsize=figsize)

    def draw_anim3(acts, bk, fk, bwk, lag=None, figsize=(9.8, 5.8)):
        """Drei getrennte Slider: bk=Aufbau, fk=Vorwärts, bwk=Rückwärts."""
        r = cpm_solve(acts, lag); lag = lag or {}
        order = r["order"]; N = len(order); rev = list(reversed(order))
        bk = max(0, min(int(bk), N)); fk = max(0, min(int(fk), N))
        bwk = max(0, min(int(bwk), N))
        computing = fk > 0 or bwk > 0
        visible = set(order) if computing else set(order[:bk])
        fwd = set(order[:fk]); bwd = set(rev[:bwk])
        show_crit = (fk >= N and bwk >= N)
        cur = None
        annot = "Slider 1 baut das Netz auf · Slider 2 rechnet vorwärts · Slider 3 rückwärts."
        if bwk > 0:
            cur = rev[bwk - 1]; ss = r["succ"][cur]
            if ss:
                terms = ", ".join(f"SSZ_{k}={r['SSZ'][k]}" for k in ss)
                annot = (f"Rückwärts:  SEZ_{cur} = min({terms}) = {r['SEZ'][cur]}"
                         f"   →   SSZ_{cur} = {r['SEZ'][cur]}−{acts[cur]['p']} = {r['SSZ'][cur]}")
            else:
                annot = (f"Rückwärts:  SEZ_{cur} = Projektende = {r['T']}"
                         f"   →   SSZ_{cur} = {r['SSZ'][cur]}")
        elif fk > 0:
            cur = order[fk - 1]; preds = acts[cur]["pred"]
            if preds:
                terms = ", ".join(
                    f"FEZ_{i}+{lag.get((i, cur), 0)}={r['FEZ'][i]+lag.get((i, cur), 0)}"
                    if lag.get((i, cur), 0) else f"FEZ_{i}={r['FEZ'][i]}" for i in preds)
                annot = (f"Vorwärts:  FSZ_{cur} = max({terms}) = {r['FSZ'][cur]}"
                         f"   →   FEZ_{cur} = {r['FSZ'][cur]}+{acts[cur]['p']} = {r['FEZ'][cur]}")
            else:
                annot = f"Vorwärts:  FSZ_{cur} = 0 (kein Vorgänger)   →   FEZ_{cur} = {r['FEZ'][cur]}"
        elif bk > 0:
            cur = order[bk - 1]
            pv = ",".join(str(p) for p in acts[cur]["pred"]) or "Start"
            annot = f"Aufbau:  Aktivität {cur} (p={acts[cur]['p']}) — Vorgänger: {pv}"
        if show_crit:
            cur = None
            cp = " → ".join(str(j) for j in order if j in r["crit"])
            annot = f"Fertig! Kritischer Pfad: {cp}   ·   Projektdauer = {r['T']}"
        title = f"Aufbau {bk}/{N}  ·  Vorwärts {fk}/{N}  ·  Rückwärts {bwk}/{N}"
        return _draw_net(acts, lag, visible=visible, fwd=fwd, bwd=bwd,
                         show_crit=show_crit, cur=cur,
                         show_ende=(computing or bk == N), title=title,
                         annot=annot, figsize=figsize)

    def cpm_table(acts, lag=None):
        r = cpm_solve(acts, lag)
        head = ("| $j$ | $p_j$ | Vorg. | $FSZ$ | $FEZ$ | $SSZ$ | $SEZ$ | $TF$ |  |\n"
                "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|\n")
        rows = ""

        def _k(j):
            try:
                return int(j)
            except (TypeError, ValueError):
                return j
        for j in sorted(acts, key=_k):
            kr = j in r["crit"]
            pv = ",".join(str(p) for p in acts[j]["pred"]) or "–"
            cells = [str(j), str(acts[j]["p"]), pv, str(r["FSZ"][j]),
                     str(r["FEZ"][j]), str(r["SSZ"][j]), str(r["SEZ"][j]),
                     f"**{r['TF'][j]}**" if kr else str(r["TF"][j]),
                     "🔶 **krit.**" if kr else ""]
            rows += "| " + " | ".join(cells) + " |\n"
        return head + rows

    def cpm_lp(acts, lag=None):
        """CPM als LP: min S_Ende, frühester Schedule ⇒ S_j = FSZ_j."""
        lag = lag or {}
        m = pl.LpProblem("CPM", pl.LpMinimize)
        S = {j: pl.LpVariable(f"S_{j}", lowBound=0) for j in acts}
        S["Ende"] = pl.LpVariable("S_Ende", lowBound=0)
        m += pl.lpSum(S.values())
        for j in acts:
            for i in acts[j]["pred"]:
                m += S[j] >= S[i] + acts[i]["p"] + lag.get((i, j), 0)
            if not any(j in acts[k]["pred"] for k in acts):
                m += S["Ende"] >= S[j] + acts[j]["p"]
        m.solve()
        return {j: (S[j].varValue or 0.0) for j in S}, S["Ende"].varValue or 0.0

    def schedule_capacity(acts, demand, cap, lag=None):
        """Ressourcenbeschränkte Planung (serielles SGS mit „kritischste zuerst"-
        Priorität): hält die Kapazität ein, respektiert Reihenfolge. Gibt
        (Startzeiten, Makespan) zurück — der Makespan kann über der CPM-Dauer
        liegen, wenn die Kapazität zu klein ist (erzwungene Verlängerung)."""
        from collections import defaultdict
        r = cpm_solve(acts, lag); lag = lag or {}
        fin, sched, usage, remaining = {}, {}, defaultdict(float), set(acts)
        while remaining:
            ready = [j for j in remaining
                     if all(i in fin for i in acts[j]["pred"])]
            j = min(ready, key=lambda x: (r["SSZ"][x], r["FSZ"][x], _nk(x)))
            p, d = acts[j]["p"], demand[j]
            t = max([fin[i] + lag.get((i, j), 0) for i in acts[j]["pred"]],
                    default=0)
            while not all(usage[tau] + d <= cap for tau in range(t, t + p)):
                t += 1
            for tau in range(t, t + p):
                usage[tau] += d
            sched[j], fin[j] = t, t + p
            remaining.discard(j)
        return sched, max(fin.values())

    def draw_resource(acts, demand, cap, shifts=None, lag=None, starts=None,
                      title="", figsize=(8.6, 4.4)):
        """Ressourcenplan im Stil von VL-Folie 18: jede Aktivität ein Balken
        über ihre Dauer, gestapelt; kritische Aktivitäten unten mit ROTEM Rand
        (+ rote Schrift), Kapazitätslinie gestrichelt, Überlast markiert.
        starts überschreibt die Startzeiten (z. B. aus schedule_capacity)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        r = cpm_solve(acts, lag); shifts = shifts or {}
        if starts is None:
            starts = {j: r["FSZ"][j] + shifts.get(j, 0) for j in acts}
        crit = r["crit"]
        Tend = int(max(max(starts[j] + acts[j]["p"] for j in acts), r["T"]))
        # Greedy-Stapelung: kritische zuerst (unten), dann übrige — je nach Start.
        order = sorted(acts, key=lambda j: (0 if j in crit else 1, starts[j], _nk(j)))
        base = [0.0] * Tend
        bars = []
        for j in order:
            s, e = int(starts[j]), int(starts[j] + acts[j]["p"])
            y0 = max(base[s:e]) if e > s else 0.0
            for t in range(s, e):
                base[t] = y0 + demand[j]
            bars.append((j, s, acts[j]["p"], y0, demand[j], j in crit))
        profile = [sum(demand[j] for j in acts
                       if int(starts[j]) <= t < int(starts[j]) + acts[j]["p"])
                   for t in range(Tend)]
        over = [t for t in range(Tend) if profile[t] > cap]
        ymax = max(max(profile) if profile else 0, cap) + 1

        fig, ax = plt.subplots(figsize=figsize)
        if over:
            ax.add_patch(Rectangle((0, cap), Tend, ymax - cap, facecolor="#C0392B",
                                   alpha=0.07, edgecolor="none", zorder=0))
        for j, s, p, y0, h, isc in bars:
            fc = "#9FBEE6" if isc else "#dce7f5"
            ec = "#C0392B" if isc else "#5b6b7d"
            ax.add_patch(Rectangle((s, y0), p, h, facecolor=fc, edgecolor=ec,
                                   linewidth=2.4 if isc else 1.0, zorder=2))
            ax.text(s + p / 2, y0 + h / 2, str(j), ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="#C0392B" if isc else "#23324a", zorder=3)
        ax.axhline(cap, color=IMSOrange, ls="--", lw=2.0, zorder=1)
        ax.text(Tend, cap + 0.06, f" Limit = {cap}", color=IMSOrange,
                fontsize=9, va="bottom", ha="right")
        for t in over:                                   # Überlast-Spalten markieren
            ax.plot([t, t + 1], [-0.05, -0.05], color="#C0392B", lw=4,
                    solid_capstyle="butt", zorder=4, clip_on=False)
        ax.set_xlim(0, Tend); ax.set_ylim(0, ymax)
        ax.set_xticks(range(Tend + 1))
        ax.set_xlabel("Zeit $t$", fontsize=9)
        ax.set_ylabel("Bedarf $r$", fontsize=9)
        ax.set_title(title or "Ressourcenplan", fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#e9ecf0", linewidth=0.6, zorder=0)
        ax.text(0.0, ymax + 0.02, "roter Rand = kritischer Pfad", fontsize=7.5,
                color="#C0392B", va="bottom")
        fig.tight_layout()
        return fig

    return (anim_state, build_acts, cpm_lp, cpm_solve, cpm_table, draw_anim,
            draw_anim3, draw_explain, draw_network, draw_resource,
            schedule_capacity)


@app.cell(hide_code=True)
def specs():
    # Mini-Case (Projektdauer 8, kritischer Pfad 2→4→6)
    ACTS_MINI = {
        1: {"name": "", "p": 2, "pred": []},
        2: {"name": "", "p": 3, "pred": []},
        3: {"name": "", "p": 1, "pred": []},
        4: {"name": "", "p": 4, "pred": [1, 2]},
        5: {"name": "", "p": 2, "pred": [2]},
        6: {"name": "", "p": 1, "pred": [4]},
    }
    DEMAND_MINI = {1: 2, 2: 3, 3: 2, 4: 2, 5: 1, 6: 2}   # Personalbedarf

    # Website-Relaunch (Projektdauer 14, kritischer Pfad 1→3→4→6→7)
    ACTS_WEB = {
        1: {"name": "Konzept", "p": 3, "pred": []},
        2: {"name": "Inhalte", "p": 4, "pred": [1]},
        3: {"name": "Design", "p": 2, "pred": [1]},
        4: {"name": "Programmierung", "p": 5, "pred": [3]},
        5: {"name": "Einpflegen", "p": 2, "pred": [2, 4]},
        6: {"name": "Testing", "p": 3, "pred": [4]},
        7: {"name": "Go-Live", "p": 1, "pred": [5, 6]},
    }
    DEMAND_WEB = {1: 2, 2: 3, 3: 2, 4: 3, 5: 2, 6: 2, 7: 1}

    # Geführtes Beispiel "Produkteinführung" (Projektdauer 10, krit. 1→2→4→6)
    # Slack auf 3 und 5 (je TF=3); Bedarf erzeugt Spitze 6 an t=4.
    ACTS_GUIDE = {
        1: {"name": "Marktanalyse", "p": 2, "pred": []},
        2: {"name": "Entwicklung", "p": 3, "pred": [1]},
        3: {"name": "Marketingkonzept", "p": 2, "pred": [1]},
        4: {"name": "Produktion", "p": 4, "pred": [2]},
        5: {"name": "Vertriebsschulung", "p": 2, "pred": [3]},
        6: {"name": "Launch", "p": 1, "pred": [4, 5]},
    }
    DEMAND_GUIDE = {1: 1, 2: 3, 3: 2, 4: 2, 5: 3, 6: 1}

    # Alternativ-Aufgabe "Messestand" (alle Bestandteile; Dauer 14, krit. 1→2→4→6→7)
    ACTS_ALT = {
        1: {"name": "Briefing", "p": 2, "pred": []},
        2: {"name": "Standdesign", "p": 4, "pred": [1]},
        3: {"name": "Catering", "p": 2, "pred": [1]},
        4: {"name": "Standbau", "p": 5, "pred": [2]},
        5: {"name": "Personal", "p": 3, "pred": [3]},
        6: {"name": "Generalprobe", "p": 2, "pred": [4, 5]},
        7: {"name": "Messe", "p": 1, "pred": [6]},
    }
    DEMAND_ALT = {1: 2, 2: 2, 3: 2, 4: 3, 5: 2, 6: 2, 7: 1}
    return (ACTS_ALT, ACTS_GUIDE, ACTS_MINI, ACTS_WEB, DEMAND_ALT, DEMAND_GUIDE,
            DEMAND_MINI, DEMAND_WEB)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Übung 8 · Netzplantechnik (CPM)

    **Planung und Entscheidung — SS 2026 · Begleitnotebook zu VL 09**

    Ein **Projekt** ist eine Menge von Aktivitäten mit **Reihenfolge­beziehungen**.
    Die *Critical Path Method* (CPM) beantwortet mit **einem** Rechenschema:
    **Wie lange dauert das Projekt? Welche Aktivitäten sind kritisch? Wo gibt es
    Puffer?**

    > **Klausurfokus:** aus einer **Vorgangsliste** das Vorgangsknotennetz aufstellen
    > und **von Hand** Vorwärts-/Rückwärtsrechnung, Pufferzeiten und kritischen Pfad
    > bestimmen. Die interaktiven Grafiken helfen beim Verstehen; rechnen Sie zuerst
    > selbst, dann kontrollieren Sie mit der Animation.

    **Notation:** $FSZ$ Frühester Start, $FEZ$ Frühestes Ende, $SSZ$ Spätester Start,
    $SEZ$ Spätestes Ende, $TF$ Gesamtpuffer.

    **Ablauf (90 Min.):**
    1. **Wiederholung** — Begriffe, Netzplan, Rechenregeln (3 Spalten)
    2. **Geführte Übung** — Mini-Netz: Netz & Vorwärts-/Rückwärtsrechnung **animiert**
    3. **Erweiterungen am Mini-Netz** — Mindestabstand & Kapazität *(Teilaufgaben zu Teil 2)*
    4. **Sandbox** — eigenes Netz **zeichnen**, lösen, Animation, Ressourcenplan
    5. **Spickzettel** — Rechenschema & Klausur-Tipps
    6. **Selbständige Aufgaben** — CPM an mehreren Fällen (Website, Verzögerung, Kapazität)
    7. **Besprechung & Ausblick**
    """)
    return


@app.cell(hide_code=True)
def _(ACTS_MINI, draw_explain, mo):
    # ── Teil 1 · Wiederholung — 3 Spalten: Begriffe | Netzplan | Regeln ──
    def _explain_svg():
        import io
        import re
        import matplotlib.pyplot as plt
        fig = draw_explain(ACTS_MINI, figsize=(7.0, 5.8))
        buf = io.StringIO(); fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg = buf.getvalue(); svg = svg[svg.index("<svg"):]
        svg = re.sub(r'\s(?:width|height)="[^"]*"', "", svg, count=2)
        svg = svg.replace("<svg ", '<svg style="width:100%;height:auto;display:block" ', 1)
        return mo.Html(svg)

    _c1 = mo.md(r"""
    ## 1 · Begriffe

    Ein **Projekt** = Aktivitäten $j$ mit Dauer $p_j$ und **Vorgängern**.

    Im **Vorgangsknotennetz**:

    - jede **Aktivität** → **Knoten** (Name, $p_j$, Zeiten),
    - jede **Vorgängerbeziehung** → **Pfeil**,
    - **Start**/**Ende** als Dummy ($p=0$).

    Jeder Knoten trägt vier Zeiten:
    $$\begin{array}{|c|c|}\hline FSZ & FEZ \\ \hline SSZ & SEZ \\ \hline\end{array}$$
    oben **früheste**, unten **späteste** Zeiten.
    """)

    _c3 = mo.md(r"""
    ## 3 · Rechenregeln

    **Vorwärts** (Start → Ende):
    $$FSZ_j = \max_{i\in\text{Vorg}(j)} FEZ_i$$
    $$FEZ_j = FSZ_j + p_j$$

    **Rückwärts** (Ende → Start):
    $$SEZ_j = \min_{k\in\text{Nachf}(j)} SSZ_k$$
    $$SSZ_j = SEZ_j - p_j$$

    **Puffer & kritisch:**
    $$TF_j = SSZ_j - FSZ_j = SEZ_j - FEZ_j$$

    - $TF_j=0$ ⇒ **kritisch** (bindend),
    - $TF_j>0$ ⇒ um $TF_j$ verschiebbar.

    Der **kritische Pfad** = $TF{=}0$-Kette von Start zu Ende; seine Länge ist die
    **Projektdauer**.
    """)

    mo.vstack([
        mo.md("---\n# Teil 1 · Wiederholung"),
        mo.hstack([
            _c1,
            mo.vstack([mo.md("## 2 · Netzplan (Mini-Case)"), _explain_svg()]),
            _c3,
        ], widths=[0.30, 0.40, 0.30], gap=1.2, align="start"),
    ], gap=0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    _liste = mo.md(r"""
    **Vorgangsliste — Projekt „Produkteinführung"**

    | $j$ | Aktivität | $p_j$ (Tage) | Vorgänger | Bedarf $r_j$ |
    |:--:|:--|:--:|:--:|:--:|
    | 1 | Marktanalyse | 2 | – | 1 |
    | 2 | Entwicklung | 3 | 1 | 3 |
    | 3 | Marketingkonzept | 2 | 1 | 2 |
    | 4 | Produktion | 4 | 2 | 2 |
    | 5 | Vertriebsschulung | 2 | 3 | 3 |
    | 6 | Launch | 1 | 4, 5 | 1 |
    """)
    mo.vstack([
        mo.md(r"""
        ---
        # Teil 2 · Geführte Übung — Produkteinführung

        Wir begleiten **ein Projekt** durch das ganze Tutorial. Stellen Sie zunächst
        das Netz auf und rechnen Sie CPM **von Hand**; danach kontrollieren Sie mit der
        **Animation** (Play oder Slider: Aufbau → Vorwärts → Rückwärts → kritischer Pfad).
        Die Spalte **Bedarf $r_j$** brauchen wir erst in Teil 3.
        """),
        _liste,
        mo.md(r"""
        **a)** Zeichnen Sie das Vorgangsknotennetz (Dummy Start/Ende).
        **b)** Vorwärtsrechnung: $FSZ_j,\,FEZ_j$ und die **Projektdauer**.
        **c)** Rückwärtsrechnung & Puffer: $SSZ_j,\,SEZ_j,\,TF_j$ — bestimmen Sie den
        **kritischen Pfad** und die Aktivität mit dem **größten Puffer**.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(ACTS_GUIDE, PlaySlider, mo):
    _N = len(ACTS_GUIDE)
    g1_build = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=_N,
                              step=1, interval_ms=850, loop=False, width=300))
    g1_fwd = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=_N,
                             step=1, interval_ms=850, loop=False, width=300))
    g1_bwd = mo.ui.anywidget(PlaySlider(value=0, min_value=0, max_value=_N,
                             step=1, interval_ms=850, loop=False, width=300))
    return g1_build, g1_bwd, g1_fwd


@app.cell(hide_code=True)
def _(ACTS_GUIDE, cpm_table, draw_anim3, g1_build, g1_bwd, g1_fwd, mo):
    def _v(w):
        pv = w.value
        return int(round(pv.get("value", 0) if isinstance(pv, dict) else pv))

    _fig = draw_anim3(ACTS_GUIDE, _v(g1_build), _v(g1_fwd), _v(g1_bwd),
                      figsize=(11, 5.6))
    mo.accordion({
        "🎬 Animierte Lösung — drei Slider: Aufbau · Vorwärts · Rückwärts":
        mo.vstack([
            mo.hstack([
                mo.vstack([mo.md("**1) Aufbau**"), g1_build]),
                mo.vstack([mo.md("**2) Vorwärts**"), g1_fwd]),
                mo.vstack([mo.md("**3) Rückwärts**"), g1_bwd]),
            ], gap=1.2, justify="start"),
            mo.as_html(_fig),
            mo.md("**CPM-Tabelle (Endergebnis):**\n\n" + cpm_table(ACTS_GUIDE)),
            mo.md(r"""
        **b)** Vorwärts: $FSZ_6=\max(FEZ_4,FEZ_5)=\max(9,6)=9$ ⇒ **Projektdauer = 10**.
        **c)** Rückwärts: $SEZ_1=\min(SSZ_2,SSZ_3)=\min(2,5)=2$. Kritisch ($TF=0$):
        $1\rightarrow2\rightarrow4\rightarrow6$. **Größter Puffer:** Marketingkonzept (3)
        und Vertriebsschulung (5) mit je $TF=3$.
        (Erst alle drei Slider bis ans Ende ziehen → der kritische Pfad erscheint.)
        """),
        ])
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Teil 3 · Erweiterungen — dieselbe Produkteinführung

    *Das sind **weitere Teilaufgaben (d–f) zum Netz aus Teil 2** — kein neues
    Beispiel.* In der Praxis kommen zum reinen CPM zwei Dinge dazu, die wir hier vor
    allem **visuell** kennenlernen:

    | Zeitliche Restriktion | Ressourcen |
    |---|---|
    | Mindest-/Höchstabstand zwischen zwei Aktivitäten | begrenzte Teamkapazität → **Kapazitätsglättung** |

    **d) Mindest-/Höchstabstand** · **e/f) Kapazitätsglättung**. Beides ändert das
    CPM-Schema kaum — der Kern bleibt: *Netz aufstellen, vorwärts/rückwärts rechnen,
    Puffer interpretieren.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    g2_lag = mo.ui.slider(start=0, stop=6, step=1, value=0,
                          label="Mindestabstand $d$ auf Kante $1\\to3$",
                          show_value=True)
    return (g2_lag,)


@app.cell(hide_code=True)
def _(ACTS_GUIDE, cpm_solve, draw_network, g2_lag, mo):
    def _g31():
        d = int(g2_lag.value)
        lag = {(1, 3): d} if d else None
        r = cpm_solve(ACTS_GUIDE, lag)
        cp = " → ".join(str(j) for j in r["order"] if j in r["crit"])
        fig = draw_network(
            ACTS_GUIDE, lag, figsize=(11, 5.4),
            title=f"Mindestabstand d={d} auf 1→3 (Marktanalyse→Marketing) — "
                  f"Projektdauer = {r['T']}")
        return r, cp, fig

    _r, _cp, _fig = _g31()
    mo.vstack([
        mo.md(r"""
        ## d) Mindest- & Höchstabstand

        **Kernidee:** Ein Abstand ist **nur eine zusätzliche Kante mit Gewicht** — das
        Rechenschema aus Teil 2 bleibt **identisch**, es kommt nur ein Term dazu:

        - **Mindestabstand** $d$ *(z. B. Material muss aushärten, bevor es weitergeht)*:
          $\;S_j \ge S_i + p_i + d$ — in der Vorwärtsrechnung zählt statt $FEZ_i$ nun
          $FEZ_i + d$.
        - **Höchstabstand** *(z. B. „spätestens $d$ Tage nach $i$ muss $j$ starten")*:
          eine rückwärts gerichtete Kante mit Gewicht $-d$, also $S_i \ge S_j - p_i - d$.

        Auf $1\to3$ (Marktanalyse → Marketingkonzept) liegt ein Mindestabstand $d$ —
        eine Kante, die **anfangs nicht kritisch** ist.

        **d1)** Wie groß ist die Projektdauer bei $d=2$? **d2)** Wie groß bei $d=4$?
        **d3)** **Ab welchem $d$** verlängert sich das Projekt, und **warum**?
        *(Slider zum Selbstkontrollieren.)*
        """),
        mo.hstack([g2_lag, mo.md(f"**Projektdauer:** {_r['T']}  ·  **kritisch:** {_cp}")],
                  justify="start", gap=2.0, align="center"),
        mo.as_html(_fig),
        mo.accordion({
            "💡 Lösung zu d)": mo.md(r"""
        Die Kante $1\to3$ liegt **nicht** auf dem kritischen Pfad; der Pfad
        $1\to3\to5\to6$ hat $TF=3$ Puffer.
        - **d1)** $d=2$: Projektdauer **10** (unverändert) — der Puffer schluckt den
          Abstand.
        - **d2)** $d=4$: Projektdauer **11**.
        - **d3)** Ab **$d=4$** verlängert sich das Projekt: $FSZ_3=FEZ_1+d=2+d$ schiebt
          $FEZ_5=4+d+2$; sobald $6+d>9$ (also $d>3$) wird $1\to3\to5\to6$ kritisch und
          länger als $1\to2\to4\to6$.

        **Merke:** Ein Abstand auf einer Kante mit Puffer wirkt erst, wenn er den
        Puffer übersteigt — gerechnet wird wie immer, nur mit einem $+d$ in der Kante.
        """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    g2_shift = mo.ui.slider(start=0, stop=3, step=1, value=0,
                            label="Vertriebsschulung (Akt. 5) verschieben (Slack $TF_5=3$)",
                            show_value=True)
    return (g2_shift,)


@app.cell(hide_code=True)
def _(ACTS_GUIDE, DEMAND_GUIDE, cpm_solve, draw_resource, g2_shift, mo):
    def _g32e():
        r = cpm_solve(ACTS_GUIDE)
        sh = int(g2_shift.value)
        shifts = {5: sh}
        starts = {j: r["FSZ"][j] + shifts.get(j, 0) for j in ACTS_GUIDE}
        Tend = max(starts[j] + ACTS_GUIDE[j]["p"] for j in ACTS_GUIDE)
        prof = [sum(DEMAND_GUIDE[j] for j in ACTS_GUIDE
                    if starts[j] <= t < starts[j] + ACTS_GUIDE[j]["p"])
                for t in range(int(Tend))]
        kf = [t for t, p in enumerate(prof) if p > 5]
        ttl = ("Frühester Start (Konflikt!)" if sh == 0
               else f"Akt. 5 um {sh} verschoben" + (" — geglättet" if not kf else ""))
        fig = draw_resource(ACTS_GUIDE, DEMAND_GUIDE, cap=5, shifts=shifts,
                            figsize=(9.2, 4.2), title=ttl)
        return fig, kf

    _fig, _kf = _g32e()
    _stat = ("✅ kein Konflikt — Spitze geglättet" if not _kf
             else f"⚠️ Überlast an Tag(en) {_kf} (rot markiert)")
    mo.vstack([
        mo.md(r"""
        ## e) Kapazitätsglättung — wenn der Slack reicht

        Bisher startet jede Aktivität **so früh wie möglich**. Aber Teams sind
        **begrenzt**: laufen zu viele Aktivitäten gleichzeitig, entsteht eine
        **Spitze** über der Kapazität.

        **Idee — rein visuell:** Aktivitäten mit **Puffer (Slack)** lassen sich
        verschieben, um die Spitze zu glätten — **ohne die Projektdauer zu ändern**.
        Aktivitäten auf dem **kritischen Pfad** (im Plot **roter Rand**, unten) sind
        **tabu**.

        Bedarf $r=(1,3,2,2,3,1)$, **Teamkapazität = 5**. Nur die
        **Vertriebsschulung (Akt. 5)** mit Slack $TF_5=3$ darf verschoben werden.

        **e1)** An welchem Tag tritt der Engpass ($r>5$) auf? **e2)** Um wie viele Tage
        müssen Sie Akt. 5 verschieben, damit kein Tag mehr über 5 liegt? **e3)** Bleibt
        die Projektdauer bei 10? *(Slider zum Selbstkontrollieren.)*
        """),
        mo.hstack([g2_shift, mo.md(f"**Status:** {_stat}")],
                  justify="start", gap=2.0, align="center"),
        mo.as_html(_fig),
        mo.accordion({
            "💡 Lösung zu e)": mo.md(r"""
        **e1)** An **Tag 4** kollidieren Entwicklung (Akt. 2, kritisch, $r=3$) und
        Vertriebsschulung (Akt. 5, $r=3$): $3+3=6>5$.
        **e2)** Akt. 5 hat $TF_5=3$ — schon **1 Tag** Verschiebung genügt.
        **e3)** **Ja, die Projektdauer bleibt 10** — eine Verschiebung im Rahmen des
        Puffers verändert das Projektende nicht. Die kritische Akt. 2 (roter Rand)
        ist dagegen tabu.
        """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    g2_cap = mo.ui.slider(start=3, stop=6, step=1, value=4,
                          label="Gesenkte Teamkapazität", show_value=True)
    return (g2_cap,)


@app.cell(hide_code=True)
def _(ACTS_GUIDE, DEMAND_GUIDE, cpm_solve, draw_resource, g2_cap,
      mo, schedule_capacity):
    def _g32f():
        cap = int(g2_cap.value)
        T = cpm_solve(ACTS_GUIDE)["T"]
        starts, ms = schedule_capacity(ACTS_GUIDE, DEMAND_GUIDE, cap)
        ttl = (f"Kapazität {cap}: bestmögliche Glättung — Projektdauer {ms}"
               + (f"  (+{ms - T}!)" if ms > T else "  (unverändert)"))
        fig = draw_resource(ACTS_GUIDE, DEMAND_GUIDE, cap=cap, starts=starts,
                            figsize=(9.2, 4.2), title=ttl)
        return fig, ms, T

    _fig, _ms, _T = _g32f()
    _stat = (f"⚠️ Projektdauer **{_ms}** statt {_T} — Slack reicht **nicht**!"
             if _ms > _T else f"✅ Projektdauer bleibt {_T}")
    mo.vstack([
        mo.md(r"""
        ## f) Kapazitätsglättung — wenn der Slack **nicht** reicht

        Was, wenn das Team **kleiner** ist? Der Plan wird **bestmöglich geglättet**
        (Aktivitäten so früh wie möglich, ohne die Kapazität zu überschreiten). Bei zu
        kleiner Kapazität müssen Aktivitäten **über ihren Puffer hinaus** verschoben
        werden — dann verlängert sich das Projekt.

        **f1)** Wie lang wird das Projekt bei **Kapazität 4**? **f2)** Ab welcher
        Kapazität bleibt die Projektdauer bei 10? *(Kapazität-Slider zum
        Selbstkontrollieren.)*
        """),
        mo.hstack([g2_cap, mo.md(f"**Status:** {_stat}")],
                  justify="start", gap=2.0, align="center"),
        mo.as_html(_fig),
        mo.accordion({
            "💡 Lösung zu f)": mo.md(r"""
        **f1)** Bei **Kapazität 4** steigt die Projektdauer auf **12** ($+2$):
        Entwicklung (Akt. 2, $r=3$) belegt $[2,5)$ fast allein; Marketing/Schulung
        passen nicht mehr in ihren Puffer und müssen dahinter.
        **f2)** Ab **Kapazität 5** bleibt die Projektdauer bei **10** (der Puffer
        reicht, vgl. Teil e).

        **Pointe:** Pufferzeiten glätten „gratis", solange sie reichen. Reichen sie
        nicht, hilft nur **mehr Kapazität** oder ein **längeres Projekt**. Diese
        allgemeine Ressourcenplanung (RCPSP) ist NP-schwer → VL 10.
        """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    # Zustand der Sandbox: Knotenanzahl + gezeichnete Kanten (bleiben beim
    # Hinzufügen/Entfernen von Knoten erhalten).
    get_count, set_count = mo.state(5)
    get_links, set_links = mo.state([])
    return get_count, get_links, set_count, set_links


@app.cell(hide_code=True)
def _(FixedEdgeDraw, get_count, get_links, mo):
    # Feste Start-/Ende-Knoten + Aktivitäten. KEIN Preset-Pfad: leeres Netz,
    # vorhandene Kanten kommen aus dem Zustand (überleben Knoten-Änderungen).
    _n = get_count()
    _names = ["Start"] + [str(i) for i in range(1, _n + 1)] + ["Ende"]
    sb_edges = mo.ui.anywidget(
        FixedEdgeDraw(names=_names, links=get_links(), height=400, width=600))
    return (sb_edges,)


@app.cell(hide_code=True)
def _(get_count, mo, sb_edges, set_count, set_links):
    # +/- Knoten: Kanten erhalten. Hinzufügen = neuer Knoten ohne Kanten.
    # Entfernen = nur die Kanten des höchsten Knotens löschen.
    def _capture():
        ev = sb_edges.value
        return list(ev.get("links", [])) if isinstance(ev, dict) else []

    def _add(_):
        set_links(_capture())
        set_count(get_count() + 1)

    def _remove(_):
        n = get_count()
        if n <= 1:
            return
        dead = str(n)
        set_links([ln for ln in _capture()
                   if str(ln.get("source")) != dead
                   and str(ln.get("target")) != dead])
        set_count(n - 1)

    sb_add = mo.ui.button(label="➕ Knoten", on_change=_add)
    sb_rem = mo.ui.button(label="➖ Knoten", on_change=_remove)
    return sb_add, sb_rem


@app.cell(hide_code=True)
def _(get_count, mo):
    _names = [str(i) for i in range(1, get_count() + 1)]
    sb_dur = mo.ui.dictionary({
        n: mo.ui.slider(start=1, stop=8, step=1, value=2,
                        label=f"$p_{{{n}}}$", show_value=True)
        for n in _names})
    return (sb_dur,)


@app.cell(hide_code=True)
def _(get_count, mo):
    _names = [str(i) for i in range(1, get_count() + 1)]
    sb_dem = mo.ui.dictionary({
        n: mo.ui.slider(start=1, stop=6, step=1, value=2,
                        label=f"$r_{{{n}}}$", show_value=True)
        for n in _names})
    return (sb_dem,)


@app.cell(hide_code=True)
def _(get_count, mo):
    _opts = [str(i) for i in range(1, get_count() + 1)]
    sb_lag_from = mo.ui.dropdown(options=_opts, value=_opts[0], label="von")
    sb_lag_to = mo.ui.dropdown(options=_opts,
                               value=_opts[1] if len(_opts) > 1 else _opts[0],
                               label="nach")
    sb_lag_d = mo.ui.slider(start=0, stop=5, step=1, value=0,
                            label="$d$", show_value=True)
    return sb_lag_d, sb_lag_from, sb_lag_to


@app.cell(hide_code=True)
def _(PlaySlider, mo):
    sb_play = mo.ui.anywidget(
        PlaySlider(value=0, min_value=0, max_value=25, step=1,
                   interval_ms=750, loop=False, width=380))
    sb_cap = mo.ui.slider(start=2, stop=12, step=1, value=5,
                          label="Kapazität", show_value=True)
    return sb_cap, sb_play


@app.cell(hide_code=True)
def _(build_acts, cpm_solve, draw_anim, draw_resource, get_count, mo, sb_add,
      sb_cap, sb_dem, sb_dur, sb_edges, sb_lag_d, sb_lag_from, sb_lag_to,
      sb_play, sb_rem):
    def _sandbox():
        names = [str(i) for i in range(1, get_count() + 1)]
        ev = sb_edges.value
        links = ev.get("links", []) if isinstance(ev, dict) else []
        durs = {n: sb_dur.value[n] for n in names if n in sb_dur.value}
        acts = build_acts(names, links, durs)   # Start/Ende-Kanten werden ignoriert
        lag = {}
        if int(sb_lag_d.value) > 0 and sb_lag_from.value != sb_lag_to.value:
            lag = {(sb_lag_from.value, sb_lag_to.value): int(sb_lag_d.value)}
        try:
            r = cpm_solve(acts, lag)
        except Exception:
            return None, None, None
        pv = sb_play.value
        step = int(round(pv.get("value", 0) if isinstance(pv, dict) else pv))
        net = draw_anim(acts, step, lag=lag, figsize=(7.4, 5.2))
        demand = {n: int(sb_dem.value[n]) for n in names if n in sb_dem.value}
        res = draw_resource(acts, demand, cap=int(sb_cap.value), lag=lag,
                            figsize=(6.8, 4.2), title="Ressourcenplan")
        return r, net, res

    _r, _net, _res = _sandbox()
    _left = mo.vstack([
        mo.md("**1) Netz zeichnen**  \n*Pfeil: Vorgänger → Nachfolger. "
              "**Start** und **Ende** sind feste Knoten.*"),
        sb_edges,
        mo.hstack([
            mo.vstack([mo.md("**2) Dauern** $p$"),
                       mo.vstack(list(sb_dur.elements.values()), gap=0.15)]),
            mo.vstack([mo.md("**Bedarf** $r$"),
                       mo.vstack(list(sb_dem.elements.values()), gap=0.15)]),
        ], gap=0.8, align="start"),
    ])
    if _r is None:
        _mid = mo.md("⚠️ **Zyklus erkannt** — eine Aktivität darf nicht (indirekt) "
                     "ihr eigener Vorgänger sein. Entferne eine Kante.")
        _right = mo.md("")
    else:
        _mid = mo.vstack([mo.md(f"**3) Animation** · Projektdauer **{_r['T']}**"),
                          sb_play, mo.as_html(_net)])
        _right = mo.vstack([mo.md("**4) Ressourcen**"), mo.as_html(_res)])
    mo.vstack([
        mo.md(r"""
        ---
        # Teil 4 · Sandbox — eigenes Netz zeichnen

        **So geht's:** (1) Mit **➕/➖ Knoten** Aktivitäten anlegen — gezeichnete
        Kanten bleiben erhalten. (2) Im **Zeichenfeld** Knoten per Pfeil verbinden
        (*Vorgänger → Nachfolger*); **Start**/**Ende** sind feste Anker. (3) **Dauern**
        $p$ und **Bedarf** $r$ je Knoten einstellen. (4) Optional einen
        **Mindestabstand** $d$ auf einer Kante setzen. (5) **Play** drücken:
        Vorwärts-/Rückwärtsrechnung läuft, kritischer Pfad und **Ressourcenplan**
        aktualisieren sich live.
        """),
        mo.hstack([sb_add, sb_rem, mo.md(f"**{get_count()} Aktivitäten**  ·"),
                   sb_cap, mo.md("· **Mindestabstand:**"),
                   sb_lag_from, sb_lag_to, sb_lag_d],
                  justify="start", gap=0.8, align="center"),
        mo.hstack([_left, _mid, _right], widths=[0.42, 0.34, 0.24], gap=1.0,
                  align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Teil 5 · Spickzettel — CPM in 4 Schritten

    | Schritt | So gehen Sie vor |
    |---|---|
    | **1. Netz** | Aktivität = Knoten ($p_j$), Vorgänger = Pfeil, Dummy Start/Ende ($p=0$) |
    | **2. Vorwärts** | $FSZ_j=\max_i FEZ_i$ (Vorgänger), $FEZ_j=FSZ_j+p_j$ · Projektdauer = größtes $FEZ$ |
    | **3. Rückwärts** | $SEZ_j=\min_k SSZ_k$ (Nachfolger), $SSZ_j=SEZ_j-p_j$ · Ende = Projektdauer |
    | **4. Puffer** | $TF_j=SSZ_j-FSZ_j=SEZ_j-FEZ_j$ · $TF=0$ ⇒ **kritisch** |

    **Klausur-Tipps**

    - **$\max$ vorwärts, $\min$ rückwärts** — häufigste Verwechslung.
    - **Mindestabstand** $d$: einfach $+d$ in der Vorwärts-/Rückwärtsregel der Kante.
    - **Kontrolle:** kleinstes $SSZ$ über alle Aktivitäten muss $0$ sein.
    - **Kritischer Pfad** = lückenlose $TF{=}0$-Kette; seine Länge ist die Projektdauer.
    - **Ressourcen:** nur **unkritische** Aktivitäten (mit Puffer) lassen sich zum
      Glätten verschieben.
    """)
    return


@app.cell(hide_code=True)
def _(ACTS_WEB, mo):
    _liste = mo.md(r"""
    **Vorgangsliste (Website-Relaunch)**

    | $j$ | Aktivität | $p_j$ (Tage) | Vorgänger | Bedarf $r_j$ |
    |:--:|:--|:--:|:--:|:--:|
    | 1 | Konzept | 3 | – | 2 |
    | 2 | Inhalte erstellen | 4 | 1 | 3 |
    | 3 | Design | 2 | 1 | 2 |
    | 4 | Programmierung | 5 | 3 | 3 |
    | 5 | Inhalte einpflegen | 2 | 2, 4 | 2 |
    | 6 | Testing | 3 | 4 | 2 |
    | 7 | Go-Live | 1 | 5, 6 | 1 |
    """)
    mo.vstack([
        mo.md(r"""
        ---
        # Teil 6 · Selbständige Aufgaben

        ## Aufgabe 3 — Website-Relaunch komplett rechnen

        Eine Agentur plant den Relaunch einer Website. Führen Sie CPM vollständig
        durch — *erst von Hand, dann die animierte Lösung aufklappen.*
        """),
        _liste,
        mo.md(r"""
        **a)** Netz zeichnen (Dummy Start/Ende).
        **b)** Vorwärts-/Rückwärtsrechnung: $FSZ, FEZ, SSZ, SEZ$ für alle $j$.
        **c)** $TF_j$, kritischer Pfad und Projektdauer.
        **d)** Welche Aktivität hat den **größten Puffer** — was bedeutet das praktisch?
        """),
    ])
    return


@app.cell(hide_code=True)
def _(ACTS_WEB, PlaySlider, mo):
    _total = 3 * len(ACTS_WEB) + 1
    a3_play = mo.ui.anywidget(
        PlaySlider(value=0, min_value=0, max_value=_total, step=1,
                   interval_ms=700, loop=False, width=440))
    return (a3_play,)


@app.cell(hide_code=True)
def _(ACTS_WEB, a3_play, cpm_solve, cpm_table, draw_anim, mo):
    def _a3():
        r = cpm_solve(ACTS_WEB)
        cp = " → ".join(str(j) for j in r["order"] if j in r["crit"])
        jmax = max(ACTS_WEB, key=lambda j: r["TF"][j])
        pv = a3_play.value
        step = int(round(pv.get("value", 0) if isinstance(pv, dict) else pv))
        return r, cp, jmax, draw_anim(ACTS_WEB, step, figsize=(17, 8.5))

    _r, _cp, _jmax, _fig = _a3()
    mo.accordion({
        "🎬 Animierte Lösung zu Aufgabe 3 (Play / Slider)": mo.vstack([
            a3_play,
            mo.as_html(_fig),
            mo.md("**CPM-Tabelle:**\n\n" + cpm_table(ACTS_WEB)),
            mo.md(rf"""
        **c)** Kritischer Pfad **{_cp}** ⇒ **Projektdauer = {_r['T']} Tage**.
        **d)** Größter Puffer: **Aktivität {_jmax} ({ACTS_WEB[_jmax]['name']})** mit
        $TF_{{{_jmax}}}={_r['TF'][_jmax]}$ — um so viele Tage verschiebbar, z. B. zur
        Auslastungsglättung. Die Programmierung (Akt. 4) ist kritisch: jede
        Verzögerung dort verschiebt das Go-Live direkt.
        """),
        ])
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Aufgabe 3 (Forts.) — Verzögerung & Puffer

    *Gleiches Projekt, weitere Teilaufgaben.* (Website-Relaunch, Projektdauer 14,
    kritischer Pfad $1\to3\to4\to6\to7$.) Stellen Sie unten die **neue Dauer** ein
    und beobachten Sie die Projektdauer.

    **e)** **Programmierung (Akt. 4)** dauert 2 Tage länger ($p_4: 5\to7$). Neue
    Projektdauer? Kritischer Pfad gleich?
    **f)** Stattdessen **Inhalte erstellen (Akt. 2)** 2 Tage länger ($p_2: 4\to6$).
    Was passiert — und warum anders als e)?
    **g)** Um wie viele Tage darf sich Akt. 2 *maximal* verzögern, ohne das
    Projektende zu verschieben?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    a4_p4 = mo.ui.slider(start=5, stop=9, step=1, value=5,
                         label="$p_4$ (Programmierung, Basis 5)", show_value=True)
    a4_p2 = mo.ui.slider(start=4, stop=10, step=1, value=4,
                         label="$p_2$ (Inhalte, Basis 4)", show_value=True)
    return a4_p2, a4_p4


@app.cell(hide_code=True)
def _(ACTS_WEB, a4_p2, a4_p4, cpm_solve, draw_network, mo):
    def _a4():
        acts = {k: {"name": ACTS_WEB[k]["name"], "pred": list(ACTS_WEB[k]["pred"]),
                    "p": ACTS_WEB[k]["p"]} for k in ACTS_WEB}
        acts[4]["p"] = int(a4_p4.value)
        acts[2]["p"] = int(a4_p2.value)
        r = cpm_solve(acts)
        return acts, r

    _acts, _r = _a4()
    _delta = _r["T"] - 14
    mo.vstack([
        mo.hstack([a4_p4, a4_p2,
                   mo.md(f"**Projektdauer:** {_r['T']} ({'+' if _delta>=0 else ''}{_delta})")],
                  justify="start", gap=1.6, align="center"),
        mo.as_html(draw_network(_acts, figsize=(9.6, 5.0),
                                title=f"Website-Relaunch — Projektdauer = {_r['T']}")),
        mo.accordion({
            "💡 Lösung zu e)–g)": mo.md(r"""
        **e)** Akt. 4 ist **kritisch** ($TF_4=0$): $+2$ schlägt 1:1 durch ⇒
        **16 Tage**. Kritischer Pfad bleibt $1\to3\to4\to6\to7$.
        **f)** Akt. 2 hat **Puffer** ($TF_2=4$): $+2 \le 4$ liegt im Puffer ⇒
        Projektdauer bleibt **14**. (Probieren Sie $p_2=9$: ab $+4$ wird Akt. 2
        kritisch und das Projekt länger.)
        **g)** Genau **$TF_2=4$ Tage**.
        """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Aufgabe 3 (Forts.) — Kapazitätsglättung

    *Gleiches Projekt.* Jede Aktivität braucht Personal — Bedarf
    $r=(2,3,2,3,2,2,1)$ für Akt. 1–7. Der Plan unten wird **bestmöglich geglättet**
    (Aktivitäten so früh wie möglich, ohne die Kapazität zu überschreiten); kritische
    Aktivitäten (**roter Rand**) haben keinen Spielraum.

    **h)** Bestimmen Sie den **Spitzenbedarf** bei frühestem Start. Ab welcher
    Teamkapazität verlängert sich das Projekt über 14 Tage — und warum?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    a5_cap = mo.ui.slider(start=4, stop=7, step=1, value=6,
                          label="Teamkapazität", show_value=True)
    return (a5_cap,)


@app.cell(hide_code=True)
def _(ACTS_WEB, DEMAND_WEB, a5_cap, cpm_solve, draw_resource, mo,
      schedule_capacity):
    def _a5():
        cap = int(a5_cap.value)
        T = cpm_solve(ACTS_WEB)["T"]
        starts, ms = schedule_capacity(ACTS_WEB, DEMAND_WEB, cap)
        ttl = (f"Kapazität {cap}: bestmögliche Glättung — Projektdauer {ms}"
               + (f"  (+{ms - T}!)" if ms > T else "  (unverändert)"))
        fig = draw_resource(ACTS_WEB, DEMAND_WEB, cap=cap, starts=starts,
                            figsize=(10.5, 4.4), title=ttl)
        return fig, ms, T

    _fig, _ms, _T = _a5()
    _stat = (f"⚠️ Projektdauer **{_ms}** statt {_T} — Slack reicht nicht!"
             if _ms > _T else f"✅ Projektdauer bleibt {_T}")
    mo.vstack([
        mo.hstack([a5_cap, mo.md(f"**Status:** {_stat}")],
                  justify="start", gap=2.0, align="center"),
        mo.as_html(_fig),
        mo.accordion({
            "💡 Lösung zu h)": mo.md(r"""
        Der **Spitzenbedarf** bei frühestem Start ist **6** (Inhalte + Programmierung
        überlappen an den Tagen 5–6). Daher:
        - **Kapazität 6 (und mehr):** kein Problem — Projektdauer bleibt **14**.
        - **Ab Kapazität 5** reicht der Puffer nicht mehr; selbst optimal geglättet
          steigt die Projektdauer (Solver: **17**).

        **Pointe:** Solange die Kapazität ≥ Spitzenbedarf ist, glätten Pufferzeiten
        „gratis". Darunter erzwingt der Engpass eine **Verlängerung** (RCPSP,
        NP-schwer → VL 10).
        """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(ACTS_ALT, mo):
    _liste = mo.md(r"""
    **Vorgangsliste — „Messestand" (alles in einem)**

    | $j$ | Aktivität | $p_j$ | Vorgänger | Bedarf $r_j$ |
    |:--:|:--|:--:|:--:|:--:|
    | 1 | Briefing | 2 | – | 2 |
    | 2 | Standdesign | 4 | 1 | 2 |
    | 3 | Catering | 2 | 1 | 2 |
    | 4 | Standbau | 5 | 2 | 3 |
    | 5 | Personal | 3 | 3 | 2 |
    | 6 | Generalprobe | 2 | 4, 5 | 2 |
    | 7 | Messe | 1 | 6 | 1 |
    """)
    mo.vstack([
        mo.md(r"""
        ---
        ## Aufgabe 4 — Alternativ-Aufgabe „Messestand" *(alle Bestandteile)*

        Eine umfassende Aufgabe zum selbst Durchrechnen — **CPM + Mindestabstand +
        Kapazität** an einem neuen Projekt.
        """),
        _liste,
        mo.md(r"""
        **a)** Netz, Vorwärts-/Rückwärtsrechnung, **kritischer Pfad** und Projektdauer.
        **b)** Zwischen **Catering (3)** und **Personal (5)** soll ein
        **Mindestabstand** $d$ liegen — ab welchem $d$ verlängert sich das Projekt?
        **c)** **Kapazität:** ab welcher Teamgröße reicht der Slack nicht mehr?
        """),
    ])
    return


@app.cell(hide_code=True)
def _(ACTS_ALT, PlaySlider, mo):
    _total = 3 * len(ACTS_ALT) + 1
    alt_play = mo.ui.anywidget(
        PlaySlider(value=0, min_value=0, max_value=_total, step=1,
                   interval_ms=650, loop=False, width=440))
    return (alt_play,)


@app.cell(hide_code=True)
def _(ACTS_ALT, alt_play, cpm_solve, cpm_table, draw_anim, mo):
    def _alt():
        r = cpm_solve(ACTS_ALT)
        cp = " → ".join(str(j) for j in r["order"] if j in r["crit"])
        pv = alt_play.value
        step = int(round(pv.get("value", 0) if isinstance(pv, dict) else pv))
        return r, cp, draw_anim(ACTS_ALT, step, figsize=(16, 8))

    _r, _cp, _fig = _alt()
    mo.accordion({
        "🎬 Lösung a) — animierte CPM-Rechnung": mo.vstack([
            alt_play,
            mo.as_html(_fig),
            mo.md("**CPM-Tabelle:**\n\n" + cpm_table(ACTS_ALT)),
            mo.md(rf"""
        Kritischer Pfad **{_cp}**, **Projektdauer = {_r['T']}**. Größter Puffer:
        Catering (3) und Personal (5) mit je $TF=4$.
        """),
        ])
    })
    return


@app.cell(hide_code=True)
def _(mo):
    alt_lag = mo.ui.slider(start=0, stop=6, step=1, value=0,
                           label="Mindestabstand $d$ auf $3\\to5$", show_value=True)
    alt_cap = mo.ui.slider(start=3, stop=7, step=1, value=5,
                           label="Teamkapazität", show_value=True)
    return alt_cap, alt_lag


@app.cell(hide_code=True)
def _(ACTS_ALT, DEMAND_ALT, alt_cap, alt_lag, cpm_solve, draw_network,
      draw_resource, mo, schedule_capacity):
    def _altbc():
        d = int(alt_lag.value)
        lag = {(3, 5): d} if d else None
        rd = cpm_solve(ACTS_ALT, lag)
        cp = " → ".join(str(j) for j in rd["order"] if j in rd["crit"])
        fig_net = draw_network(ACTS_ALT, lag, figsize=(11, 5.4),
                               title=f"b) Mindestabstand d={d} auf 3→5 — Dauer {rd['T']}")
        cap = int(alt_cap.value)
        T = cpm_solve(ACTS_ALT)["T"]
        st, ms = schedule_capacity(ACTS_ALT, DEMAND_ALT, cap)
        fig_res = draw_resource(ACTS_ALT, DEMAND_ALT, cap=cap, starts=st,
                                figsize=(10.5, 4.4),
                                title=f"c) Kapazität {cap} — Projektdauer {ms}"
                                      + (f" (+{ms - T}!)" if ms > T else ""))
        return rd, cp, fig_net, ms, T, fig_res

    _rd, _cp, _fn, _ms, _T, _fr = _altbc()
    mo.vstack([
        mo.hstack([alt_lag, mo.md(f"Dauer **{_rd['T']}**, kritisch {_cp}"),
                   alt_cap, mo.md(f"Kapazität-Dauer **{_ms}**")],
                  justify="start", gap=1.4, align="center"),
        mo.hstack([mo.as_html(_fn), mo.as_html(_fr)], widths=[0.5, 0.5],
                  gap=0.8, align="start"),
        mo.accordion({
            "💡 Lösung b) + c)": mo.md(r"""
        **b)** Die Kante $3\to5$ hat **Puffer 4**. Für $d\le 3$ ändert sich nichts
        ($TF$ schluckt den Abstand); bei $d=4$ wird der Pfad $1\to3\to5\to6\to7$
        **mit-kritisch**, ab $d=5$ **verlängert** sich das Projekt (Dauer 15, 16, …)
        und der kritische Pfad wechselt dorthin.
        **c)** Der Spitzenbedarf ist **5**. Bei Kapazität $\ge 5$ bleibt die Dauer
        **14**; bei Kapazität **4** reicht der Slack nicht — selbst optimal geglättet
        steigt die Projektdauer (auf **17**).
        """)
        }),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Teil 7 · Besprechung & Ausblick

    - **Ein Schema, drei Antworten:** Vorwärts-/Rückwärtsrechnung liefert
      **Projektdauer**, **Pufferzeiten** und **kritischen Pfad** in einem Durchlauf.
    - **Kritisch = bindend:** kritische Aktivitäten haben keinen Spielraum —
      Verzögerungen dort kosten Projektzeit.
    - **Erweiterungen** (Mindestabstand, Ressourcen) ändern das Schema kaum: ein
      Term $+d$ bzw. ein Blick auf den Ressourcenplan.
    - **Puffer sind ein Hebel:** Kapazitätsspitzen lassen sich durch Verschieben
      unkritischer Aktivitäten glätten.
    - **Ausblick VL 10:** Crashing (Zeit-Kosten-Tradeoff) und Ressourcenplanung
      (RCPSP) als (ganzzahlige) Optimierung.
    """)
    return


if __name__ == "__main__":
    app.run()
