# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pulp",
#     "pandas",
#     "scipy",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(
    width="full",
    app_title="PuE Übung 6: Sensitivitätsanalyse",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pulp as pl
    return mo, np, pl


@app.cell(hide_code=True)
def solver(pl):
    # ─────────────────────────────────────────────────────────────────────
    # Hintergrund-Solver — löst PuLP-Modelle via scipy/HiGHS.
    # Funktioniert auch im Browser (Pyodide/WASM), wo CBC nicht verfügbar ist.
    # Schreibt .varValue, .slack, .pi & prob.status wie ein echter Solve zurück.
    # Studierende sehen davon nichts — PuLP läuft komplett im Hintergrund.
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
        con_map = {}
        for name, con in prob.constraints.items():
            row = _np.zeros(n)
            for v, coef in con.items():
                row[idx[v.name]] = coef
            rhs = -con.constant
            if con.sense == pl.LpConstraintLE:
                A_ub_rows.append(row); b_ub_vals.append(rhs)
                con_map[name] = ("ub", len(A_ub_rows) - 1, +1)
            elif con.sense == pl.LpConstraintGE:
                A_ub_rows.append(-row); b_ub_vals.append(-rhs)
                con_map[name] = ("ub", len(A_ub_rows) - 1, -1)
            else:
                A_eq_rows.append(row); b_eq_vals.append(rhs)
                con_map[name] = ("eq", len(A_eq_rows) - 1, +1)

        A_ub = _np.array(A_ub_rows) if A_ub_rows else None
        b_ub = _np.array(b_ub_vals) if b_ub_vals else None
        A_eq = _np.array(A_eq_rows) if A_eq_rows else None
        b_eq = _np.array(b_eq_vals) if b_eq_vals else None

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method="highs")

        if result.success:
            prob.status = 1
        elif result.status == 2:
            prob.status = -1
        elif result.status == 3:
            prob.status = -2
        else:
            prob.status = 0

        if result.success:
            for i, v in enumerate(variables):
                v.varValue = float(result.x[i])

        ineq_marg = (list(result.ineqlin.marginals)
                     if (result.success and getattr(result, "ineqlin", None) is not None)
                     else [])
        eq_marg = (list(result.eqlin.marginals)
                   if (result.success and getattr(result, "eqlin", None) is not None)
                   else [])
        obj_sign = -1.0 if is_max else 1.0

        for name, con in prob.constraints.items():
            if result.success:
                lhs = sum(coef * (v.varValue or 0) for v, coef in con.items())
                rhs = -con.constant
                if con.sense == pl.LpConstraintGE:
                    con.slack = lhs - rhs
                else:
                    con.slack = rhs - lhs
            kind, ridx, sign = con_map[name]
            if kind == "ub" and ridx < len(ineq_marg):
                con.pi = obj_sign * sign * ineq_marg[ridx]
            elif kind == "eq" and ridx < len(eq_marg):
                con.pi = obj_sign * eq_marg[ridx]
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
    solver_ready = True  # marimo-Sentinel: erzwingt Reihenfolge solver → Demos
    return


@app.cell(hide_code=True)
def geometry():
    import math as _math

    def line_box(a, b, c, box):
        """Endpunkte der Geraden a·x + b·y = c innerhalb der Box (für Plot)."""
        xmin, xmax, ymin, ymax = box
        pts = []
        if abs(b) > 1e-12:
            for xv in (xmin, xmax):
                yv = (c - a * xv) / b
                if ymin - 1e-7 <= yv <= ymax + 1e-7:
                    pts.append((xv, yv))
        if abs(a) > 1e-12:
            for yv in (ymin, ymax):
                xv = (c - b * yv) / a
                if xmin - 1e-7 <= xv <= xmax + 1e-7:
                    pts.append((xv, yv))
        out = []
        for p in pts:
            if not any(abs(p[0] - q[0]) < 1e-6 and abs(p[1] - q[1]) < 1e-6 for q in out):
                out.append(p)
        return out[:2]

    def region_verts(cons, box):
        """Geordnete Eckpunkte des zulässigen Bereichs.

        cons : Liste von (a, b, c, sense), sense ∈ {"<=", ">="}, Bedeutung a·x+b·y ⋛ c.
        box  : (xmin, xmax, ymin, ymax) — begrenzt auch unbeschränkte Bereiche.
        """
        xmin, xmax, ymin, ymax = box
        lines = [(a, b, c) for (a, b, c, s) in cons]
        lines += [(1.0, 0.0, xmin), (1.0, 0.0, xmax),
                  (0.0, 1.0, ymin), (0.0, 1.0, ymax)]
        pts = []
        n = len(lines)
        for i in range(n):
            for j in range(i + 1, n):
                a1, b1, c1 = lines[i]
                a2, b2, c2 = lines[j]
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-12:
                    continue
                x = (c1 * b2 - c2 * b1) / det
                y = (a1 * c2 - a2 * c1) / det
                pts.append((x, y))

        def ok(p):
            x, y = p
            if x < xmin - 1e-6 or x > xmax + 1e-6 or y < ymin - 1e-6 or y > ymax + 1e-6:
                return False
            for (a, b, c, s) in cons:
                v = a * x + b * y
                if s == "<=" and v > c + 1e-6:
                    return False
                if s == ">=" and v < c - 1e-6:
                    return False
            return True

        feas = [p for p in pts if ok(p)]
        uniq = []
        for p in feas:
            if not any(abs(p[0] - q[0]) < 1e-5 and abs(p[1] - q[1]) < 1e-5 for q in uniq):
                uniq.append(p)
        if len(uniq) < 3:
            return uniq
        cx = sum(p[0] for p in uniq) / len(uniq)
        cy = sum(p[1] for p in uniq) / len(uniq)
        uniq.sort(key=lambda p: _math.atan2(p[1] - cy, p[0] - cx))
        return uniq
    return line_box, region_verts


@app.cell(hide_code=True)
def colors():
    IMSBlue = "#023B88"
    IMSOrange = "#D87237"
    rot = "#C0392B"
    gruen = "#27AE60"
    lila = "#8E44AD"
    farbe_gray = "#7F8C8D"
    farbe_feasible = "#9FBEE6"
    return IMSBlue, IMSOrange, farbe_feasible, farbe_gray, gruen, lila, rot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Übung 6 · Sensitivitätsanalyse

    **Planung und Entscheidung — SS 2026 · Begleitnotebook zu VL 07**

    Bisher haben wir LPs *gelöst*. Heute fragen wir: **Was sagt uns die Lösung
    über die Realität?** Wie viel ist eine Ressource wert? Wie robust ist unser
    Plan, wenn sich Parameter ändern?

    > In diesem Notebook **schreibt ihr keinen Code** — ihr verstellt Regler,
    > beobachtet, wie sich Optimum, Schattenpreise und der zulässige Bereich
    > verändern, und lest die Ergebnisse wie einen Management-Report.

    Ablauf:

    1. **Wiederholung** — worum geht es, und die zwei Wege (manuell / analytisch).
    2. **Geführte Übung** — Brauerei Würzburg: Parameter selbst verstellen und die
       Wirkung auf Lösung, Schattenpreise und Gewinn beobachten.
    3. *(folgt)* Modifizieren, selbständige Aufgabe, Besprechung.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Teil 1 · Wiederholung

    ## 1.1 · Worum geht es?

    Bisher haben wir LPs **gelöst** und die Koeffizienten als **sicher** angenommen.
    Real schwanken aber Kapazitäten, Preise und Nachfrage. Die Sensitivitätsanalyse
    fragt deshalb: **Wie reagieren die Optimallösung $x^*$ und der Gewinn $z^*$ auf
    solche Änderungen — und ab wann kippt die Lösung in einen anderen Eckpunkt?**

    Dabei unterscheiden wir, **was** sich ändert:

    - die **rechte Seite** $b_j$ — die verfügbare Menge einer Ressource (Kapazität)
    - ein **Zielkoeffizient** $c_i$ — der Deckungsbeitrag bzw. Preis eines Produkts
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _intro = mo.md(r"""
    ---

    ## 1.2 · Zwei Wege zur Antwort

    Auf die Frage „wie wirkt eine Parameteränderung?" gibt es **zwei Wege** — sie
    führen zum selben Ergebnis und geben dieser Übung ihre Struktur:
    """)

    _links = mo.md(r"""
    ### Teil 1 · Manuell
    *Parameter durchspielen*

    - einen Parameter über einen Bereich durchspielen, **jedes Mal neu lösen** und
      das Ergebnis auftragen
    - **$b_j$ erhöhen** → $z^*$ verläuft **stückweise linear**: gleichmäßige Steigung
      bis zu einem **Knick**, dort wird eine **andere Ressource zum Engpass**
    - **$c_i$ verändern** → ebenfalls **stückweise linear**; am **Knick** wechselt
      der **optimale Produktmix**

    Universell für jedes Modell — aber viele Solver-Läufe, nur einzelne Stützstellen,
    und die Wechselpunkte sieht man der Tabelle nicht an.
    """)

    _rechts = mo.md(r"""
    ### Teil 2 · Analytisch
    *aus dem Tableau ablesen*

    - direkt aus dem **finalen Simplex-Tableau** — ohne erneut zu lösen
    - **Schattenpreis**: was eine zusätzliche Einheit einer **knappen** Ressource an
      Gewinn bringt (eine nicht knappe Ressource → Schattenpreis **null**)
    - **Reduzierte Kosten**: ob sich ein bisher **nicht produziertes Produkt**
      überhaupt lohnt

    Exakt und sofort — aber nur **lokal** gültig, solange dieselbe Ecke optimal
    bleibt (Gültigkeitsbereich).
    """)

    _bridge = mo.md(r"""
    **Die Brücke zwischen beiden:** Die **Steigung** der manuellen $z^*$-Kurve
    *ist* der **Schattenpreis**, und ihr **Knick** markiert das **Ende des
    Gültigkeitsbereichs**. Der analytische Weg liefert beides direkt — und spart
    die ganze Schleife.
    """)

    mo.vstack([
        _intro,
        mo.hstack([_links, _rechts], widths="equal", gap=2, align="start"),
        _bridge,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Teil 2 · Geführte Übung — Brauerei Würzburg

    Die **Brauerei Würzburg** plant ihre Wochenproduktion zweier Sorten:
    **Helles** ($x_H$) und **Pils** ($x_P$), jeweils in Hektolitern (hl).
    Deckungsbeitrag: **4 €/hl** Helles, **6 €/hl** Pils. Drei Ressourcen begrenzen
    die Produktion:

    $$
    \begin{aligned}
    \max\ z = \;& c_H\,x_H + c_P\,x_P \\
    \text{s.t.}\;& 1\,x_H + 2\,x_P \leq b_M && \text{(Malz, t)} \\
    & 3\,x_H + 2\,x_P \leq b_G && \text{(Gärtanks, Tankstunden)} \\
    & 1\,x_H + 1\,x_P \leq b_A && \text{(Abfüllung, hl)} \\
    & x_H,\ x_P \geq 0
    \end{aligned}
    $$

    Im Ausgangsfall ($c_H=4,\ c_P=6,\ b_M=16,\ b_G=24,\ b_A=12$) ist das Optimum
    $x_H^*=4,\ x_P^*=6$ mit $z^*=52$ € — **Malz und Gärtanks binden, die Abfüllung
    hat noch Reserve.**

    ### Aufgabe 1 — manuelle Sensitivität

    Verstellt die fünf Regler und **lest aus den drei Panels ab** (kein Rechnen):

    - **Links** zulässiger Bereich, Iso-Gewinn-Gerade, Optimum.
    - **Mitte** Solver-Report: Mengen, Schlupf, Schattenpreis je Ressource.
    - **Rechts** für jeden Parameter die Kurve $z^*(\cdot)$; der orange Punkt ist euer
      aktueller Wert. **$b$-Kurven:** Steigung $=$ Schattenpreis. **$c$-Kurven:**
      Steigung $=$ optimale Produktmenge.

    **1a · Engpass & Schattenpreise** *(alle Regler auf Standard).*
    Welche zwei Ressourcen **binden**, welche hat **Reserve** — und wie viel? Lest die
    drei Schattenpreise ab und formuliert für Malz einen Satz („Eine Tonne Malz mehr
    bringt … €"). Warum ist die Kurve $z^*(b_A)$ am aktuellen Punkt **flach**?

    **1b · Zulässige Änderung von $b_G$ (Gärtanks).**
    Erhöht und senkt $b_G$. Zwischen welchen Werten bleibt die **Steigung** der
    $z^*(b_G)$-Kurve konstant (gleicher Schattenpreis)? Wie groß ist damit die
    **zulässige Änderung nach oben und nach unten**? Prüft die Faustregel: um wie viel
    steigt $z^*$, wenn $b_G$ von 24 auf 30 wächst?

    **1c · Was passiert am Knick? (graphisch)**
    Schiebt $b_G$ **über** den oberen Knick hinaus. Welche Restriktionslinie **löst
    sich links vom Optimum** (wird locker), und welche zwei Restriktionen bilden danach
    die optimale Ecke? Was wird aus dem **Schattenpreis von Gärtanks** — begründet mit
    dem komplementären Schlupf.

    **1d · Zielkoeffizient: wann kippt der Mix?**
    Setzt die $b$ zurück und verändert $c_P$. Ab welchem Wert **verschwindet Pils** aus
    dem Optimum, ab welchem wird **nur noch Pils** gebraut (links: das Optimum springt
    in eine andere Ecke)? Was bedeutet die **Steigung** von $z^*(c_P)$ im mittleren
    Stück?
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    # Brauerei Würzburg mit 3 Restriktionen — löst im Hintergrund (scipy/HiGHS).
    # ASCII-Restriktionsnamen für PuLP, deutsche Anzeigenamen separat.
    def solve_brauerei(cH=4.0, cP=6.0, bM=16.0, bG=24.0, bA=12.0):
        specs = [("Malz", "Malz", 1, 2, bM),
                 ("Gärtanks", "Gaertank", 3, 2, bG),
                 ("Abfüllung", "Abfuellung", 1, 1, bA)]
        m = pl.LpProblem("BrauereiWuerzburg", pl.LpMaximize)
        xH = pl.LpVariable("Helles", lowBound=0)
        xP = pl.LpVariable("Pils", lowBound=0)
        m += cH * xH + cP * xP, "Deckungsbeitrag"
        for _disp, pname, a1, a2, rhs in specs:
            m += a1 * xH + a2 * xP <= rhs, pname
        m.solve()
        xHv = pl.value(xH) or 0.0
        xPv = pl.value(xP) or 0.0
        out = {"status": pl.LpStatus[m.status], "xH": xHv, "xP": xPv,
               "z": pl.value(m.objective) or 0.0, "cH": cH, "cP": cP, "cons": {}}
        for disp, pname, a1, a2, rhs in specs:
            con = m.constraints[pname]
            out["cons"][disp] = {"rhs": rhs, "lhs": a1 * xHv + a2 * xPv,
                                 "slack": con.slack, "pi": con.pi, "a": (a1, a2)}
        return out
    return (solve_brauerei,)


@app.cell(hide_code=True)
def _(mo):
    gf_reset = mo.ui.button(label="↺ Zurücksetzen", kind="neutral")
    return (gf_reset,)


@app.cell(hide_code=True)
def _(gf_reset, mo):
    _ = gf_reset.value  # Klick → Defaults
    gf_cH = mo.ui.slider(start=0, stop=12, step=0.5, value=4,
                         label="$c_H$ DB Helles (€/hl)", show_value=True)
    gf_cP = mo.ui.slider(start=0, stop=12, step=0.5, value=6,
                         label="$c_P$ DB Pils (€/hl)", show_value=True)
    gf_bM = mo.ui.slider(start=4, stop=28, step=1, value=16,
                         label="$b_M$ Malz (t)", show_value=True)
    gf_bG = mo.ui.slider(start=4, stop=36, step=1, value=24,
                         label="$b_G$ Gärtanks (Tankstd.)", show_value=True)
    gf_bA = mo.ui.slider(start=4, stop=20, step=1, value=12,
                         label="$b_A$ Abfüllung (hl)", show_value=True)
    return gf_bA, gf_bG, gf_bM, gf_cH, gf_cP


@app.cell(hide_code=True)
def _(
    IMSBlue,
    IMSOrange,
    farbe_feasible,
    gf_bA,
    gf_bG,
    gf_bM,
    gf_cH,
    gf_cP,
    gf_reset,
    gruen,
    line_box,
    lila,
    mo,
    np,
    region_verts,
    solve_brauerei,
):
    def _render_guided():
        import matplotlib.pyplot as plt
        plt.close("all")

        cH, cP = float(gf_cH.value), float(gf_cP.value)
        bM, bG, bA = float(gf_bM.value), float(gf_bG.value), float(gf_bA.value)
        r = solve_brauerei(cH, cP, bM, bG, bA)
        xH, xP, z = r["xH"], r["xP"], r["z"]
        meta = {"Malz": IMSBlue, "Gärtanks": gruen, "Abfüllung": lila}
        cons = [(1, 2, bM, "<="), (3, 2, bG, "<="), (1, 1, bA, "<=")]

        # ── LINKS: zulässiger Bereich ──────────────────────────────────────
        poly = region_verts(cons, (0.0, 300.0, 0.0, 300.0))
        vx = max((p[0] for p in poly), default=8.0)
        vy = max((p[1] for p in poly), default=8.0)
        xmax = max(6.0, vx + 2.0); ymax = max(6.0, vy + 2.0)
        box = (-0.5, xmax, -0.5, ymax); rbox = (0.0, xmax, 0.0, ymax)

        figL, ax = plt.subplots(figsize=(6.8, 6.4))
        ax.set_axisbelow(True); ax.grid(True, alpha=0.3, ls="--")
        ax.set_xlim(*box[:2]); ax.set_ylim(*box[2:])
        ax.set_xlabel(r"$x_H$ — Helles (hl)"); ax.set_ylabel(r"$x_P$ — Pils (hl)")
        ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
        ax.set_title("Zulässiger Bereich & Optimum")
        verts = region_verts(cons, rbox)
        if len(verts) >= 3:
            ax.fill([p[0] for p in verts], [p[1] for p in verts],
                    color=farbe_feasible, alpha=0.55, zorder=1)
        for nm, (a1, a2, rhs, _s) in zip(["Malz", "Gärtanks", "Abfüllung"], cons):
            binds = r["cons"][nm]["slack"] < 1e-6
            col = meta[nm]
            p = line_box(a1, a2, rhs, box)
            if len(p) == 2:
                ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=col,
                        lw=2.6 if binds else 1.8, ls="-" if binds else "--",
                        label=f"{nm} ({'bindet' if binds else 'locker'})", zorder=3)
        for v in verts:
            ax.scatter(v[0], v[1], s=28, color="black", zorder=5)
        if cP > 1e-9:
            iso = line_box(cH, cP, cH * xH + cP * xP, box)
            if len(iso) == 2:
                ax.plot([iso[0][0], iso[1][0]], [iso[0][1], iso[1][1]],
                        color=IMSOrange, ls=":", lw=2.0, zorder=4, label="Iso-Gewinn")
        ax.scatter([xH], [xP], s=300, marker="*", color=IMSOrange,
                   edgecolor="black", lw=1.0, zorder=9)
        ax.annotate(rf"$x^*=({xH:.0f},{xP:.0f})$", (xH, xP),
                    textcoords="offset points", xytext=(8, -18), fontsize=10,
                    color=IMSOrange, fontweight="bold", zorder=10)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
        plt.tight_layout()

        # ── MITTE: Solver-Report ───────────────────────────────────────────
        rows = ""
        for nm in ["Malz", "Gärtanks", "Abfüllung"]:
            c = r["cons"][nm]
            binds = c["slack"] < 1e-6
            st = "**bindet**" if binds else "locker"
            rows += (f"| {nm} | {c['rhs']:.0f} | {c['lhs']:.1f} | "
                     f"{c['slack']:.1f} | {st} | {c['pi']:.2f} |\n")
        panel = mo.md(
            "### Solver-Report\n\n"
            f"**Status:** `{r['status']}`\n\n"
            f"**Mengen:** $x_H = {xH:.1f}$ hl,  $x_P = {xP:.1f}$ hl\n\n"
            f"**Gewinn:** $z^* = {z:.1f}$ €\n\n"
            "| Ressource | $b$ | Verbrauch | Schlupf | Status | $\\lambda$ |\n"
            "|---|---|---|---|---|---|\n"
            + rows +
            "\n*$\\lambda$ = Schattenpreis: Gewinn je zusätzlicher Einheit. "
            "Schlupf $> 0 \\Rightarrow \\lambda = 0$ — keine knappe Ressource "
            "(komplementärer Schlupf).*"
        )

        # ── RECHTS: 5 Wertfunktionen z*(Parameter) ─────────────────────────
        # Obere Reihe: Zielkoeffizienten c (2 Plots) · untere Reihe: rechte Seiten b (3)
        PAR = [(r"$c_H$ DB Helles", "cH", cH, (0, 12), 0, 0),
               (r"$c_P$ DB Pils", "cP", cP, (0, 12), 0, 1),
               (r"Malz $b_M$", "bM", bM, (4, 28), 1, 0),
               (r"Gärtanks $b_G$", "bG", bG, (4, 36), 1, 1),
               (r"Abfüllung $b_A$", "bA", bA, (4, 20), 1, 2)]
        base = dict(cH=cH, cP=cP, bM=bM, bG=bG, bA=bA)
        figR, axarr = plt.subplots(2, 3, figsize=(10.6, 6.6))
        for lab, key, cur, (lo, hi), rr, cc in PAR:
            ax2 = axarr[rr][cc]
            xs = np.linspace(lo, hi, 28)
            zs = []
            for v in xs:
                kw = dict(base); kw[key] = v
                zs.append(solve_brauerei(**kw)["z"])
            ax2.plot(xs, zs, color=IMSBlue, lw=2.4)
            ax2.axvline(cur, color=IMSOrange, ls=":", lw=1.2)
            ax2.scatter([cur], [z], s=95, color=IMSOrange, edgecolor="black",
                        lw=0.8, zorder=5)
            ax2.set_title(lab, fontsize=11.5)
            if cc == 0:
                ax2.set_ylabel(r"$z^*$ (€)", fontsize=10)
            ax2.grid(True, alpha=0.3, ls="--"); ax2.tick_params(labelsize=8.5)
        # freie Zelle oben rechts → Kurz-Legende
        axarr[0][2].axis("off")
        axarr[0][2].text(0.5, 0.5,
                         "oben:  Zielkoeffizienten $c$\n"
                         "unten:  rechte Seiten $b$\n\n"
                         "Steigung am Punkt\n= Schattenpreis",
                         ha="center", va="center", fontsize=10.5,
                         transform=axarr[0][2].transAxes)
        figR.tight_layout(pad=0.6)

        return mo.vstack([
            gf_reset,
            mo.hstack([gf_cH, gf_cP, gf_bM, gf_bG, gf_bA], widths="equal", gap=1.0),
            mo.hstack([mo.as_html(figL), panel, mo.as_html(figR)],
                      widths=[1.0, 0.6, 1.55], gap=1.0, align="start"),
        ])

    _render_guided()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "💡 Lösungen zu Aufgabe 1 (zum Aufklappen)": mo.md(r"""
    **1a — Engpass & Schattenpreise.** Im Ausgangsfall binden **Malz** und
    **Gärtanks** (Schlupf 0); **Abfüllung** hat **2 hl Reserve** (Schlupf 2).
    Schattenpreise: Malz $2{,}50$ €/t, Gärtanks $0{,}50$ €/Tankstd., Abfüllung $0$.
    → *„Eine Tonne Malz mehr bringt $2{,}50$ € mehr Gewinn."* Die Kurve $z^*(b_A)$ ist
    **flach**, weil Abfüllung nicht knapp ist: Schlupf $>0 \Rightarrow$ Schattenpreis
    $0$ (komplementärer Schlupf).

    **1b — Zulässige Änderung von $b_G$.** Die Steigung (= Schattenpreis $0{,}5$) bleibt
    konstant für $b_G \in [16,\ 32]$ → zulässige Erhöhung **$+8$** (bis 32), zulässige
    Senkung **$-8$** (bis 16).
    Faustregel: $\Delta z = \lambda \cdot \Delta b_G = 0{,}5 \cdot 6 = 3$ € → $z^*$ steigt von **52 auf 55** €.

    **1c — Am Knick.** Über $b_G = 32$ hinaus wandert die **Gärtanks**-Linie nach außen
    und **löst sich vom Optimum** (wird locker). Die optimale Ecke ist dann
    **Malz $\cap$ Abfüllung** bei $(x_H,x_P) = (8,4)$ mit $z^* = 56$ €. Der
    Schattenpreis von Gärtanks fällt auf **$0$** — eine nicht mehr bindende Ressource
    ist nichts wert (komplementärer Schlupf).

    **1d — Zielkoeffizient.** Der Mix $(4,6)$ bleibt optimal für
    $c_P \in [\approx 2{,}7,\ 8]$. **Darunter** kippt das Optimum auf $(8,0)$ — nur
    **Helles**, Pils fällt aus dem Mix. **Darüber** auf $(0,8)$ — nur **Pils**. Die
    Steigung von $z^*(c_P)$ im mittleren Stück ist $x_P^* = 6$: die **optimale
    Pils-Menge**.
    """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Teil 3 · Geführte Übung 2 — analytisch (Simplex-Tableau)

    Dasselbe Brauerei-Problem, jetzt **analytisch**: wir verfolgen den **Simplex**
    Schritt für Schritt. Der Regler führt durch die Tableaus — **links** wandert die
    Ecke im zulässigen Bereich mit, in der **Mitte** seht ihr das aktuelle **Tableau**
    (Pivot-Zelle orange), **rechts** die Interpretation.

    $$
    \begin{aligned}
    \max\ z = \;& 4x_H + 6x_P \\
    \text{s.t.}\;& x_H + 2x_P \leq 16 && \text{(Malz)} \\
    & 3x_H + 2x_P \leq 24 && \text{(Gärtanks)} \\
    & x_H + x_P \leq 12 && \text{(Abfüllung)} \\
    & x_H, x_P \geq 0
    \end{aligned}
    $$

    In Standardform mit Schlupfvariablen $s_1$ (Malz), $s_2$ (Gärtanks),
    $s_3$ (Abfüllung). Zieht den Regler bis zum **letzten Schritt** — dort beginnt
    Aufgabe 2a.
    """)
    return


@app.cell(hide_code=True)
def _():
    from fractions import Fraction as _Fr

    # Feinschrittiger Simplex: jeder Algorithmus-Schritt einzeln.
    #   kinds: init · col (Pivotspalte) · row (Min-Ratio) · pivot · optimum
    def simplex_micro(cH=4, cP=6, bM=16, bG=24, bA=12):
        c = [_Fr(cH), _Fr(cP), _Fr(0), _Fr(0), _Fr(0)]
        A = [[_Fr(1), _Fr(2), _Fr(1), _Fr(0), _Fr(0)],
             [_Fr(3), _Fr(2), _Fr(0), _Fr(1), _Fr(0)],
             [_Fr(1), _Fr(1), _Fr(0), _Fr(0), _Fr(1)]]
        b = [_Fr(bM), _Fr(bG), _Fr(bA)]
        basis = [2, 3, 4]
        T = [r[:] + [bb] for r, bb in zip(A, b)]
        z = [-cj for cj in c] + [_Fr(0)]
        names = ["x_H", "x_P", "s1", "s2", "s3"]

        def vert(Tb, bs):
            xb = {names[bi]: float(Tb[r][-1]) for r, bi in enumerate(bs)}
            return (xb.get("x_H", 0.0), xb.get("x_P", 0.0))

        steps = []

        def snap(kind, ent=None, leave=None, ratios=None, pivrc=None):
            steps.append({"kind": kind, "T": [r[:] for r in T], "z": z[:],
                          "basis": basis[:], "ent": ent, "leave": leave,
                          "ratios": ratios, "pivrc": pivrc, "vertex": vert(T, basis)})

        snap("init")
        guard = 0
        while True:
            guard += 1
            if guard > 30:
                break
            ent = min(range(5), key=lambda j: z[j])
            if z[ent] >= 0:
                snap("optimum"); break
            snap("col", ent=ent)
            ratios = [(T[i][-1] / T[i][ent] if T[i][ent] > 0 else None) for i in range(3)]
            valid = [(ratios[i], i) for i in range(3) if ratios[i] is not None]
            if not valid:
                snap("optimum"); break
            leave = min(valid, key=lambda t: (t[0], t[1]))[1]
            snap("row", ent=ent, leave=leave, ratios=ratios)
            pv = T[leave][ent]
            T[leave] = [x / pv for x in T[leave]]
            for i in range(3):
                if i != leave and T[i][ent] != 0:
                    f = T[i][ent]; T[i] = [a - f * bb for a, bb in zip(T[i], T[leave])]
            if z[ent] != 0:
                f = z[ent]; z = [a - f * bb for a, bb in zip(z, T[leave])]
            basis[leave] = ent
            snap("pivot", ent=ent, leave=leave, pivrc=(leave, ent))
        return steps

    SX_TEX = ["x_H", "x_P", "s_1", "s_2", "s_3"]

    def fmt_frac(x):
        x = _Fr(x)
        if x.denominator == 1:
            return str(x.numerator)
        s = "-" if x < 0 else ""
        return rf"${s}\frac{{{abs(x.numerator)}}}{{{x.denominator}}}$"

    return SX_TEX, fmt_frac, simplex_micro


@app.cell(hide_code=True)
def _(mo):
    cb_mengen = mo.ui.checkbox(label="Optimale Mengen (RHS-Spalte)")
    cb_pi = mo.ui.checkbox(label="Schattenpreise (z-Zeile · Schlupfspalten)")
    cb_binv = mo.ui.checkbox(label="B⁻¹ (Slack-Spalten im Körper)")
    return cb_binv, cb_mengen, cb_pi


@app.cell(hide_code=True)
def _(mo, simplex_micro):
    sx_steps = simplex_micro()
    sx_step = mo.ui.slider(start=0, stop=len(sx_steps) - 1, step=1, value=0,
                           label="Simplex-Schritt", show_value=True)
    return sx_step, sx_steps


@app.cell(hide_code=True)
def _(
    IMSBlue,
    IMSOrange,
    SX_TEX,
    cb_binv,
    cb_mengen,
    cb_pi,
    farbe_feasible,
    fmt_frac,
    gruen,
    line_box,
    lila,
    mo,
    region_verts,
    sx_step,
    sx_steps,
):
    def _render_simplex():
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plt.close("all")

        k = int(sx_step.value)
        step = sx_steps[k]
        kind = step["kind"]
        T, z, basis = step["T"], step["z"], step["basis"]
        ent, leave = step["ent"], step["leave"]
        flags = {"mengen": cb_mengen.value, "pi": cb_pi.value, "binv": cb_binv.value}
        COLS = ["Basis", r"$x_H$", r"$x_P$", r"$s_1$", r"$s_2$", r"$s_3$", "RHS"]

        # ── Highlight-Zellen je nach Schritt-Art ───────────────────────────
        hl = {}
        if kind == "col":
            for i in range(1, 5):
                hl[(i, ent + 1)] = "#CFE0F3"
        elif kind == "row":
            for i in range(1, 5):
                hl[(i, ent + 1)] = "#CFE0F3"
            for j in range(1, 7):
                hl[(leave + 1, j)] = "#FCE9B6"
            hl[(leave + 1, ent + 1)] = IMSOrange
        elif kind == "pivot":
            pr, pc = step["pivrc"]
            for i in range(1, 5):
                hl[(i, pc + 1)] = "#CFE0F3"
            hl[(pr + 1, pc + 1)] = IMSOrange
        elif kind == "optimum":
            if flags["mengen"]:
                for i in (1, 2, 3):
                    hl[(i, 6)] = "#BFE3C9"
            if flags["binv"]:
                for i in (1, 2, 3):
                    for j in (3, 4, 5):
                        hl[(i, j)] = "#CFE0F3"
            if flags["pi"]:
                for j in (3, 4, 5):
                    hl[(4, j)] = "#F3C9A6"

        # ── Tableau zeichnen ───────────────────────────────────────────────
        body = []
        for r in range(3):
            body.append([f"${SX_TEX[basis[r]]}$"]
                        + [fmt_frac(T[r][j]) for j in range(5)] + [fmt_frac(T[r][5])])
        body.append([r"$z$"] + [fmt_frac(z[j]) for j in range(5)] + [fmt_frac(z[5])])
        nR, nC, cw, chh = 5, 7, 1.0, 0.82
        x0, ytop = 0.2, nR * chh

        def cx(j):
            return x0 + cw * (j + 0.5)

        def cy(i):
            return ytop - chh * (i + 0.5)

        figT, ax = plt.subplots(figsize=(6.8, 3.6)); ax.axis("off")
        ax.set_xlim(0, x0 + nC * cw); ax.set_ylim(0, ytop + 0.1)
        for i in range(nR):
            for j in range(nC):
                fc = "white"
                if i == 0:
                    fc = "#7F8C8D"
                elif j == 0:
                    fc = "#EEF1F4"
                elif i == nR - 1:
                    fc = "#E5E7E9"
                al = 1.0 if (i == 0 or fc == "white") else 0.5
                if (i, j) in hl:
                    fc = hl[(i, j)]; al = 0.85
                ax.add_patch(Rectangle((cx(j) - cw / 2, cy(i) - chh / 2), cw, chh,
                             fc=fc, ec="#444", lw=1.0, alpha=al, zorder=2))
                txt = COLS[j] if i == 0 else body[i - 1][j]
                tc = "white" if i == 0 else "black"
                fw = "bold" if (i == 0 or j == 0 or i == nR - 1) else "normal"
                ax.text(cx(j), cy(i), txt, ha="center", va="center",
                        fontsize=12.5, color=tc, fontweight=fw, zorder=4)
        ax.plot([cx(0) + cw / 2] * 2, [ytop - nR * chh, ytop], color="#222", lw=1.6)
        ax.plot([cx(6) - cw / 2] * 2, [ytop - nR * chh, ytop], color="#222", lw=1.6)
        ax.plot([x0, x0 + nC * cw], [cy(nR - 1) + chh / 2] * 2, color="#222", lw=1.6)
        ax.plot([x0, x0 + nC * cw], [ytop - chh] * 2, color="#222", lw=1.6)
        figT.tight_layout(pad=0.3)

        # ── Graph: Pfad + aktuelle Ecke ────────────────────────────────────
        cons = [(1, 2, 16, "<="), (3, 2, 24, "<="), (1, 1, 12, "<=")]
        box = (-0.5, 10, -0.5, 10); rbox = (0, 10, 0, 10)
        cur = step["vertex"]
        zval = float(z[-1])
        figG, axg = plt.subplots(figsize=(5.4, 5.0))
        axg.set_xlim(*box[:2]); axg.set_ylim(*box[2:]); axg.grid(True, alpha=0.3, ls="--")
        axg.set_xlabel(r"$x_H$ — Helles (hl)"); axg.set_ylabel(r"$x_P$ — Pils (hl)")
        axg.axhline(0, color="k", lw=0.8); axg.axvline(0, color="k", lw=0.8)
        V = region_verts(cons, rbox)
        if len(V) >= 3:
            axg.fill([p[0] for p in V], [p[1] for p in V],
                     color=farbe_feasible, alpha=0.5, zorder=1)
        for (a, b, c, _s), col, lab in zip(cons, [IMSBlue, gruen, lila],
                                           ["Malz", "Gärtanks", "Abfüllung"]):
            p = line_box(a, b, c, box)
            if len(p) == 2:
                axg.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=col,
                         lw=1.8, label=lab, zorder=3)
        for v in V:
            axg.scatter(v[0], v[1], s=28, color="k", zorder=4)
        # Iso-Gewinn-Gerade durch die aktuelle Ecke ($4x_H + 6x_P = z$)
        iso = line_box(4, 6, zval, box)
        if len(iso) == 2:
            axg.plot([iso[0][0], iso[1][0]], [iso[0][1], iso[1][1]],
                     color=IMSOrange, ls=":", lw=2.0, zorder=5, label="Iso-Gewinn")
        path = []
        for i in range(k + 1):
            vv = sx_steps[i]["vertex"]
            if not path or path[-1] != vv:
                path.append(vv)
        for i in range(len(path) - 1):
            axg.annotate("", xy=path[i + 1], xytext=path[i],
                         arrowprops=dict(arrowstyle="-|>", color=IMSOrange, lw=2.2), zorder=6)
        if kind == "optimum" and flags["mengen"]:
            cxv, cyv = cur
            # Koordinaten x_H, x_P (graue Projektionspfeile auf die Achsen)
            axg.annotate("", xy=(cxv, 0), xytext=(cxv, cyv),
                         arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6, ls="--"), zorder=7)
            axg.annotate("", xy=(0, cyv), xytext=(cxv, cyv),
                         arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.6, ls="--"), zorder=7)
            axg.annotate(rf"$x_H={cxv:.0f}$", (cxv, 0), textcoords="offset points",
                         xytext=(4, -14), color="#555", fontweight="bold", fontsize=10.5)
            axg.annotate(rf"$x_P={cyv:.0f}$", (0, cyv), textcoords="offset points",
                         xytext=(-46, -4), color="#555", fontweight="bold", fontsize=10.5)
            # Schlupf s_3 (Abfüllung): Doppelpfeil vom Optimum zur lockeren Linie
            axg.annotate("", xy=(5, 7), xytext=(cxv, cyv),
                         arrowprops=dict(arrowstyle="<|-|>", color=lila, lw=2.0), zorder=7)
            axg.annotate(r"$s_3=2$", (5, 7), textcoords="offset points",
                         xytext=(6, 2), fontsize=11, color=lila, fontweight="bold")
        axg.scatter([cur[0]], [cur[1]], s=320, marker="*", color=IMSOrange,
                    edgecolor="k", lw=1.0, zorder=9)
        axg.legend(loc="upper right", fontsize=8)
        axg.set_title(f"Schritt {k}: Ecke ({cur[0]:.0f}, {cur[1]:.0f})")
        figG.tight_layout()

        # ── Interpretationstext ────────────────────────────────────────────
        xH, xP = cur
        zval = float(z[-1])
        if kind == "init":
            text = mo.md(
                "### Starttableau\n\n"
                "Alle Schlupfvariablen $s_1, s_2, s_3$ sind in der Basis, "
                "die Ecke ist $(0,0)$ mit $z=0$. Wir suchen die **Pivotspalte**."
            )
        elif kind == "col":
            text = mo.md(
                f"### Pivotspalte wählen\n\n"
                f"In der $z$-Zeile ist ${SX_TEX[ent]}$ mit {fmt_frac(z[ent])} am "
                f"**negativsten** → ${SX_TEX[ent]}$ kommt **in die Basis** "
                f"(blaue Spalte)."
            )
        elif kind == "row":
            rr = []
            for r in range(3):
                nm = SX_TEX[basis[r]]
                if step["ratios"][r] is not None:
                    rr.append(f"${nm}$: {fmt_frac(T[r][-1])} ÷ {fmt_frac(T[r][ent])} "
                              f"= {fmt_frac(step['ratios'][r])}")
                else:
                    rr.append(f"${nm}$: —")
            text = mo.md(
                "### Min-Ratio-Test\n\n"
                "RHS ÷ Pivotspalte (nur positive Einträge):\n\n- "
                + "\n- ".join(rr)
                + f"\n\nKleinster Quotient → ${SX_TEX[basis[leave]]}$ **verlässt** die "
                f"Basis (gelbe Zeile). Das **Pivot-Element** ist orange."
            )
        elif kind == "pivot":
            text = mo.md(
                f"### Pivotieren\n\n"
                f"${SX_TEX[ent]}$ ersetzt ${SX_TEX[basis[leave]]}$ in der Basis: "
                f"Pivotzeile normieren, übrige Zeilen + $z$-Zeile eliminieren. Neue "
                f"Ecke **$({xH:.0f}, {xP:.0f})$**, $z = {zval:.0f}$.\n\n"
                f"Gibt es noch einen negativen Eintrag in der $z$-Zeile? Dann weiter."
            )
        else:  # optimum
            base = (
                f"### Optimum — $({xH:.0f}, {xP:.0f})$, $z^* = {zval:.0f}$ €\n\n"
                f"Die $z$-Zeile ist überall $\\geq 0$ → **fertig**.\n\n"
                f"**Aufgabe 2a:** Hakt unten die Tableau-Teile an, um ihre Bedeutung "
                f"einzublenden (die **reduzierten Kosten** kommen in einer eigenen "
                f"Teilaufgabe)."
            )
            EXPL = {
                "mengen": "🟩 **Mengen (RHS-Spalte):** $x_H=4$, $x_P=6$, $s_3=2$ "
                "(freie Abfüllkapazität). $s_1=s_2=0$ → Malz und Gärtanks voll ausgelastet.",
                "pi": "🟧 **Schattenpreise ($z$-Zeile · Schlupfspalten):** "
                "$\\lambda_\\text{Malz}=\\frac{5}{2}$, $\\lambda_\\text{Gär}=\\frac{1}{2}$, "
                "$\\lambda_\\text{Abf}=0$. Nur **bindende** Ressourcen sind etwas wert.",
                "binv": "🟦 **$B^{-1}$ (Slack-Spalten im Körper):** sagt, wie sich die "
                "Basisvariablen bei einer RHS-Änderung verschieben "
                "($\\Delta x_B = B^{-1}\\,\\Delta b$) — Grundlage von Aufgabe 2b.",
            }
            active = [EXPL[key] for key in ["mengen", "pi", "binv"] if flags[key]]
            text = mo.md(base + ("\n\n" + "\n\n".join(active) if active else ""))

        # ── Layout ─────────────────────────────────────────────────────────
        head = [sx_step]
        if kind == "optimum":
            head.append(mo.hstack([cb_mengen, cb_pi, cb_binv], justify="start", gap=1.0))
        return mo.vstack(head + [
            mo.hstack([mo.as_html(figG), mo.as_html(figT), text],
                      widths=[1.0, 1.1, 0.95], gap=1.0, align="start"),
        ])

    _render_simplex()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Aufgabe 2b — rechte Seite ändern (Gültigkeitsbereich)

    Verstellt eine **rechte Seite** $b$ — die Änderung wird sofort **eingeblendet**
    (alte + neue Lage, Rechnung). Dann:

    - **→ in der Basis bewegen** übernimmt die neue Lösung als aktuelle — solange die
      **Basis gültig** bleibt ($x_B = B^{-1}b \geq 0$). Es ändert sich nur die
      RHS-Spalte.
    - **↻ neu lösen** löst das LP neu (Basiswechsel) — nötig, sobald der
      **Gültigkeitsbereich überschritten** ist.

    Dreht eine RHS so weit, bis eine Basisvariable **negativ** würde: dann geht nur
    noch *neu lösen*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    get_base, set_base = mo.state((16, 24, 12))
    return get_base, set_base


@app.cell(hide_code=True)
def _(mo):
    rb_M = mo.ui.slider(start=8, stop=28, step=1, value=16,
                        label="$b_M$ Malz (t)", show_value=True)
    rb_G = mo.ui.slider(start=12, stop=40, step=1, value=24,
                        label="$b_G$ Gärtanks (Tankstd.)", show_value=True)
    rb_A = mo.ui.slider(start=6, stop=18, step=1, value=12,
                        label="$b_A$ Abfüllung (hl)", show_value=True)
    return rb_A, rb_G, rb_M


@app.cell(hide_code=True)
def _(get_base, simplex_micro):
    # Aktuelle Basis (committed) → ihr Endtableau, B^-1 und Basiswerte
    base_b = get_base()
    base_tab = simplex_micro(bM=base_b[0], bG=base_b[1], bA=base_b[2])[-1]
    basis_base = base_tab["basis"]
    Binv_base = [[base_tab["T"][r][2], base_tab["T"][r][3], base_tab["T"][r][4]]
                 for r in range(3)]
    xB_base = [base_tab["T"][r][5] for r in range(3)]
    return Binv_base, base_b, base_tab, basis_base, xB_base


@app.cell(hide_code=True)
def _(Binv_base, mo, rb_A, rb_G, rb_M, set_base):
    from fractions import Fraction as _Fr

    def _commit_move(_v):
        prop = (int(rb_M.value), int(rb_G.value), int(rb_A.value))
        bnew = [_Fr(prop[0]), _Fr(prop[1]), _Fr(prop[2])]
        xB = [sum(Binv_base[r][j] * bnew[j] for j in range(3)) for r in range(3)]
        if all(x >= 0 for x in xB):          # nur bei gültiger Basis übernehmen
            set_base(prop)

    def _commit_resolve(_v):
        set_base((int(rb_M.value), int(rb_G.value), int(rb_A.value)))

    btn_move = mo.ui.button(label="→ in der Basis bewegen", on_change=_commit_move)
    btn_resolve = mo.ui.button(label="↻ neu lösen", kind="warn", on_change=_commit_resolve)
    return btn_move, btn_resolve


@app.cell(hide_code=True)
def _(
    Binv_base,
    IMSBlue,
    IMSOrange,
    SX_TEX,
    base_b,
    base_tab,
    basis_base,
    btn_move,
    btn_resolve,
    farbe_feasible,
    gruen,
    line_box,
    lila,
    mo,
    rb_A,
    rb_G,
    rb_M,
    region_verts,
    rot,
    xB_base,
):
    def _fr(x):
        from fractions import Fraction as _F
        x = _F(x)
        if x.denominator == 1:
            return str(x.numerator)
        return rf"${'-' if x < 0 else ''}\frac{{{abs(x.numerator)}}}{{{x.denominator}}}$"

    def _render_rhs():
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, PathPatch
        from matplotlib.path import Path
        from fractions import Fraction as _Fr
        plt.close("all")

        prop = (int(rb_M.value), int(rb_G.value), int(rb_A.value))
        changed = prop != base_b
        coef = [(1, 2), (3, 2), (1, 1)]
        ccol = [IMSBlue, gruen, lila]
        cnames = ["Malz", "Gärtanks", "Abfüllung"]
        XH_COL, XP_COL = "#00838F", "#AD1457"

        def varcolor(bi):
            return XH_COL if bi == 0 else XP_COL if bi == 1 else ccol[bi - 2]

        def cons_of(bb):
            return [(coef[i][0], coef[i][1], bb[i], "<=") for i in range(3)]

        def ft(x):
            x = _Fr(x)
            return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"

        def span(t, c):
            return f"<span style='color:{c}'>{t}</span>"

        xHb, xPb = base_tab["vertex"]
        zb = base_tab["z"][-1]
        bnew = [_Fr(prop[0]), _Fr(prop[1]), _Fr(prop[2])]
        xBp = [sum(Binv_base[r][j] * bnew[j] for j in range(3)) for r in range(3)]
        valid = all(x >= 0 for x in xBp)
        Tproj = [base_tab["T"][r][:5] + [xBp[r]] for r in range(3)]
        dproj = {basis_base[r]: xBp[r] for r in range(3)}
        xHp = float(dproj.get(0, 0)); xPp = float(dproj.get(1, 0))
        zp = 4 * dproj.get(0, _Fr(0)) + 6 * dproj.get(1, _Fr(0))
        proj_tab = {"vertex": (xHp, xPp), "basis": basis_base, "T": Tproj}

        # Statische Achsen — kein Umskalieren beim Verstellen (besser lesbar).
        # Fenster gross genug für alle Slider-Kombinationen + Platz für Klammern.
        XMAX, YMAX = 14.0, 14.0
        box = (-2.5, XMAX, -2.0, YMAX); rbox = (0, XMAX, 0, YMAX)

        figG, axg = plt.subplots(figsize=(7.6, 6.8))
        axg.set_xlim(*box[:2]); axg.set_ylim(*box[2:]); axg.grid(True, alpha=0.3, ls="--")
        axg.set_xlabel(r"$x_H$ — Helles (hl)"); axg.set_ylabel(r"$x_P$ — Pils (hl)")
        axg.axhline(0, color="k", lw=0.8); axg.axvline(0, color="k", lw=0.8)

        def draw_iso(zval, col):
            p = line_box(4, 6, zval, box)
            if len(p) == 2:
                axg.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=col, ls=":",
                         lw=2.0, zorder=5)

        def draw_arrows(tab, alpha, labels):
            xval = {tab["basis"][r]: tab["T"][r][5] for r in range(3)}
            xH = float(xval.get(0, 0)); xP = float(xval.get(1, 0))
            axg.annotate("", xy=(xH, 0), xytext=(xH, xP),
                         arrowprops=dict(arrowstyle="-|>", color=XH_COL, lw=1.8,
                                         ls="--", alpha=alpha), zorder=7)
            axg.annotate("", xy=(0, xP), xytext=(xH, xP),
                         arrowprops=dict(arrowstyle="-|>", color=XP_COL, lw=1.8,
                                         ls="--", alpha=alpha), zorder=7)
            if labels:
                axg.annotate(rf"$x_H={ft(xval.get(0, _Fr(0)))}$", (xH, 0),
                             textcoords="offset points", xytext=(3, -13),
                             color=XH_COL, fontsize=10, fontweight="bold")
                axg.annotate(rf"$x_P={ft(xval.get(1, _Fr(0)))}$", (0, xP),
                             textcoords="offset points", xytext=(-50, -3),
                             color=XP_COL, fontsize=10, fontweight="bold")
            for r in range(3):
                bi = tab["basis"][r]
                if bi >= 2:
                    i = bi - 2; a1, a2 = coef[i]; sval = tab["T"][r][5]
                    if sval > 0:
                        t = float(sval) / (a1 * a1 + a2 * a2)
                        foot = (xH + t * a1, xP + t * a2)
                        axg.annotate("", xy=foot, xytext=(xH, xP),
                                     arrowprops=dict(arrowstyle="-|>", color=ccol[i],
                                                     lw=1.8, alpha=alpha), zorder=7)
                        if labels:
                            axg.annotate(rf"$s_{i + 1}={ft(sval)}$", foot,
                                         textcoords="offset points", xytext=(5, 3),
                                         color=ccol[i], fontsize=9.5, fontweight="bold")

        def hbrace(x0, x1, y, h, color, label):
            xm = (x0 + x1) / 2
            verts = [(x0, y), (x0, y - h), (xm, y - h), (x1, y - h), (x1, y)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
            axg.add_patch(PathPatch(Path(verts, codes), fill=False, edgecolor=color,
                                    lw=1.5, zorder=8))
            axg.annotate(label, (xm, y - h), textcoords="offset points", xytext=(0, -8),
                         ha="center", va="top", color=color, fontsize=9.5,
                         fontweight="bold", zorder=9)

        def vbrace(y0, y1, x, h, color, label):
            ym = (y0 + y1) / 2
            verts = [(x, y0), (x - h, y0), (x - h, ym), (x - h, y1), (x, y1)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CURVE3, Path.CURVE3]
            axg.add_patch(PathPatch(Path(verts, codes), fill=False, edgecolor=color,
                                    lw=1.5, zorder=8))
            axg.annotate(label, (x - h, ym), textcoords="offset points", xytext=(-5, 0),
                         ha="right", va="center", color=color, fontsize=9.5,
                         fontweight="bold", zorder=9)

        Vb = region_verts(cons_of(base_b), rbox)
        if len(Vb) >= 3:
            axg.fill([p[0] for p in Vb], [p[1] for p in Vb],
                     color=farbe_feasible, alpha=0.45, zorder=1)
        for i in range(3):
            a1, a2 = coef[i]
            p = line_box(a1, a2, base_b[i], box)
            if len(p) == 2:
                axg.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=ccol[i],
                         lw=1.8, label=cnames[i], zorder=3)
        if changed:
            for i in range(3):
                if prop[i] != base_b[i]:
                    a1, a2 = coef[i]
                    p2 = line_box(a1, a2, prop[i], box)
                    if len(p2) == 2:
                        axg.plot([p2[0][0], p2[1][0]], [p2[0][1], p2[1][1]],
                                 color=ccol[i], lw=2.4, ls="--", zorder=4)
            draw_iso(float(zb), "#bbbbbb")
            if valid:
                draw_iso(float(zp), IMSOrange)
            draw_arrows(base_tab, alpha=0.28, labels=False)   # alt (blass)
            draw_arrows(proj_tab, alpha=1.0, labels=False)     # neu (kräftig)

            # Differenzen mit Klammern + Werten
            bxv = {base_tab["basis"][r]: base_tab["T"][r][5] for r in range(3)}
            nxv = {basis_base[r]: xBp[r] for r in range(3)}
            xHo, xHn = float(bxv.get(0, 0)), float(nxv.get(0, 0))
            xPo, xPn = float(bxv.get(1, 0)), float(nxv.get(1, 0))
            if abs(xHn - xHo) > 1e-9:
                hbrace(min(xHo, xHn), max(xHo, xHn), -0.25, 0.7, XH_COL,
                       f"Δ$x_H$={ft(nxv.get(0, _Fr(0)) - bxv.get(0, _Fr(0)))}")
            if abs(xPn - xPo) > 1e-9:
                vbrace(min(xPo, xPn), max(xPo, xPn), -0.25, 0.7, XP_COL,
                       f"Δ$x_P$={ft(nxv.get(1, _Fr(0)) - bxv.get(1, _Fr(0)))}")
            if valid:
                for r in range(3):           # Slack-Differenz (erster Slack der Basis)
                    bi = basis_base[r]
                    if bi >= 2:
                        i = bi - 2; a1, a2 = coef[i]
                        so, sn = base_tab["T"][r][5], xBp[r]
                        if so != sn:
                            to = float(so) / (a1 * a1 + a2 * a2)
                            tn = float(sn) / (a1 * a1 + a2 * a2)
                            fo = (xHo + to * a1, xPo + to * a2)
                            fn = (xHn + tn * a1, xPn + tn * a2)
                            axg.annotate("", xy=fn, xytext=fo,
                                         arrowprops=dict(arrowstyle="<|-|>", color=ccol[i],
                                                         lw=1.3, ls=":"), zorder=8)
                            mid = ((fo[0] + fn[0]) / 2, (fo[1] + fn[1]) / 2)
                            axg.annotate(rf"Δ$s_{i + 1}$={ft(sn - so)}", mid,
                                         textcoords="offset points", xytext=(5, 4),
                                         color=ccol[i], fontsize=9, fontweight="bold", zorder=9)
                        break
                if float(zp) != float(zb):   # Iso-Differenz = Δz
                    xq = 1.2
                    yo = (float(zb) - 4 * xq) / 6; yn = (float(zp) - 4 * xq) / 6
                    axg.annotate("", xy=(xq, yn), xytext=(xq, yo),
                                 arrowprops=dict(arrowstyle="<|-|>", color=IMSOrange,
                                                 lw=1.3), zorder=8)
                    axg.annotate(rf"Δz={ft(zp - zb)}", (xq, (yo + yn) / 2),
                                 textcoords="offset points", xytext=(6, 0),
                                 color=IMSOrange, fontsize=9.5, fontweight="bold", zorder=9)
            axg.scatter([xHb], [xPb], s=55, facecolor="white", edgecolor="k",
                        lw=1.2, zorder=6)
            pcol = IMSOrange if valid else rot
            axg.scatter([xHp], [xPp], s=150, marker="*", color=pcol,
                        edgecolor="k", lw=1.0, zorder=10)
            if not valid:
                axg.annotate("unzulässig", (xHp, xPp), textcoords="offset points",
                             xytext=(8, -2), color=rot, fontweight="bold", fontsize=9.5)
            axg.set_title("Änderung eingeblendet")
        else:
            draw_iso(float(zb), IMSOrange)
            draw_arrows(base_tab, alpha=1.0, labels=True)
            axg.scatter([xHb], [xPb], s=150, marker="*", color=IMSOrange,
                        edgecolor="k", lw=1.0, zorder=10)
            axg.set_title("Aktueller Zustand")
        axg.legend(loc="upper right", fontsize=8)
        figG.tight_layout()

        COLS = ["Basis", r"$x_H$", r"$x_P$", r"$s_1$", r"$s_2$", r"$s_3$", "RHS"]

        def draw_tab(T, z, bss, title, rhs_cols=None, mark_neg=False):
            body = []
            for r in range(3):
                body.append([f"${SX_TEX[bss[r]]}$"]
                            + [_fr(T[r][j]) for j in range(5)] + [_fr(T[r][5])])
            body.append([r"$z$"] + [_fr(z[j]) for j in range(5)] + [_fr(z[5])])
            nR, nC, cw, chh = 5, 7, 1.0, 0.7
            x0, ytop = 0.2, nR * chh
            fig, ax = plt.subplots(figsize=(5.8, 2.7)); ax.axis("off")
            ax.set_xlim(0, x0 + nC * cw); ax.set_ylim(0, ytop + 0.55)
            ax.text((x0 + nC * cw) / 2, ytop + 0.28, title, ha="center", fontsize=11,
                    fontweight="bold")
            for i in range(nR):
                for j in range(nC):
                    fc = "white"
                    if i == 0:
                        fc = "#7F8C8D"
                    elif j == 0:
                        fc = "#EEF1F4"
                    elif i == nR - 1:
                        fc = "#E5E7E9"
                    if j == 6 and 1 <= i <= 3:
                        fc = "#EAEFF5"
                        if rhs_cols and (i - 1) in rhs_cols:
                            fc = rhs_cols[i - 1]
                        if mark_neg and T[i - 1][5] < 0:
                            fc = "#E8A6A0"
                    al = 1.0 if (i == 0 or fc == "white") else 0.6
                    ax.add_patch(Rectangle((x0 + cw * j, ytop - chh * (i + 1)), cw, chh,
                                 fc=fc, ec="#444", lw=0.9, alpha=al, zorder=2))
                    txt = COLS[j] if i == 0 else body[i - 1][j]
                    tc = "white" if i == 0 else "black"
                    fw = "bold" if (i == 0 or j == 0 or i == nR - 1) else "normal"
                    ax.text(x0 + cw * (j + 0.5), ytop - chh * (i + 0.5), txt,
                            ha="center", va="center", fontsize=11, color=tc,
                            fontweight=fw, zorder=4)
            fig.tight_layout(pad=0.2)
            return fig

        rhs_cols = {r: varcolor(basis_base[r]) + "55" for r in range(3)}
        if changed:
            zproj = base_tab["z"][:5] + [zp]
            tab1 = mo.as_html(draw_tab(base_tab["T"], base_tab["z"], basis_base,
                                       "aktuell", rhs_cols=rhs_cols))
            tab2 = mo.as_html(draw_tab(Tproj, zproj, basis_base,
                                       r"projiziert:  $x_B = B^{-1} b$",
                                       rhs_cols=rhs_cols, mark_neg=True))
            db = [prop[i] - base_b[i] for i in range(3)]
            incr, deltas = [], []
            for r in range(3):
                bi = basis_base[r]; v = SX_TEX[bi]; col = varcolor(bi)
                chg = " + ".join(f"({ft(Binv_base[r][j])})·({db[j]:+d})" for j in range(3))
                incr.append(span(f"${v}$ = {ft(xB_base[r])} + [{chg}] = **{ft(xBp[r])}**", col))
                deltas.append(span(f"Δ${v}$ = {ft(xBp[r] - xB_base[r])}", col))
            dz = zp - zb
            verdict = (f"✅ alle $x_B \\geq 0$ → **Basis gültig**. Mit "
                       f"**„in der Basis bewegen“** übernehmen."
                       if valid else
                       f"⚠️ eine Basisvariable wird **negativ** → **Basis ungültig** "
                       f"(Gültigkeitsbereich überschritten). Nur **„neu lösen“** geht.")
            update_md = mo.md(
                "**Update** &nbsp; $x_B^{neu} = x_B^{alt} + B^{-1}\\,\\Delta b$\n\n"
                + "  \n".join(incr))
            changes_md = mo.md(
                "**Änderungen:** " + " · ".join(deltas)
                + f" · Δz = {span(ft(dz) + ' €', IMSOrange)}\n\n" + verdict)
            right = mo.hstack([
                mo.vstack([tab1, update_md]),
                mo.vstack([tab2, changes_md]),
            ], widths="equal", gap=0.8, align="start")
        else:
            rows = [span(f"${SX_TEX[basis_base[r]]} = {ft(base_tab['T'][r][5])}$",
                         varcolor(basis_base[r])) for r in range(3)]
            single = mo.as_html(draw_tab(base_tab["T"], base_tab["z"], basis_base,
                                         "aktuelles Endtableau", rhs_cols=rhs_cols))
            panel = mo.md(
                f"### Aktueller Zustand\n\n"
                f"Optimum **$({xHb:.0f}, {xPb:.0f})$**, $z^* = {ft(zb)}$ €, "
                f"$b = {base_b}$.\n\n"
                f"Basisvariablen: " + " · ".join(rows) + "\n\n"
                f"Die Pfeile (Achsen + Schlupf) und die RHS-Zellen sind je Variable "
                f"gleich gefärbt. Verstellt eine RHS, um die Änderung einzublenden."
            )
            right = mo.vstack([single, panel])

        return mo.vstack([
            mo.hstack([rb_M, rb_G, rb_A], widths="equal", gap=1.0),
            mo.hstack([btn_move, btn_resolve], justify="start", gap=1.0),
            mo.hstack([mo.as_html(figG), right],
                      widths=[1.0, 1.5], gap=1.2, align="start"),
        ])

    _render_rhs()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Teil 4 · Reduzierte Kosten & selbständige Aufgaben

    ## Konzept: Reduzierte Kosten — lohnt sich ein neues Produkt?

    Ein **neues Produkt** $i$ verbraucht von jeder Ressource $j$ die Menge $a_{ij}$.
    Jede Ressource ist im Optimum genau so viel wert wie ihr **Schattenpreis**
    $\lambda_j$. Der **Ressourcenwert** des Produkts ist also $\sum_j a_{ij}\lambda_j$
    — das sind die *Opportunitätskosten*: so viel Gewinn entgeht, wenn wir die
    Ressourcen statt für den bisherigen Mix für das neue Produkt einsetzen.

    **Reduzierte Kosten:**

    $$ RC_i \;=\; \underbrace{c_i}_{\text{Deckungsbeitrag}} \;-\; \underbrace{\sum_j a_{ij}\,\lambda_j}_{\text{Ressourcenwert}} $$

    | $RC_i$ | Bedeutung | Entscheidung |
    |---|---|---|
    | $>0$ | DB übersteigt den Ressourcenwert | **aufnehmen** (verbessert $z$ → neu optimieren) |
    | $=0$ | genau gleich | **indifferent** (typisch für Basisvariablen) |
    | $<0$ | Ressourcen sind anderswo mehr wert | **verwerfen** |

    **Beispiel Brauerei** ($\lambda_\text{Malz}=\tfrac52,\ \lambda_\text{Gär}=\tfrac12,\ \lambda_\text{Abf}=0$):
    Ein „Bockbier" mit DB 6 €/hl, das 1 Malz, 3 Tankstunden und 1 Abfüllung braucht:
    $$ RC = 6 - \left(1\cdot\tfrac52 + 3\cdot\tfrac12 + 1\cdot 0\right) = 6 - 4 = 2 > 0 \;\Rightarrow\; \mathbf{aufnehmen.} $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 🗒️ Spickzettel — analytische Sensitivität

    | Größe | Formel / Ablesen | Merksatz |
    |---|---|---|
    | **Schattenpreis** $\lambda_j$ | Endtableau: $z$-Zeile unter Schlupfspalte $s_j$ | $\Delta z$ je $+1$ Einheit $b_j$; nichtbindend $\Rightarrow \lambda_j=0$ |
    | **RHS-Wirkung** | $\Delta z = \lambda_j\cdot\Delta b_j$ | nur im Gültigkeitsbereich |
    | **Mengen-Update** | $x_B = B^{-1}b = x_B^{alt} + B^{-1}\Delta b$ | $B^{-1}$ = Schlupfspalten im Tableau |
    | **Gültigkeitsbereich** | solange **alle** $x_B \ge 0$ | erste Basisvariable, die $0$ wird = Grenze |
    | **Reduzierte Kosten** | $RC_i = c_i - \sum_j a_{ij}\lambda_j$ | $>0$ aufnehmen · $<0$ verwerfen · $=0$ indifferent |
    | **Kompl. Schlupf** | $\text{Schlupf}_j>0 \Rightarrow \lambda_j=0$ | knappe Ressource $\Leftrightarrow$ Preis $>0$ |

    *Damit löst ihr die folgenden drei Aufgaben selbständig.*
    """)
    return


@app.cell(hide_code=True)
def _(IMSBlue, IMSOrange, farbe_feasible, gruen, line_box, lila, mo, region_verts):
    def _aufg1():
        import matplotlib.pyplot as plt
        plt.close("all")
        cons = [(1, 2, 16, "<="), (3, 2, 24, "<="), (1, 1, 12, "<=")]
        box = (-0.6, 12, -0.6, 12); rbox = (0, 12, 0, 12)
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.set_xlim(*box[:2]); ax.set_ylim(*box[2:]); ax.grid(True, alpha=0.3, ls="--")
        ax.set_xlabel(r"$x_H$ — Helles (hl)"); ax.set_ylabel(r"$x_P$ — Pils (hl)")
        ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
        V = region_verts(cons, rbox)
        if len(V) >= 3:
            ax.fill([p[0] for p in V], [p[1] for p in V], color=farbe_feasible,
                    alpha=0.5, zorder=1)
        for (a, b, c, _s), col, lab in zip(cons, [IMSBlue, gruen, lila],
                                           ["Malz", "Gärtanks", "Abfüllung"]):
            p = line_box(a, b, c, box)
            if len(p) == 2:
                # Abfüllung ist im Basisfall nicht bindend → gestrichelt
                dashed = (lab == "Abfüllung")
                ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=col, lw=1.8,
                        ls="--" if dashed else "-",
                        label=lab + (" (locker)" if dashed else ""), zorder=3)
        for v in V:
            ax.scatter(v[0], v[1], s=26, color="k", zorder=4)
        ax.scatter([4], [6], s=240, marker="*", color=IMSOrange, edgecolor="k",
                   lw=1.0, zorder=9)
        ax.annotate(r"$x^*=(4,6)$", (4, 6), textcoords="offset points",
                    xytext=(8, -16), color=IMSOrange, fontweight="bold", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Aufgabe 1 — hier $\\Delta$ einzeichnen")
        fig.tight_layout()
        return fig

    _q = mo.md(r"""
    ---

    ## Aufgabe 1 — Sensitivität ablesen & einzeichnen

    **Gegeben** (nichts selbst zu lösen): die Brauerei mit drei Restriktionen, ihre
    **graphische Lösung** (unten) und das **Endtableau**.

    $$\begin{aligned}
    \max\ z = \;& 4x_H + 6x_P\\
    \text{s.t.}\quad x_H + 2x_P &\le 16 && \text{(Malz)}\\
    3x_H + 2x_P &\le 24 && \text{(Gärtanks)}\\
    x_H + x_P &\le 12 && \text{(Abfüllung)}\\
    x_H,\, x_P &\ge 0
    \end{aligned}$$

    Optimum $x_H^*=4,\ x_P^*=6,\ z^*=52$. Endtableau (Basis $x_P, x_H, s_3$):

    | Basis | $x_H$ | $x_P$ | $s_1$ | $s_2$ | $s_3$ | RHS |
    |---|---|---|---|---|---|---|
    | $x_P$ | 0 | 1 | $\tfrac34$ | $-\tfrac14$ | 0 | 6 |
    | $x_H$ | 1 | 0 | $-\tfrac12$ | $\tfrac12$ | 0 | 4 |
    | $s_3$ | 0 | 0 | $-\tfrac14$ | $-\tfrac14$ | 1 | 2 |
    | $z$ | 0 | 0 | $\tfrac52$ | $\tfrac12$ | 0 | 52 |

    **a)** Lies die drei **Schattenpreise** aus dem Tableau ab. Welche Ressource ist
    **nicht knapp** — woran erkennst du das?
    **b)** Die Brauerei kann **4 zusätzliche Tankstunden** zu $0{,}40$ €/h mieten.
    Lohnt sich das? Bis zu welcher Gärtanks-Menge gilt der Schattenpreis?
    **c)** **Zeichne** in die Grafik ein: das neue Optimum bei $b_\text{Gär}=28$ sowie
    die Verschiebungen $\Delta x_H$, $\Delta x_P$ und $\Delta s_\text{Abf}$.
    """)

    _loes = mo.accordion({
        "💡 Lösung Aufgabe 1": mo.md(r"""
    **a)** Aus der $z$-Zeile unter $s_1,s_2,s_3$: $\lambda_\text{Malz}=\tfrac52$,
    $\lambda_\text{Gär}=\tfrac12$, $\lambda_\text{Abf}=0$. Die **Abfüllung** ist nicht
    knapp — ihre Schlupfvariable $s_3=2$ steht in der Basis (Schlupf $>0$), daher
    $\lambda_\text{Abf}=0$ (komplementärer Schlupf).

    **b)** $\lambda_\text{Gär}=0{,}50 > 0{,}40$ → **ja**, jede Stunde bringt netto
    $0{,}10$ €. Gültig, solange die Basis hält: $x_H=-8+\tfrac12 b_\text{Gär}\ge0$ und
    $s_3=8-\tfrac14 b_\text{Gär}\ge0$ → $b_\text{Gär}\in[16,\,32]$. Die $+4$ (auf 28)
    liegen drin → $\Delta z = 0{,}5\cdot 4 = 2$ €.

    **c)** Bei $b_\text{Gär}=28$: $x^*=(6,5)$, $z^*=54$. Verschiebungen
    $\Delta x_H=+2$, $\Delta x_P=-1$, $\Delta s_\text{Abf}=-1$ (von $2$ auf $1$).
    """)
    })
    mo.vstack([_q, mo.as_html(_aufg1()), _loes])
    return


@app.cell(hide_code=True)
def _(mo):
    _q = mo.md(r"""
    ---

    ## Aufgabe 2 — Reduzierte Kosten: neues Produkt?

    Schattenpreise des aktuellen Optimums: $\lambda_\text{Malz}=\tfrac52$,
    $\lambda_\text{Gär}=\tfrac12$, $\lambda_\text{Abf}=0$. Die Brauerei prüft drei neue
    Sorten:

    | Sorte | DB $c$ [€/hl] | Malz | Gärtanks | Abfüllung |
    |---|---|---|---|---|
    | **Bockbier** | 6 | 1 | 3 | 1 |
    | **Radler** | 2,5 | 1 | 0 | 1 |
    | **Export** | 4 | 2 | 1 | 1 |

    **a)** Berechne für jede Sorte die **reduzierten Kosten** $RC = c - \sum_j a_j\lambda_j$.
    **b)** Welche Sorte(n) **aufnehmen**, welche **verwerfen**?
    **c)** Was bedeutet $RC=0$ ökonomisch?
    """)
    _loes = mo.accordion({
        "💡 Lösung Aufgabe 2": mo.md(r"""
    **a)** Ressourcenwert $\sum a_j\lambda_j$ und $RC$:

    - **Bockbier:** $1\cdot\tfrac52+3\cdot\tfrac12+1\cdot0 = 4$ → $RC = 6-4 = +2$
    - **Radler:** $1\cdot\tfrac52+0+1\cdot0 = \tfrac52$ → $RC = 2{,}5-2{,}5 = 0$
    - **Export:** $2\cdot\tfrac52+1\cdot\tfrac12+1\cdot0 = 5{,}5$ → $RC = 4-5{,}5 = -1{,}5$

    **b)** **Bockbier aufnehmen** ($RC>0$, verbessert $z$ → neu optimieren), **Export
    verwerfen** ($RC<0$). Radler ist Grenzfall.

    **c)** $RC=0$: das Produkt ist gerade **kostendeckend** zu den aktuellen
    Schattenpreisen — seine Aufnahme ändert $z$ (zunächst) nicht. Genau das gilt für
    alle **Basisvariablen** im Optimum.
    """)
    })
    mo.vstack([_q, _loes])
    return


@app.cell(hide_code=True)
def _(mo):
    _q = mo.md(r"""
    ---

    ## Aufgabe 3 — Möbelwerkstatt (3 Produkte, 3 Ressourcen)

    Eine **Möbelwerkstatt** fertigt drei Produkte $A\,(x_1)$, $B\,(x_2)$, $C\,(x_3)$ mit
    Deckungsbeiträgen $5, 4, 3$ €. Drei Ressourcen begrenzen die Produktion:

    $$\begin{aligned}
    \max\ z = \;& 5x_1+4x_2+3x_3\\
    \text{s.t.}\quad 2x_1+3x_2+\ x_3 &\le 5 && \text{(Holz)}\\
    4x_1+\ x_2+2x_3 &\le 11 && \text{(Maschinenzeit)}\\
    3x_1+4x_2+2x_3 &\le 8 && \text{(Montagezeit)}\\
    x_1,x_2,x_3 &\ge 0
    \end{aligned}$$

    **Finales Simplex-Tableau** (Schlupf $s_1, s_2, s_3$):

    | Basis | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | RHS |
    |---|---|---|---|---|---|---|---|
    | $x_1$ | 1 | 2 | 0 | 2 | 0 | $-1$ | 2 |
    | $x_3$ | 0 | $-1$ | 1 | $-3$ | 0 | 2 | 1 |
    | $s_2$ | 0 | $-5$ | 0 | $-2$ | 1 | 0 | 1 |
    | $z$ | 0 | 3 | 0 | 1 | 0 | 1 | 13 |

    **Beantworte alles direkt aus dem Tableau:**
    **a)** Optimale Lösung und $z^*$ ablesen — welche Produkte werden gefertigt?
    **b)** Schattenpreise ablesen ($z$-Zeile unter $s_1,s_2,s_3$) — welche Ressource ist **nicht knapp**?
    **c)** Warum lohnt sich Produkt $B$ **nicht**?
    **d)** Neues Produkt $D$ (DB 6) braucht 1 Holz, 2 Maschinen-, 2 Montagestunden:
    berechne $RC_D$ — aufnehmen?
    """)
    _loes = mo.accordion({
        "💡 Lösung Aufgabe 3": mo.md(r"""
    **a)** RHS-Spalte: $x_1=2$, $x_3=1$, $s_2=1$, $x_2=0$ → $x^*=(2,0,1)$, $z^*=13$ €;
    **A** und **C** werden gefertigt, **B** nicht.

    **b)** $z$-Zeile unter $s_1,s_2,s_3$: $\lambda=(1,0,1)$. **Maschinenzeit ($R_2$)** ist
    nicht knapp — ihre Schlupfvariable $s_2=1$ steht in der Basis (Schlupf $>0$), daher
    $\lambda_{R_2}=0$.

    **c)** Der Eintrag der $z$-Zeile unter $x_2$ ist $3>0$ → $B$ in die Basis zu nehmen
    würde $z$ um $3$ €/Stück **senken**.
    Gleichbedeutend $RC_B = 4-(3\cdot1+1\cdot0+4\cdot1) = -3 < 0$.

    **d)** $RC_D = 6-(1\cdot1+2\cdot0+2\cdot1) = 6-3 = +3 > 0$ → **aufnehmen** (neu
    optimieren).
    """)
    })
    mo.vstack([_q, _loes])
    return


if __name__ == "__main__":
    app.run()
