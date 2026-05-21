# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(
    width="full",
    app_title="PuE Übung 4: Simplex-Algorithmus",
    layout_file="layouts/pue-ue-04-simplex.slides.json",
    css_file="custom.css",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    from scipy.optimize import linprog
    return linprog, mo, np


@app.cell(hide_code=True)
def _():
    IMSBlue = "#023B88"
    IMSOrange = "#D87237"
    rot = "#C0392B"
    gruen = "#27AE60"
    farbe_feasible = "#9FBEE6"
    return IMSBlue, IMSOrange, farbe_feasible, gruen, rot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Übung 4: Simplex-Algorithmus

    **Planung und Entscheidung — SS 2026**

    In Übung 3 haben wir LPs *gesehen* — heute lernen wir, sie systematisch zu *rechnen*.
    Wir folgen direkt der Mechanik aus **VL 05**: Standardform → Tableau → Pivotschritte.

    1. **Wiederholung** *(5 Min.)* — Tableau-Aufbau & Pivotregel als Cheat-Card
    2. **Geführte Übung — Brauerei Würzburg** *(15 Min.)* — 2-Variablen-LP mit
       Tableau **und** zulässiger Menge nebeneinander; jeden Pivotschritt
       *sehen* und *rechnen* gleichzeitig.
    3. **Spielwiese — LP-Baukasten** *(10 Min.)* — eigenes LP zusammenklicken,
       Validität prüfen, Schritt-für-Schritt-Simplex und alle Sonderfälle der VL
       (unbeschränkt, unzulässig, degeneriert) live ausprobieren.
    4. **Cheat Sheet** *(5 Min.)* — Algorithmus + Sonderfälle auf einer Seite.
    5. **Selbständige Aufgaben** *(60 Min.)* — Tableaus von Hand bis zum Optimum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _titel = mo.md(r"""
    ---

    ## 1 · Wiederholung — Tableau & Pivotregel

    Kompakter Recap aus VL 05. Details werden in der geführten Übung gleich praktisch durchgespielt.
    """)

    _tableau = mo.md(r"""
    #### Tableau-Anatomie

    $$
    \begin{array}{l|cccc|c}
    \text{Basis} & x_1 & x_2 & s_1 & s_2 & \text{RHS} \\\hline
    s_1 & a_{11} & a_{12} & 1 & 0 & b_1 \\
    s_2 & a_{21} & a_{22} & 0 & 1 & b_2 \\\hline
    z & -c_1 & -c_2 & 0 & 0 & 0 \\
    \end{array}
    $$

    - **Schlupfvariablen** bilden Einheitsmatrix → **Startbasis** an Ecke $(0,0)$.
    - **z-Zeile** trägt die **negativen** Zielkoeffizienten (bei $\max$).
    - **RHS** = aktueller Wert der jeweiligen Basisvariablen bzw. von $z$.
    """)

    _pivot = mo.md(r"""
    #### Pivotregel (Dantzig)

    1. **Optimalitätstest**: Alle Einträge der $z$-Zeile $\geq 0$? → *fertig*.
    2. **Pivotspalte**: Spalte mit dem **negativsten** Eintrag in der $z$-Zeile.
       *„Welche Variable bringt am meisten Gewinn pro Einheit?"*
    3. **Pivotzeile (Min-Ratio-Test)**: $\displaystyle \min_i \left\{\frac{b_i}{a_{ij}} \,\Big|\, a_{ij} > 0\right\}$.
       *„Welche Ressource limitiert zuerst?"*
    4. **Pivotelement**: Schnittpunkt von Pivotspalte und Pivotzeile.
    5. **Pivotieren**: Pivotzeile durch Pivotelement teilen; alle anderen Zeilen so updaten,
       dass die Pivotspalte zum Einheitsvektor wird.
    """)

    _farben = mo.md(r"""
    #### Vom Problem zum Tableau

    **1. LP in Standardform** — Ungleichungen mit Schlupfvariablen $s_i \geq 0$ zu Gleichungen machen:

    $$
    \begin{aligned}
    \max\ z &= c_1 x_1 + c_2 x_2 \\
    a_{11} x_1 + a_{12} x_2 &\leq b_1 \quad\longrightarrow\quad a_{11} x_1 + a_{12} x_2 + s_1 = b_1 \\
    a_{21} x_1 + a_{22} x_2 &\leq b_2 \quad\longrightarrow\quad a_{21} x_1 + a_{22} x_2 + s_2 = b_2
    \end{aligned}
    $$

    **2. Startbasis ablesen** — die Schlupfvariablen bilden bereits eine
    Einheitsmatrix, also: Basis $= \{s_1, s_2\}$, Ecke $= (0, 0)$, $z = 0$.

    **3. Tableau aufstellen** — pro NB eine Zeile, jede Variable eine Spalte,
    RHS rechts. In die $z$-Zeile die **negativen** Zielkoeffizienten $-c_j$
    eintragen (damit der Optimalitätstest „$z$-Zeile $\geq 0$" greift).

    #### Sonderfälle (Vorschau)

    - **Unbeschränkt**: Pivotspalte hat *keine* positiven Einträge.
    - **Unzulässig**: nach Phase I bleibt eine künstliche Variable in der Basis.
    - **Degeneration**: Min-Ratio = 0 oder mehrfach minimal → Basiswechsel ohne ZF-Verbesserung.
    """)

    mo.vstack([
        _titel,
        mo.hstack([_tableau, _pivot], gap=2.0, widths="equal"),
        _farben,
    ], gap=0.6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2 · Geführte Übung — Brauerei Würzburg

    > Die **Brauerei Würzburg** plant die Wochenproduktion. Sie braut zwei
    > Sorten: **Helles** ($x_H$, in hl) und **Pils** ($x_P$, in hl). Pro hl
    > Helles erzielt sie **4 €** Deckungsbeitrag, pro hl Pils **6 €**.
    >
    > Zwei Ressourcen begrenzen die Produktion:
    > - **Malz**: 1 t/hl Helles, 2 t/hl Pils, **maximal 16 t** verfügbar.
    > - **Gärtanks**: 3 Tankstunden/hl Helles, 2 Tankstunden/hl Pils,
    >   **maximal 24 Tankstunden** verfügbar.
    >
    > Wie viele hl beider Sorten sollten gebraut werden, um den
    > Deckungsbeitrag zu maximieren?

    **LP-Formulierung** (Wiederholung aus UE 03):

    $$
    \begin{aligned}
    \max\ z \;=\;& 4\, x_H + 6\, x_P \\[0.2em]
    \text{s.t.}\;& x_H + 2\, x_P \leq 16 && \text{(Malz)} \\
    & 3\, x_H + 2\, x_P \leq 24 && \text{(Gärtanks)} \\
    & x_H,\ x_P \geq 0
    \end{aligned}
    $$

    **Standardform mit Schlupfvariablen** $s_M, s_G \geq 0$:

    $$
    \begin{aligned}
    x_H + 2\, x_P + s_M &= 16 \\
    3\, x_H + 2\, x_P + s_G &= 24
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    brauerei_stage = mo.ui.slider(
        start=0, stop=9, step=1, value=0,
        label="Schritt im Simplex-Algorithmus",
        show_value=True,
    )
    return (brauerei_stage,)


@app.cell(hide_code=True)
def _(mo):
    # Substep-Slider: nur in Pivotieren-Stages 4 & 8 relevant.
    # 0 = vor Pivotieren, 1 = Normieren, 2 = Eliminieren Datenzeile, 3 = z-Zeile fertig.
    brauerei_substep = mo.ui.slider(
        start=0, stop=3, step=1, value=0,
        label="Pivotieren-Substep (nur Schritt 4 & 8)",
        show_value=True,
    )
    return (brauerei_substep,)


@app.cell(hide_code=True)
def _(
    IMSBlue,
    IMSOrange,
    brauerei_stage,
    brauerei_substep,
    farbe_feasible,
    gruen,
    mo,
    render_flowchart,
    render_tableau_fig,
    rot,
):
    _stage = int(brauerei_stage.value)
    _substep = int(brauerei_substep.value)

    # 10 granulare Stages: jede Phase des Simplex-Algorithmus einzeln.
    # Tuple-Struktur:
    # (titel_md, snapshot_idx, pivot_col_or_None, pivot_row_or_None,
    #  highlight_z, flowchart_box, ecke_aktuell)
    _STAGES = [
        # Stage 0: Tableau aufstellen
        (r"**Schritt 0** — Tableau $T_0$ aufstellen, Basis $\{s_M, s_G\}$, Ecke $(0, 0)$",
         0, None, None, False, "tableau", (0, 0)),
        # Stage 1: Optimalität prüfen (T_0)
        (r"**Schritt 1** — Optimalitätsprüfung $T_0$: $z$-Zeile $(-4, -6, 0, 0)$ enthält negative Werte → **nicht optimal**",
         0, None, None, True, "optimal", (0, 0)),
        # Stage 2: Pivotspalte
        (r"**Schritt 2** — Pivotspalte bestimmen: $x_P$ (negativster Koeffizient $-6$ in der $z$-Zeile)",
         0, 1, None, False, "pivotspalte", (0, 0)),
        # Stage 3: Pivotzeile (col + row + element)
        (r"**Schritt 3** — Pivotzeile via Min-Ratio-Test: $s_M\!: 16/2 = 8$ vs. $s_G\!: 24/2 = 12$ → **$s_M$ tritt aus**, Pivotelement $a_{12} = 2$",
         0, 1, 0, False, "pivotzeile", (0, 0)),
        # Stage 4: Pivotieren → T_1
        (r"**Schritt 4** — Pivotieren: $x_P$ in die Basis, $s_M$ raus → **$T_1$**, Basis $\{x_P, s_G\}$, Ecke $(0, 8)$, $z = 48$",
         1, None, None, False, "pivotieren", (0, 8)),
        # Stage 5: Optimalität prüfen (T_1)
        (r"**Schritt 5** — Optimalitätsprüfung $T_1$: $z$-Zeile $(-1, 0, 3, 0)$ enthält $-1$ → **noch nicht optimal**",
         1, None, None, True, "optimal", (0, 8)),
        # Stage 6: Pivotspalte
        (r"**Schritt 6** — Pivotspalte bestimmen: $x_H$ (einziger negativer Koeffizient $-1$)",
         1, 0, None, False, "pivotspalte", (0, 8)),
        # Stage 7: Pivotzeile
        (r"**Schritt 7** — Pivotzeile via Min-Ratio-Test: $x_P\!: 8/\tfrac12 = 16$ vs. $s_G\!: 8/2 = 4$ → **$s_G$ tritt aus**, Pivotelement $a_{21} = 2$",
         1, 0, 1, False, "pivotzeile", (0, 8)),
        # Stage 8: Pivotieren → T_2
        (r"**Schritt 8** — Pivotieren: $x_H$ in die Basis, $s_G$ raus → **$T_2$**, Basis $\{x_P, x_H\}$, Ecke $(4, 6)$, $z = 52$",
         2, None, None, False, "pivotieren", (4, 6)),
        # Stage 9: Stop (Optimum)
        (r"**Schritt 9** — Optimalitätsprüfung $T_2$: $z$-Zeile $(0, 0, \tfrac52, \tfrac12)$ alle $\geq 0$ → **STOP — Optimum erreicht!**",
         2, None, None, True, "stop", (4, 6)),
    ]
    _titel, _snap_idx, _piv_col, _piv_row, _hl_z, _flow_box, _ecke = _STAGES[_stage]
    _titel_md = mo.md(_titel)

    # --- Hardcoded Brauerei-Tableaus ---
    _COL = ["x_H", "x_P", "s_M", "s_G"]
    _BR_TABS = [
        ([
            [1, 2, 1, 0, 16],
            [3, 2, 0, 1, 24],
            [-4, -6, 0, 0, 0],
        ], ["s_M", "s_G"]),
        ([
            [0.5, 1, 0.5, 0, 8],
            [2, 0, -1, 1, 8],
            [-1, 0, 3, 0, 48],
        ], ["x_P", "s_G"]),
        ([
            [0, 1, 0.75, -0.25, 6],
            [1, 0, -0.5, 0.5, 4],
            [0, 0, 2.5, 0.5, 52],
        ], ["x_P", "x_H"]),
    ]
    # --- Substep-Tableaus für Pivotieren (Stages 4 & 8) ---
    _PIV_SUBSTEPS = {
        4: [
            {
                "T": [[1, 2, 1, 0, 16], [3, 2, 0, 1, 24], [-4, -6, 0, 0, 0]],
                "basis": ["s_M", "s_G"],
                "pivot_row": 0, "pivot_col": 1, "hl_z": False,
                "msg": r"**Sub 0 — vor Pivotieren**: Pivotelement $a_{12} = 2$, Pivotzeile $s_M$, Pivotspalte $x_P$.",
            },
            {
                "T": [[0.5, 1, 0.5, 0, 8], [3, 2, 0, 1, 24], [-4, -6, 0, 0, 0]],
                "basis": ["x_P", "s_G"],
                "pivot_row": 0, "pivot_col": 1, "hl_z": False,
                "msg": r"**Sub 1 — Normieren**: Pivotzeile $\div 2$ → $(\tfrac12, 1, \tfrac12, 0, 8)$. Basisvariable wechselt $s_M \to x_P$.",
            },
            {
                "T": [[0.5, 1, 0.5, 0, 8], [2, 0, -1, 1, 8], [-4, -6, 0, 0, 0]],
                "basis": ["x_P", "s_G"],
                "pivot_row": 1, "pivot_col": 1, "hl_z": False,
                "msg": r"**Sub 2 — $s_G$-Zeile eliminieren**: $(3, 2, 0, 1, 24) - 2 \cdot (\tfrac12, 1, \tfrac12, 0, 8) = (2, 0, -1, 1, 8)$.",
            },
            {
                "T": [[0.5, 1, 0.5, 0, 8], [2, 0, -1, 1, 8], [-1, 0, 3, 0, 48]],
                "basis": ["x_P", "s_G"],
                "pivot_row": None, "pivot_col": 1, "hl_z": True,
                "msg": r"**Sub 3 — z-Zeile aktualisieren**: $(-4, -6, 0, 0, 0) + 6 \cdot (\tfrac12, 1, \tfrac12, 0, 8) = (-1, 0, 3, 0, 48)$. Spalte $x_P$ ist jetzt Einheitsvektor → **Pivot 1 fertig** ($T_1$).",
            },
        ],
        8: [
            {
                "T": [[0.5, 1, 0.5, 0, 8], [2, 0, -1, 1, 8], [-1, 0, 3, 0, 48]],
                "basis": ["x_P", "s_G"],
                "pivot_row": 1, "pivot_col": 0, "hl_z": False,
                "msg": r"**Sub 0 — vor Pivotieren**: Pivotelement $a_{21} = 2$, Pivotzeile $s_G$, Pivotspalte $x_H$.",
            },
            {
                "T": [[0.5, 1, 0.5, 0, 8], [1, 0, -0.5, 0.5, 4], [-1, 0, 3, 0, 48]],
                "basis": ["x_P", "x_H"],
                "pivot_row": 1, "pivot_col": 0, "hl_z": False,
                "msg": r"**Sub 1 — Normieren**: Pivotzeile $\div 2$ → $(1, 0, -\tfrac12, \tfrac12, 4)$. Basisvariable wechselt $s_G \to x_H$.",
            },
            {
                "T": [[0, 1, 0.75, -0.25, 6], [1, 0, -0.5, 0.5, 4], [-1, 0, 3, 0, 48]],
                "basis": ["x_P", "x_H"],
                "pivot_row": 0, "pivot_col": 0, "hl_z": False,
                "msg": r"**Sub 2 — $x_P$-Zeile eliminieren**: $(\tfrac12, 1, \tfrac12, 0, 8) - \tfrac12 \cdot (1, 0, -\tfrac12, \tfrac12, 4) = (0, 1, \tfrac34, -\tfrac14, 6)$.",
            },
            {
                "T": [[0, 1, 0.75, -0.25, 6], [1, 0, -0.5, 0.5, 4], [0, 0, 2.5, 0.5, 52]],
                "basis": ["x_P", "x_H"],
                "pivot_row": None, "pivot_col": 0, "hl_z": True,
                "msg": r"**Sub 3 — z-Zeile aktualisieren**: $(-1, 0, 3, 0, 48) + 1 \cdot (1, 0, -\tfrac12, \tfrac12, 4) = (0, 0, \tfrac52, \tfrac12, 52)$. **Optimum erreicht!**",
            },
        ],
    }

    # Tableau + Highlights wählen — mit Substep-Override für Stages 4 & 8
    if _stage in _PIV_SUBSTEPS:
        _sub = _PIV_SUBSTEPS[_stage][_substep]
        _T_data = _sub["T"]
        _basis_lbls = _sub["basis"]
        _piv_col = _sub["pivot_col"]
        _piv_row = _sub["pivot_row"]
        _hl_z = _sub["hl_z"]
        _substep_msg = _sub["msg"]
    else:
        _T_data, _basis_lbls = _BR_TABS[_snap_idx]
        _substep_msg = None

    _tab_fig = mo.as_html(render_tableau_fig(
        _T_data, _basis_lbls, _COL,
        pivot_col=_piv_col, pivot_row=_piv_row, highlight_z=_hl_z,
        cell_w=1.05, cell_h=0.8, fontsize=13,
    ))

    # --- Flowchart (mit aktuell aktivem Box) ---
    _flow_fig = mo.as_html(render_flowchart(current=_flow_box))

    # --- Pfeil-Geometrie: Bewegungsrichtung im Polytop ---
    # Stage → (start, end, mode) mit mode "preview" (gestrichelt, Richtung) oder "full" (durchgezogen, bis Ziel-Ecke)
    _STAGE_ARROWS = {
        2: ((0, 0), (0, 7.0), "preview"),    # x_P steigt von (0,0): senkrecht nach oben
        3: ((0, 0), (0, 8), "full"),         # Min-Ratio fest → bis Ecke (0,8)
        6: ((0, 8), (3.0, 6.5), "preview"),  # x_H steigt von (0,8): Richtung (1, -0.5)
        7: ((0, 8), (4, 6), "full"),         # Min-Ratio fest → bis Ecke (4,6)
    }
    _arrow = _STAGE_ARROWS.get(_stage)

    # --- 2D-Plot der zulässigen Menge ---
    def _render_brauerei():
        import matplotlib.pyplot as plt
        plt.close("all")
        if _stage <= 3:
            n_pfad = 1
        elif _stage <= 7:
            n_pfad = 2
        else:
            n_pfad = 3
        ecken_pfad = [(0, 0), (0, 8), (4, 6)]

        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, 10)
        ax.set_ylim(-0.5, 10)
        ax.set_xlabel(r"$x_H$ — hl Helles", fontsize=12)
        ax.set_ylabel(r"$x_P$ — hl Pils", fontsize=12)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axvline(x=0, color="black", linewidth=0.8)

        verts = [(0, 0), (8, 0), (4, 6), (0, 8)]
        ax.fill(
            [v[0] for v in verts] + [verts[0][0]],
            [v[1] for v in verts] + [verts[0][1]],
            color=farbe_feasible, alpha=0.55, label="Zulässige Menge", zorder=1,
        )

        ax.plot([0, 16], [8, 0], color=IMSBlue, linewidth=2.0,
                label=r"Malz: $x_H + 2 x_P \leq 16$")
        ax.plot([0, 8], [12, 0], color=gruen, linewidth=2.0,
                label=r"Gärtanks: $3 x_H + 2 x_P \leq 24$")

        # Iso-Gewinn-Linien z = 4 x_H + 6 x_P = k (analog VL „Geometrische Interpretation").
        # Schrittweise eingeblendet, passend zum Simplex-Pfad.
        def _iso(k, label_xy=None):
            # x_P = (k - 4 x_H) / 6
            _xs = [-1.0, 11.0]
            _ys = [(k - 4 * _x) / 6.0 for _x in _xs]
            ax.plot(_xs, _ys, color="#777", linewidth=1.3,
                    linestyle=":", alpha=0.85, zorder=2)
            if label_xy is not None:
                ax.annotate(rf"$z = {k}$", xy=label_xy, fontsize=10,
                            color="#555", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.18",
                                      facecolor="white",
                                      edgecolor="none", alpha=0.85),
                            zorder=12)

        if _stage >= 1:
            _iso(0, label_xy=(6.2, -0.35))
        if _stage >= 4:
            _iso(48, label_xy=(7.6, 2.85))
        if _stage >= 8:
            _iso(52, label_xy=(8.6, 2.85))

        # Bisheriger Simplex-Pfad (gestrichelt)
        if n_pfad >= 2:
            pfad_x = [v[0] for v in ecken_pfad[:n_pfad]]
            pfad_y = [v[1] for v in ecken_pfad[:n_pfad]]
            ax.plot(pfad_x, pfad_y, color=IMSOrange, linewidth=2.4,
                    linestyle="--", marker="o", markersize=8,
                    markeredgecolor="black", markerfacecolor=IMSOrange,
                    zorder=10, label="Simplex-Pfad")

        # Stage-spezifischer Bewegungs-Pfeil
        if _arrow is not None:
            (sx, sy), (ex, ey), mode = _arrow
            if mode == "preview":
                ax.annotate(
                    "", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#023B88",
                        lw=2.8,
                        linestyle=(0, (5, 3)),
                        mutation_scale=20,
                    ),
                    zorder=15,
                )
                _mid = ((sx + ex) / 2, (sy + ey) / 2)
                _label_txt = (r"$x_P\,\uparrow$" if _stage == 2
                              else r"$x_H\,\nearrow$ (Edge)")
                ax.annotate(
                    _label_txt, xy=_mid, xytext=(_mid[0] + 0.6, _mid[1]),
                    fontsize=12, fontweight="bold", color="#023B88",
                    zorder=16,
                )
            else:  # "full"
                ax.annotate(
                    "", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="#023B88",
                        lw=3.2,
                        mutation_scale=22,
                    ),
                    zorder=15,
                )

        cx, cy = _ecke
        ax.scatter([cx], [cy], s=320, color=rot, edgecolor="black",
                   linewidth=1.5, zorder=20)
        ax.annotate(f"  ({cx}, {cy})\n  $z = {4*cx + 6*cy}$",
                    xy=(cx, cy), xytext=(cx + 0.3, cy + 0.3),
                    fontsize=12, fontweight="bold", color=rot, zorder=21)

        labels = {(0, 0): "A", (8, 0): "B", (4, 6): "C", (0, 8): "D"}
        for (vx, vy), lab in labels.items():
            ax.annotate(lab, xy=(vx, vy), xytext=(vx - 0.45, vy - 0.65),
                        fontsize=12, fontweight="bold", color="black")

        ax.legend(loc="upper right", fontsize=10, framealpha=0.92)
        ax.set_title("Zulässige Menge & Simplex-Pfad", fontsize=13)
        plt.tight_layout(pad=0.3)
        return fig

    _fig = mo.as_html(_render_brauerei())

    # --- Stage-Detail-Text (LaTeX-fähig) ---
    _details = {
        0: r"""
        *Aktuelle Lösung:* $x_H = 0$, $x_P = 0$, $s_M = 16$, $s_G = 24$, $z = 0$.
        Startbasis sind die Schlupfvariablen — alle Ressourcen ungenutzt.
        """,
        1: r"""
        Wir lesen die $z$-Zeile: $(-4, -6, 0, 0)$ enthält negative Werte. Die
        Optimalitätsbedingung verlangt **alle Einträge $\geq 0$** — also weiter mit Pivotwahl.
        """,
        2: r"""
        Wir wählen die Spalte mit dem **negativsten Eintrag** in der $z$-Zeile: $-6$ unter $x_P$.
        Diese Variable bringt pro Einheit den größten Zuwachs an $z$.
        """,
        3: r"""
        Min-Ratio-Test entlang der Pivotspalte $x_P$:
        - $s_M$: $\;b_1 / a_{12} = 16 / 2 = 8$
        - $s_G$: $\;b_2 / a_{22} = 24 / 2 = 12$

        Minimum = 8 → **$s_M$ verlässt** die Basis. Pivotelement = $a_{12} = 2$.
        """,
        4: r"""
        Pivotzeile $\div\,2$, dann andere Zeilen so eliminieren, dass die $x_P$-Spalte
        zum Einheitsvektor wird. Wir landen bei Ecke $(0, 8)$ mit $z = 48$ — Malz ist jetzt bindend.
        """,
        5: r"""
        $z$-Zeile von $T_1$: $(-1, 0, 3, 0)$. Der Eintrag $-1$ unter $x_H$ zeigt:
        wir können $z$ noch verbessern, indem wir $x_H$ in die Basis nehmen.
        """,
        6: r"""
        $x_H$ ist die einzige Variable mit negativem $z$-Eintrag → automatisch Pivotspalte.
        """,
        7: r"""
        Min-Ratio-Test entlang der $x_H$-Spalte:
        - $x_P$: $\;8 / \tfrac12 = 16$
        - $s_G$: $\;8 / 2 = 4$

        Minimum = 4 → **$s_G$ verlässt** die Basis. Pivotelement = $a_{21} = 2$.
        """,
        8: r"""
        Nach dem zweiten Pivot landen wir an Ecke $(4, 6)$ mit $z = 52$.
        Beide ursprünglichen Ressourcen — Malz & Gärtanks — sind jetzt bindend.
        """,
        9: r"""
        Alle Einträge der $z$-Zeile $\geq 0$ → **Optimalitätsbedingung erfüllt, STOP**.

        Lösung: $x_H^\star = 4$ hl Helles, $x_P^\star = 6$ hl Pils, $z^\star = 52\,€$.
        Beide Ressourcen ($s_M = s_G = 0$) sind voll ausgelastet.
        *Die Werte $\tfrac52$ und $\tfrac12$ in der $z$-Zeile unter den Schlupfvariablen
        bekommen in VL 07 eine ökonomische Bedeutung — heute nur das Tableau-Handwerk.*
        """,
    }
    if _substep_msg is not None:
        _detail_md = mo.md(_substep_msg)
    else:
        _detail_md = mo.md(_details.get(_stage, ""))

    # Substep-Slider sichtbar nur in den Pivotieren-Stages
    if _stage in (4, 8):
        _substep_widget = brauerei_substep
    else:
        _substep_widget = mo.md(
            "_(Pivotieren-Substeps werden in den Schritten 4 und 8 aktiv —"
            " hier kannst du normieren / eliminieren / z-Zeile-Update einzeln nachvollziehen.)_"
        )

    _accordion = mo.accordion({
        "💡 Was passiert hier eigentlich? — Brücke Tableau ↔ Geometrie": mo.md(r"""
        Jede **Basis** im Tableau entspricht **genau einer Ecke** der zulässigen Menge:

        | Tableau | Basis | Geometrie | $z$ |
        |---|---|---|---|
        | $T_0$ | $\{s_M, s_G\}$  | Ecke $A = (0,\,0)$  | 0 |
        | $T_1$ | $\{x_P, s_G\}$  | Ecke $D = (0,\,8)$  | 48 |
        | $T_2$ | $\{x_P, x_H\}$  | Ecke $C = (4,\,6)$  | 52 |

        Ein **Pivotschritt** ist nichts anderes als eine **Bewegung von einer Ecke
        zu einer benachbarten Ecke** entlang einer Kante des Polyeders. Welche
        Kante? Die, in deren Richtung $z$ am stärksten wächst (→ Pivotspalte).
        Wie weit? Bis die nächste Nebenbedingung bindend wird (→ Min-Ratio-Test).

        Der Algorithmus läuft so lange, bis kein "besserer Nachbar" mehr existiert
        ($z$-Zeile $\geq 0$). Dank Konvexität des zulässigen Bereichs ist das
        gefundene **lokale Optimum auch das globale Optimum**.
        """)
    })

    _tab_col = mo.vstack([_tab_fig, _detail_md], gap=0.4)
    mo.vstack([
        brauerei_stage,
        _substep_widget,
        _titel_md,
        mo.hstack([_fig, _tab_col, _flow_fig],
                  gap=1.0, widths=[1.1, 1.2, 1.0], align="start"),
        _accordion,
    ], gap=0.6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3 · Spielwiese — LP-Baukasten

    Jetzt ihr! Klickt euch ein eigenes LP zusammen, prüft die Validität,
    und schaut dem Simplex Pivot für Pivot bei der Arbeit zu. **Probiert
    bewusst alle drei Sonderfälle aus**, die in VL 05 angesprochen wurden:

    | Sonderfall | So provoziert ihr ihn |
    |---|---|
    | **Unbeschränkt** | Kaum bindende NBs — Zielfunktion wächst ohne Limit. |
    | **Unzulässig**   | Widersprüchliche $\geq$- und $\leq$-NBs (z. B. $x_1 \geq 5$ und $x_1 \leq 2$). |
    | **Degeneriert**  | 3 NBs durch denselben Eckpunkt → Min-Ratio-Test mit Gleichstand. |

    *Tipp:* Defaults sind das **Klee-Minty-LP** (Worst Case für Simplex —
    $2^n - 1$ Pivots bei Dantzig-Regel). Erhöht $n_v$ schrittweise und
    schaut, wie die Anzahl der Stages explodiert!
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def loese_lp_mit_schritten(c, A, signs, b, sense="max", max_iter=60):
        """Zwei-Phasen-Simplex (wie in VL 05).

        Phase I (nur wenn ≥/=-NBs vorhanden):
          max  w = -∑ y_i   → zulässige Startbasis suchen.
        Phase II:
          Originalziel einsetzen, künstliche y_i sind im Folgenden vom
          Pivot ausgeschlossen.

        Snapshot-Felder: T, basis, pivot, note, var_labels, artif_cols, phase
          phase ∈ {"I", "II", "single"} — "single" = keine künstlichen Vars,
          nur eine Phase nötig.
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        signs = list(signs)

        if sense == "min":
            c = -c

        n_orig = len(c)
        m = len(b)

        # Negative RHS → Zeile mit -1 multiplizieren, Vorzeichen flippen.
        for i in range(m):
            if b[i] < 0:
                A[i] = -A[i]
                b[i] = -b[i]
                if signs[i] == "<=":
                    signs[i] = ">="
                elif signs[i] == ">=":
                    signs[i] = "<="

        # Spaltenzuteilung: x_1..x_n, dann s_i (per NB-Index, nur wenn ≤ oder ≥),
        # dann y_i (per NB-Index, nur wenn ≥ oder =).
        needs_s = [s in ("<=", ">=") for s in signs]
        needs_y = [s in (">=", "=") for s in signs]
        n_s = sum(needs_s)
        n_y = sum(needs_y)
        n_total = n_orig + n_s + n_y

        var_labels = [f"x_{j+1}" for j in range(n_orig)]
        s_col_for_nb = {}
        y_col_for_nb = {}
        col = n_orig
        for i in range(m):
            if needs_s[i]:
                s_col_for_nb[i] = col
                var_labels.append(f"s_{i+1}")
                col += 1
        for i in range(m):
            if needs_y[i]:
                y_col_for_nb[i] = col
                var_labels.append(f"y_{i+1}")
                col += 1

        T = np.zeros((m + 1, n_total + 1))
        basis = []
        artif_cols = []
        for i in range(m):
            T[i, :n_orig] = A[i]
            T[i, -1] = b[i]
            if signs[i] == "<=":
                T[i, s_col_for_nb[i]] = 1.0
                basis.append(s_col_for_nb[i])
            elif signs[i] == ">=":
                T[i, s_col_for_nb[i]] = -1.0
                T[i, y_col_for_nb[i]] = 1.0
                basis.append(y_col_for_nb[i])
                artif_cols.append(y_col_for_nb[i])
            else:  # "="
                T[i, y_col_for_nb[i]] = 1.0
                basis.append(y_col_for_nb[i])
                artif_cols.append(y_col_for_nb[i])

        snapshots = []

        def _pivot_inplace(pcol, prow):
            T[prow] = T[prow] / T[prow, pcol]
            for k in range(m + 1):
                if k != prow and abs(T[k, pcol]) > 1e-12:
                    T[k] = T[k] - T[k, pcol] * T[prow]

        def _setup_z_row(c_vec):
            T[-1, :-1] = -c_vec
            T[-1, -1] = 0.0
            for k, b_idx in enumerate(basis):
                if abs(T[-1, b_idx]) > 1e-12:
                    T[-1] -= T[-1, b_idx] * T[k]

        if artif_cols:
            # ===== Phase I: max w = -∑ y_i =====
            c_phase1 = np.zeros(n_total)
            for yc in artif_cols:
                c_phase1[yc] = -1.0
            _setup_z_row(c_phase1)

            snapshots.append({
                "T": T.copy(), "basis": list(basis), "pivot": None,
                "note": "Phase I — Start: künstliche Basis, $w = -\\sum y_i$.",
                "var_labels": list(var_labels),
                "artif_cols": list(artif_cols),
                "phase": "I",
            })

            for _ in range(max_iter):
                z_row = T[-1, :-1]
                if z_row.min() >= -1e-7:
                    break
                pivot_col = int(np.argmin(z_row))
                ratios = [(T[i, -1] / T[i, pivot_col], i)
                          for i in range(m) if T[i, pivot_col] > 1e-9]
                if not ratios:
                    return snapshots, "unbounded", var_labels
                min_r = min(r for r, _ in ratios)
                tied = [i for r, i in ratios if abs(r - min_r) < 1e-9]
                degenerate = (len(tied) > 1) or (min_r < 1e-9)
                pivot_row = min(tied)
                old_b = basis[pivot_row]
                _pivot_inplace(pivot_col, pivot_row)
                basis[pivot_row] = pivot_col

                note = (f"Phase I: {var_labels[pivot_col]} eintritt, "
                        f"{var_labels[old_b]} austritt"
                        + (" — degeneriert" if degenerate else ""))
                snapshots.append({
                    "T": T.copy(), "basis": list(basis),
                    "pivot": (pivot_row, pivot_col), "note": note,
                    "var_labels": list(var_labels),
                    "artif_cols": list(artif_cols),
                    "phase": "I",
                })

            # Phase I result: w* = -∑ y_i = T[-1, -1] (nach Reduktion).
            # Wenn w* < 0 (irgendein y_i > 0) → infeasible.
            sum_y = 0.0
            for k, b_idx in enumerate(basis):
                if b_idx in artif_cols:
                    sum_y += max(T[k, -1], 0.0)
            if sum_y > 1e-6:
                return snapshots, "infeasible", var_labels

            # ===== Phase II: Originalziel einsetzen =====
            c_phase2 = np.zeros(n_total)
            c_phase2[:n_orig] = c
            _setup_z_row(c_phase2)

            snapshots.append({
                "T": T.copy(), "basis": list(basis), "pivot": None,
                "note": "Phase II — Originalziel $z = c^\\top x$ eingesetzt.",
                "var_labels": list(var_labels),
                "artif_cols": list(artif_cols),
                "phase": "II",
            })
            phase_tag = "II"
        else:
            # Einphasig: natürliche Basis aus Schlupfvariablen.
            c_full = np.zeros(n_total)
            c_full[:n_orig] = c
            _setup_z_row(c_full)
            snapshots.append({
                "T": T.copy(), "basis": list(basis), "pivot": None,
                "note": "Initiales Tableau — Startbasis = Schlupfvariablen.",
                "var_labels": list(var_labels),
                "artif_cols": list(artif_cols),
                "phase": "single",
            })
            phase_tag = "single"

        # Phase II / single-phase Simplex
        for _ in range(max_iter):
            z_row = T[-1, :-1].copy()
            # In Phase II: künstliche Variablen vom Pivot ausschließen.
            for yc in artif_cols:
                z_row[yc] = 0.0
            if z_row.min() >= -1e-7:
                return snapshots, "optimal", var_labels

            pivot_col = int(np.argmin(z_row))
            ratios = [(T[i, -1] / T[i, pivot_col], i)
                      for i in range(m) if T[i, pivot_col] > 1e-9]
            if not ratios:
                return snapshots, "unbounded", var_labels
            min_r = min(r for r, _ in ratios)
            tied = [i for r, i in ratios if abs(r - min_r) < 1e-9]
            degenerate = (len(tied) > 1) or (min_r < 1e-9)
            pivot_row = min(tied)
            old_b = basis[pivot_row]
            _pivot_inplace(pivot_col, pivot_row)
            basis[pivot_row] = pivot_col

            note = (f"{phase_tag}: {var_labels[pivot_col]} eintritt, "
                    f"{var_labels[old_b]} austritt"
                    + (" — degeneriert" if degenerate else ""))
            snapshots.append({
                "T": T.copy(), "basis": list(basis),
                "pivot": (pivot_row, pivot_col), "note": note,
                "var_labels": list(var_labels),
                "artif_cols": list(artif_cols),
                "phase": phase_tag,
            })

        return snapshots, "iter_limit", var_labels

    def basislsg_dv(snapshot, n_orig):
        T = snapshot["T"]
        basis = snapshot["basis"]
        x = np.zeros(n_orig)
        for row, b_idx in enumerate(basis):
            if b_idx < n_orig:
                x[b_idx] = T[row, -1]
        return x
    return (loese_lp_mit_schritten,)


@app.cell(hide_code=True)
def _(IMSOrange, np):
    from fractions import Fraction

    def _fmt_val(v):
        if v is None:
            return ""
        if abs(v - round(v)) < 1e-7:
            return f"{int(round(v))}"
        try:
            f = Fraction(float(v)).limit_denominator(64)
            if f.denominator == 1:
                return f"{f.numerator}"
            sign = "-" if f.numerator < 0 else ""
            return rf"${sign}\frac{{{abs(f.numerator)}}}{{{f.denominator}}}$"
        except Exception:
            return f"{v:.3g}"

    def render_tableau_fig(T, basis_labels, col_labels,
                           pivot_row=None, pivot_col=None, highlight_z=False,
                           cell_w=0.9, cell_h=0.65, fontsize=11):
        """Rendert ein Simplex-Tableau als matplotlib-Figur.

        Highlight-Logik (kombinierbar):
        - pivot_col: int → Spalte blau
        - pivot_row: int → Zeile orange
        - beide → zusätzlich Pivotelement rot+fett
        - highlight_z: True → z-Zeile schwach grün (für Optimalitätsprüfung)
        """
        import matplotlib.pyplot as plt
        plt.close("all")

        T = np.array(T, dtype=float)
        m = T.shape[0] - 1
        n = T.shape[1] - 1
        grid_w = 2 + n
        grid_h = 2 + m

        fig_w = max(4.0, grid_w * cell_w + 0.3)
        fig_h = max(2.0, grid_h * cell_h + 0.3)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(0, grid_w)
        ax.set_ylim(0, grid_h)
        ax.invert_yaxis()
        ax.axis("off")

        HEADER = "#EFEFEF"
        PCOL = "#D6E2F4"
        PROW = "#FBE0C9"
        PEL = "#F2A29B"
        ZBG = "#D8F0D8"

        def cell(r, c, text, bg="white", bold=False):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=bg,
                                 edgecolor="#888", linewidth=1.0)
            ax.add_patch(rect)
            ax.text(c + 0.5, r + 0.5, text, ha="center", va="center",
                    fontsize=fontsize,
                    fontweight=("bold" if bold else "normal"))

        # Header
        cell(0, 0, "Basis", bg=HEADER, bold=True)
        for j, lab in enumerate(col_labels):
            bg = PCOL if (pivot_col is not None and j == pivot_col) else HEADER
            cell(0, j + 1, f"${lab}$", bg=bg, bold=True)
        cell(0, n + 1, "RHS", bg=HEADER, bold=True)

        # Daten
        for i in range(m):
            row_in = (pivot_row is not None and i == pivot_row)
            cell(i + 1, 0, f"${basis_labels[i]}$",
                 bg=(PROW if row_in else HEADER), bold=True)
            for j in range(n):
                val = _fmt_val(T[i, j])
                col_in = (pivot_col is not None and j == pivot_col)
                if row_in and col_in:
                    bg, bold = PEL, True
                elif col_in:
                    bg, bold = PCOL, False
                elif row_in:
                    bg, bold = PROW, False
                else:
                    bg, bold = "white", False
                cell(i + 1, j + 1, val, bg=bg, bold=bold)
            cell(i + 1, n + 1, _fmt_val(T[i, -1]),
                 bg=(PROW if row_in else "white"))

        # z-Zeile
        z_r = m + 1
        z_label_bg = ZBG if highlight_z else HEADER
        cell(z_r, 0, "$z$", bg=z_label_bg, bold=True)
        for j in range(n):
            col_in = (pivot_col is not None and j == pivot_col)
            if highlight_z and col_in:
                bg = PCOL
            elif highlight_z:
                bg = ZBG
            elif col_in:
                bg = PCOL
            else:
                bg = "white"
            cell(z_r, j + 1, _fmt_val(T[-1, j]), bg=bg)
        cell(z_r, n + 1, _fmt_val(T[-1, -1]),
             bg=(ZBG if highlight_z else "white"))

        ax.plot([0, grid_w], [z_r, z_r], color="black", linewidth=2.0)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

    def render_flowchart(current=None):
        """Simplex-Flowchart wie in VL 05, Folie 12 (mit roter Loop-Arrow).

        current: einer von "tableau", "optimal", "stop", "pivotspalte",
                 "pivotzeile", "pivotieren" oder None.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        plt.close("all")

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.set_xlim(0, 9)
        ax.set_ylim(0.2, 6.0)
        ax.axis("off")
        ax.set_aspect("equal")

        BW, BH = 2.0, 0.85
        IMSO = IMSOrange

        boxes = {
            "tableau":     (4.0, 5.4, "Tableau\naufstellen"),
            "optimal":     (4.0, 3.9, "Optimalität\nprüfen"),
            "stop":        (7.2, 3.9, "Stop"),
            "pivotspalte": (4.0, 2.4, "Pivotspalte\nbestimmen"),
            "pivotzeile":  (4.0, 0.9, "Pivotzeile\nbestimmen"),
            "pivotieren":  (1.2, 1.65, "Pivotieren"),
        }

        for name, (x, y, label) in boxes.items():
            is_cur = (name == current)
            face = IMSO if is_cur else "white"
            edge = IMSO if is_cur else "#333"
            txt = "white" if is_cur else "#222"
            lw = 2.6 if is_cur else 1.5
            rect = FancyBboxPatch(
                (x - BW / 2, y - BH / 2), BW, BH,
                boxstyle="round,pad=0.03,rounding_size=0.12",
                facecolor=face, edgecolor=edge, linewidth=lw,
                zorder=4,
            )
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=txt, zorder=5)

        def arr(p1, p2, color="#444", lw=1.6, rad=0.0, zorder=2):
            ax.add_patch(FancyArrowPatch(
                p1, p2,
                arrowstyle="-|>", mutation_scale=14,
                color=color, linewidth=lw,
                connectionstyle=f"arc3,rad={rad}",
                zorder=zorder,
            ))

        # Schwarze Pfeile
        arr((4.0, 5.4 - BH / 2), (4.0, 3.9 + BH / 2))
        arr((4.0 + BW / 2, 3.9), (7.2 - BW / 2, 3.9))
        ax.text(5.6, 4.15, "ja", fontsize=11, color="#444", ha="center")
        arr((4.0, 3.9 - BH / 2), (4.0, 2.4 + BH / 2))
        ax.text(4.18, 3.15, "nein", fontsize=11, color="#444", ha="left")
        arr((4.0, 2.4 - BH / 2), (4.0, 0.9 + BH / 2))

        # Rote Loop-Pfeile
        RED = "#C0392B"
        arr((4.0 - BW / 2, 0.9), (1.2 + BW / 2, 1.65),
            color=RED, lw=2.2)
        arr((1.2, 1.65 + BH / 2), (4.0 - BW / 2, 3.9),
            color=RED, lw=2.2, rad=-0.35)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return fig

    def render_progress_steps(steps, current):
        """Horizontale Schritt-Indikator-Bar mit hervorgehobenem aktuellen Schritt."""
        import matplotlib.pyplot as plt
        plt.close("all")
        n = len(steps)
        fig_w = max(6.0, 1.7 * n)
        fig, ax = plt.subplots(figsize=(fig_w, 1.5))

        ax.plot([0.5, n - 0.5], [0.62, 0.62],
                color="#CFCFCF", linewidth=3, zorder=1)

        for i, (top, sub) in enumerate(steps):
            is_cur = (i == current)
            face = IMSOrange if is_cur else "white"
            edge = IMSOrange if is_cur else "#888"
            txt_col = "white" if is_cur else "#555"
            circ = plt.Circle((i + 0.5, 0.62), 0.22, facecolor=face,
                              edgecolor=edge, linewidth=2.4, zorder=5)
            ax.add_patch(circ)
            ax.text(i + 0.5, 0.62, str(i), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color=txt_col, zorder=6)
            ax.text(i + 0.5, 0.20, top, ha="center", va="top",
                    fontsize=11,
                    fontweight=("bold" if is_cur else "normal"),
                    color=("black" if is_cur else "#444"))
            if sub:
                ax.text(i + 0.5, -0.05, sub, ha="center", va="top",
                        fontsize=9, color="#777")
        ax.set_xlim(0, n)
        ax.set_ylim(-0.4, 1.1)
        ax.axis("off")
        plt.tight_layout(pad=0.1)
        return fig
    return render_flowchart, render_tableau_fig


@app.cell(hide_code=True)
def _(mo):
    # SP_STATE — zentraler State für alle Spielwiese-Inputs (für Preset-Buttons)
    sp_get_n_var, sp_set_n_var = mo.state(4)
    sp_get_n_con, sp_set_n_con = mo.state(4)
    sp_get_sense, sp_set_sense = mo.state("max")
    sp_get_c, sp_set_c = mo.state([8.0, 4.0, 2.0, 1.0])
    sp_get_A, sp_set_A = mo.state([
        [1.0, 0.0, 0.0, 0.0],
        [4.0, 1.0, 0.0, 0.0],
        [8.0, 4.0, 1.0, 0.0],
        [16.0, 8.0, 4.0, 1.0],
        [32.0, 16.0, 8.0, 4.0],
    ])
    sp_get_b, sp_set_b = mo.state([5.0, 25.0, 125.0, 625.0, 3125.0])
    sp_get_signs, sp_set_signs = mo.state(["<=", "<=", "<=", "<=", "<="])
    sp_get_step, sp_set_step = mo.state(0)
    return (
        sp_get_A,
        sp_get_b,
        sp_get_c,
        sp_get_n_con,
        sp_get_n_var,
        sp_get_signs,
        sp_get_step,
        sp_set_A,
        sp_set_b,
        sp_set_c,
        sp_set_n_con,
        sp_set_n_var,
        sp_set_sense,
        sp_set_signs,
        sp_set_step,
    )


@app.cell(hide_code=True)
def _(
    mo,
    sp_set_A,
    sp_set_b,
    sp_set_c,
    sp_set_n_con,
    sp_set_n_var,
    sp_set_sense,
    sp_set_signs,
    sp_set_step,
):
    # SP_PRESETS — Buttons, die das LP auf einen Sonderfall setzen + Slider auf 0.

    def _pad(lst, length, fill=0.0):
        return list(lst) + [fill] * (length - len(lst))

    def _pad_signs(lst, length):
        return list(lst) + ["<="] * (length - len(lst))

    def _pad_A(rows, n_rows, n_cols):
        out = [_pad(r, 4) for r in rows]
        out += [[0.0] * 4] * (n_rows - len(out))
        return out

    PRESETS = {
        "klee_minty": {
            "n_var": 4, "n_con": 4, "sense": "max",
            "c": [8.0, 4.0, 2.0, 1.0],
            "A": [
                [1.0, 0.0, 0.0, 0.0],
                [4.0, 1.0, 0.0, 0.0],
                [8.0, 4.0, 1.0, 0.0],
                [16.0, 8.0, 4.0, 1.0],
                [32.0, 16.0, 8.0, 4.0],
            ],
            "b": [5.0, 25.0, 125.0, 625.0, 3125.0],
            "signs": ["<="] * 5,
        },
        "brauerei": {
            "n_var": 2, "n_con": 2, "sense": "max",
            "c": _pad([4.0, 6.0], 4),
            "A": _pad_A([[1.0, 2.0], [3.0, 2.0]], 5, 4),
            "b": _pad([16.0, 24.0], 5),
            "signs": ["<="] * 5,
        },
        "unbounded": {
            "n_var": 2, "n_con": 1, "sense": "max",
            "c": _pad([1.0, 1.0], 4),
            "A": _pad_A([[1.0, -1.0]], 5, 4),
            "b": _pad([5.0], 5),
            "signs": ["<="] * 5,
        },
        "infeasible": {
            "n_var": 2, "n_con": 2, "sense": "max",
            "c": _pad([1.0, 0.0], 4),
            "A": _pad_A([[1.0, 0.0], [1.0, 0.0]], 5, 4),
            "b": _pad([5.0, 2.0], 5),
            "signs": _pad_signs([">=", "<="], 5),
        },
        "degenerate": {
            "n_var": 2, "n_con": 3, "sense": "max",
            "c": _pad([1.0, 1.0], 4),
            "A": _pad_A([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], 5, 4),
            "b": _pad([4.0, 4.0, 4.0], 5),
            "signs": ["<="] * 5,
        },
        "equality": {
            "n_var": 2, "n_con": 2, "sense": "max",
            "c": _pad([1.0, 1.0], 4),
            "A": _pad_A([[1.0, 1.0], [1.0, 0.0]], 5, 4),
            "b": _pad([6.0, 4.0], 5),
            "signs": _pad_signs(["=", "<="], 5),
        },
    }

    def _apply(name):
        def _h(_):
            p = PRESETS[name]
            sp_set_n_var(p["n_var"])
            sp_set_n_con(p["n_con"])
            sp_set_sense(p["sense"])
            sp_set_c(list(p["c"]))
            sp_set_A([list(r) for r in p["A"]])
            sp_set_b(list(p["b"]))
            sp_set_signs(list(p["signs"]))
            sp_set_step(0)
        return _h

    btn_km = mo.ui.button(label="Preset 1", on_click=_apply("klee_minty"))
    btn_br = mo.ui.button(label="Preset 2", on_click=_apply("brauerei"))
    btn_un = mo.ui.button(label="Unbeschränkt", on_click=_apply("unbounded"))
    btn_in = mo.ui.button(label="Unzulässig", on_click=_apply("infeasible"))
    btn_de = mo.ui.button(label="Degeneriert", on_click=_apply("degenerate"))
    btn_eq = mo.ui.button(label="Gleichungs-NB", on_click=_apply("equality"))
    btn_reset_step = mo.ui.button(label="Reset Schritt",
                                  on_click=lambda _: sp_set_step(0))
    return btn_br, btn_de, btn_eq, btn_in, btn_km, btn_reset_step, btn_un


@app.cell(hide_code=True)
def _(
    mo,
    sp_get_A,
    sp_get_b,
    sp_get_c,
    sp_get_n_con,
    sp_get_n_var,
    sp_set_A,
    sp_set_b,
    sp_set_c,
    sp_set_n_con,
    sp_set_n_var,
    sp_set_step,
):
    # SP_UI1 — Dimensionen, gebunden an State (Optimierungsrichtung ist immer max).
    # Bei Vergrößerung von n_var/n_con werden die neu sichtbaren Slots mit
    # einer *dichten* Filler-Matrix befüllt (jeder Eintrag ≠ 0, damit neue
    # DVs auch in bestehenden NBs effektiv wirken). Dieser Filler produziert
    # bei (4,5) ca. 5 Pivots — Klee-Minty bleibt als eigenständiger Preset.
    DENSE_C = [5.0, 7.0, 6.0, 4.0]
    DENSE_A = [
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 1.0, 1.0, 3.0],
        [3.0, 2.0, 2.0, 1.0],
        [1.0, 3.0, 1.0, 2.0],
        [2.0, 1.0, 3.0, 2.0],
    ]
    DENSE_B = [18.0, 22.0, 25.0, 20.0, 24.0]

    def _on_nvar(new_val):
        new_nv = int(new_val)
        old_nv = int(sp_get_n_var())
        if new_nv > old_nv:
            cur_c = list(sp_get_c())
            for j in range(old_nv, new_nv):
                cur_c[j] = DENSE_C[j]
            sp_set_c(cur_c)
            cur_A = [list(r) for r in sp_get_A()]
            for i in range(len(cur_A)):
                for j in range(old_nv, new_nv):
                    cur_A[i][j] = DENSE_A[i][j]
            sp_set_A(cur_A)
        sp_set_n_var(new_nv)
        sp_set_step(0)

    def _on_ncon(new_val):
        new_nc = int(new_val)
        old_nc = int(sp_get_n_con())
        if new_nc > old_nc:
            cur_A = [list(r) for r in sp_get_A()]
            for i in range(old_nc, new_nc):
                cur_A[i] = list(DENSE_A[i])
            sp_set_A(cur_A)
            cur_b = list(sp_get_b())
            for i in range(old_nc, new_nc):
                cur_b[i] = DENSE_B[i]
            sp_set_b(cur_b)
        sp_set_n_con(new_nc)
        sp_set_step(0)

    n_var_slider = mo.ui.slider(
        start=2, stop=4, step=1, value=sp_get_n_var(),
        label="Anzahl DV", show_value=True,
        on_change=_on_nvar,
    )
    n_con_slider = mo.ui.slider(
        start=1, stop=5, step=1, value=sp_get_n_con(),
        label="Anzahl NB", show_value=True,
        on_change=_on_ncon,
    )
    return n_con_slider, n_var_slider


@app.cell(hide_code=True)
def _(
    mo,
    n_con_slider,
    n_var_slider,
    sp_get_A,
    sp_get_b,
    sp_get_c,
    sp_get_signs,
    sp_set_A,
    sp_set_b,
    sp_set_c,
    sp_set_signs,
):
    # SP_UI2 — Koeffizienten-Eingaben, gebunden an State (Pattern B-Variante).
    # Closures rufen die Getter zur Callback-Zeit auf (keine Stale-Snapshots).
    _nv = int(n_var_slider.value)
    _nc = int(n_con_slider.value)
    _c = sp_get_c()
    _A = sp_get_A()
    _b = sp_get_b()
    _s = sp_get_signs()

    def _make_c(j):
        def _h(v):
            cur = list(sp_get_c())
            cur[j] = float(v)
            sp_set_c(cur)
        return _h

    def _make_A(i, j):
        def _h(v):
            cur = [list(r) for r in sp_get_A()]
            cur[i][j] = float(v)
            sp_set_A(cur)
        return _h

    def _make_b(i):
        def _h(v):
            cur = list(sp_get_b())
            cur[i] = float(v)
            sp_set_b(cur)
        return _h

    def _make_sign(i):
        def _h(v):
            cur = list(sp_get_signs())
            cur[i] = v
            sp_set_signs(cur)
        return _h

    c_arr = mo.ui.array([
        mo.ui.number(value=_c[j], step=1.0, label=f"c_{j+1}",
                     on_change=_make_c(j))
        for j in range(_nv)
    ])
    A_arr = mo.ui.array([
        mo.ui.array([
            mo.ui.number(value=_A[i][j], step=1.0,
                         label=f"a_{i+1}{j+1}",
                         on_change=_make_A(i, j))
            for j in range(_nv)
        ])
        for i in range(_nc)
    ])
    signs_arr = mo.ui.array([
        mo.ui.dropdown(options=["<=", "=", ">="], value=_s[i],
                       on_change=_make_sign(i))
        for i in range(_nc)
    ])
    b_arr = mo.ui.array([
        mo.ui.number(value=_b[i], step=1.0, label=f"b_{i+1}",
                     on_change=_make_b(i))
        for i in range(_nc)
    ])
    return A_arr, b_arr, c_arr, signs_arr


@app.cell(hide_code=True)
def _(
    A_arr,
    b_arr,
    c_arr,
    loese_lp_mit_schritten,
    n_con_slider,
    n_var_slider,
    signs_arr,
):
    # SP_SOLVE — solve LP from UI values, returns snapshots (unsichtbar)
    n_v = int(n_var_slider.value)
    n_c = int(n_con_slider.value)
    sense = "max"
    c_vals = [float(v) for v in c_arr.value]
    A_vals = [[float(v) for v in row] for row in A_arr.value]
    sign_vals = list(signs_arr.value)
    b_vals = [float(v) for v in b_arr.value]
    snapshots, status, var_labels = loese_lp_mit_schritten(
        c_vals, A_vals, sign_vals, b_vals, sense=sense
    )
    return (
        A_vals,
        b_vals,
        c_vals,
        n_c,
        n_v,
        sense,
        sign_vals,
        snapshots,
        status,
        var_labels,
    )


@app.cell(hide_code=True)
def _(mo):
    # Spielwiese-Substep-Slider — nur für Pivotieren-Stages relevant.
    sp_substep_slider = mo.ui.slider(
        start=0, stop=3, step=1, value=0,
        label="Pivotieren-Substep (nur in Pivot-Schritten)",
        show_value=True,
    )
    return (sp_substep_slider,)


@app.cell(hide_code=True)
def _(mo, snapshots, sp_get_step, sp_set_step, status):
    # SP_STEP — granular Algorithmus-Schritt-Slider, an State gebunden.
    _n_pivots = max(0, len(snapshots) - 1)
    if status == "unbounded":
        _n_stages = 4 * _n_pivots + 4
    else:
        _n_stages = 4 * _n_pivots + 2
    _cur = max(0, min(int(sp_get_step()), _n_stages - 1))
    step_slider = mo.ui.slider(
        start=0, stop=max(0, _n_stages - 1), step=1, value=_cur,
        label=f"Algorithmus-Schritt (0–{_n_stages - 1})",
        show_value=True,
        on_change=sp_set_step,
    )
    return (step_slider,)


@app.cell(hide_code=True)
def _(
    A_arr,
    A_vals,
    b_arr,
    b_vals,
    btn_br,
    btn_de,
    btn_eq,
    btn_in,
    btn_km,
    btn_reset_step,
    btn_un,
    c_arr,
    c_vals,
    linprog,
    mo,
    n_c,
    n_con_slider,
    n_v,
    n_var_slider,
    np,
    render_flowchart,
    render_tableau_fig,
    sense,
    sign_vals,
    signs_arr,
    snapshots,
    sp_substep_slider,
    status,
    step_slider,
    var_labels,
):
    # SP_RENDER — sichtbare Spielwiese-Slide

    # ============== Cross-Check via scipy linprog (Status-Banner) ==============
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for _i, _sgn in enumerate(sign_vals):
        if _sgn == "<=":
            A_ub.append(A_vals[_i]); b_ub.append(b_vals[_i])
        elif _sgn == ">=":
            A_ub.append([-a for a in A_vals[_i]]); b_ub.append(-b_vals[_i])
        else:
            A_eq.append(A_vals[_i]); b_eq.append(b_vals[_i])
    c_obj = [-ci for ci in c_vals] if sense == "max" else list(c_vals)
    try:
        _res = linprog(
            c_obj,
            A_ub=A_ub if A_ub else None, b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None, b_eq=b_eq if b_eq else None,
            bounds=[(0, None)] * n_v, method="highs",
        )
        scipy_status = _res.status
        scipy_x = list(_res.x) if _res.x is not None else None
        scipy_z = ((-_res.fun if sense == "max" else _res.fun)
                   if _res.fun is not None else None)
    except Exception:
        scipy_status, scipy_x, scipy_z = -1, None, None

    if scipy_status == 2 or status == "infeasible":
        banner = mo.callout(
            mo.md(r"""**🚫 Unzulässig** — Nebenbedingungen widersprechen sich;
                  Phase I konnte die künstlichen Variablen nicht auf $0$ drücken
                  ($w^\star < 0$)."""),
            kind="danger",
        )
    elif scipy_status == 3 or status == "unbounded":
        banner = mo.callout(
            mo.md(r"""**♾️ Unbeschränkt** — Min-Ratio-Test schlägt fehl
                  (kein positiver Eintrag in Pivotspalte)."""),
            kind="warn",
        )
    elif scipy_status == 0 and scipy_x is not None:
        _x_str = ",\\ ".join(f"x_{j+1} = {v:.3g}"
                              for j, v in enumerate(scipy_x))
        banner = mo.callout(
            mo.md(rf"""**✅ Optimum** ({sense}): ${_x_str}$, $z = {scipy_z:.4g}$"""),
            kind="success",
        )
    else:
        banner = mo.callout(mo.md("Nicht klassifiziert."), kind="info")

    # ============== Stage-Generator ==============
    # Erzeugt eine Liste von Stages, je 1 pro Algorithmus-Schritt
    # (Tableau aufstellen → Optimalität → Pivotspalte → Pivotzeile → Pivotieren …).
    # Phasen-Tag (I/II) wird in jeden Stage-Titel übernommen (Zwei-Phasen-Verfahren).
    def _gen_stages():
        out = []
        n_snap = len(snapshots)

        # Stage 0: initial — Tag aus snapshots[0]["phase"]
        _ph0 = snapshots[0].get("phase", "single")
        if _ph0 == "I":
            _start_title = (r"**Phase I — Tableau aufstellen** mit künstlichen "
                            r"Variablen $y_i$, Hilfsziel $\max\ w = -\sum y_i$")
            _start_msg = (r"Phase I sucht eine zulässige Startbasis. Künstliche $y_i$ "
                          r"sind initial in der Basis; wir drücken sie auf $0$.")
        else:
            _start_title = r"**Tableau aufstellen** — Initialtableau, Startbasis = Schlupfvariablen"
            _start_msg = r"Startbasis: Schlupfvariablen $s_i$. Original-DVs $= 0$."
        out.append({
            "snap_idx": 0, "highlight_z": False,
            "piv_col": None, "piv_row": None,
            "flow": "tableau",
            "title": _start_title,
            "msg": _start_msg,
        })

        for k in range(n_snap):
            is_last = (k == n_snap - 1)
            _ph = snapshots[k].get("phase", "single")
            _ph_tag = (f" *(Phase {_ph})*" if _ph in ("I", "II") else "")

            # --- Phase-Übergang I → II als eigene Stage ---
            if (k > 0 and snapshots[k - 1].get("phase") == "I"
                    and _ph == "II"):
                out.append({
                    "snap_idx": k, "highlight_z": False,
                    "piv_col": None, "piv_row": None,
                    "flow": "tableau",
                    "title": r"**Phase I → Phase II** — Originalziel $z = c^\top x$ einsetzen",
                    "msg": (r"Phase I ist abgeschlossen ($w^\star = 0$): zulässige Basis gefunden. "
                            r"Jetzt $z$-Zeile durch das **Originalziel** ersetzen und Basis-Spalten "
                            r"wieder auf $0$ reduzieren. Künstliche $y_i$ dürfen nicht mehr eintreten."),
                })

            # --- Optimalitätsprüfung / Endstadien ---
            if is_last and status == "optimal":
                out.append({
                    "snap_idx": k, "highlight_z": True,
                    "piv_col": None, "piv_row": None,
                    "flow": "stop",
                    "title": rf"**Optimalitätsprüfung $T_{{{k}}}$**{_ph_tag} — alle $z$-Einträge $\geq 0$",
                    "msg": r"**Optimum erreicht — STOP.** Lösung in der RHS-Spalte ablesen.",
                })
                break
            if is_last and status == "infeasible":
                out.append({
                    "snap_idx": k, "highlight_z": True,
                    "piv_col": None, "piv_row": None,
                    "flow": "stop",
                    "title": rf"**Phase I beendet** — $w^\star < 0$, künstliche $y_i > 0$ in Basis",
                    "msg": (r"**UNZULÄSSIG**: Phase I konnte $\sum y_i$ nicht auf $0$ drücken — "
                            r"das Originalproblem hat keine zulässige Lösung."),
                })
                break
            if is_last and status not in ("unbounded",):
                out.append({
                    "snap_idx": k, "highlight_z": True,
                    "piv_col": None, "piv_row": None,
                    "flow": "stop",
                    "title": rf"**Optimalitätsprüfung $T_{{{k}}}$**{_ph_tag} — Iterationslimit erreicht",
                    "msg": "Algorithmus abgebrochen (Iterationslimit).",
                })
                break

            # Normaler Optimalitätscheck (nicht optimal — weiter)
            out.append({
                "snap_idx": k, "highlight_z": True,
                "piv_col": None, "piv_row": None,
                "flow": "optimal",
                "title": rf"**Optimalitätsprüfung $T_{{{k}}}$**{_ph_tag} — $z$-Zeile noch nicht $\geq 0$",
                "msg": r"Mindestens ein negativer Eintrag in der $z$-Zeile → weiter mit Pivotwahl.",
            })

            # --- Pivot col / row / pivotieren ---
            if k + 1 < n_snap:
                _next = snapshots[k + 1]
                if _next.get("pivot") is None:
                    # Phase-Übergang ohne Pivot — übersprungen (oben behandelt)
                    continue
                _piv_r, _piv_c = _next["pivot"]
                _enter = var_labels[_piv_c]
                _leave = var_labels[snapshots[k]["basis"][_piv_r]]

                out.append({
                    "snap_idx": k, "highlight_z": False,
                    "piv_col": _piv_c, "piv_row": None,
                    "flow": "pivotspalte",
                    "title": rf"**Pivotspalte bestimmen**{_ph_tag} — ${_enter}$ tritt ein",
                    "msg": rf"Spalte mit negativstem $z$-Koeffizient: ${_enter}$.",
                })
                out.append({
                    "snap_idx": k, "highlight_z": False,
                    "piv_col": _piv_c, "piv_row": _piv_r,
                    "flow": "pivotzeile",
                    "title": rf"**Pivotzeile bestimmen**{_ph_tag} — ${_leave}$ tritt aus, Pivotelement rot",
                    "msg": rf"Min-Ratio-Test: kleinstes $b_i / a_{{i,{_piv_c+1}}}$ in Zeile ${_leave}$.",
                })
                out.append({
                    "snap_idx": k + 1, "highlight_z": False,
                    "piv_col": None, "piv_row": None,
                    "flow": "pivotieren",
                    "title": rf"**Pivotieren**{_ph_tag} → $T_{{{k+1}}}$",
                    "msg": rf"Pivotzeile $\div$ Pivotelement; übrige Zeilen $\to$ Pivotspalte = Einheitsvektor. ${_enter}$ in Basis, ${_leave}$ raus.",
                })
            elif is_last and status == "unbounded":
                # Unbounded: zeige den fehlgeschlagenen Min-Ratio-Test
                _T = snapshots[k]["T"]
                _piv_c = int(np.argmin(_T[-1, :-1]))
                _enter = var_labels[_piv_c]
                out.append({
                    "snap_idx": k, "highlight_z": False,
                    "piv_col": _piv_c, "piv_row": None,
                    "flow": "pivotspalte",
                    "title": rf"**Pivotspalte bestimmen**{_ph_tag} — ${_enter}$",
                    "msg": rf"Wir würden ${_enter}$ in die Basis nehmen wollen.",
                })
                out.append({
                    "snap_idx": k, "highlight_z": False,
                    "piv_col": _piv_c, "piv_row": None,
                    "flow": "pivotzeile",
                    "title": r"**Min-Ratio-Test scheitert** → **UNBOUNDED**",
                    "msg": rf"Pivotspalte ${_enter}$ hat **keinen positiven Eintrag** — Zielfunktion wächst beliebig in dieser Richtung.",
                })
                break

        return out

    stages = _gen_stages()
    cur = max(0, min(int(step_slider.value), len(stages) - 1))
    s = stages[cur]
    snap = snapshots[s["snap_idx"]]

    # ============== Pivotieren-Substep-Logik ==============
    # Wenn aktuelle Stage "pivotieren" ist, baue intermediäres Tableau
    # je nach Substep-Slider (0=vor, 1=Normieren, 2=Eliminieren, 3=z-Update).
    _is_piv = (s["flow"] == "pivotieren") and (s["snap_idx"] >= 1)
    _sub = int(sp_substep_slider.value) if _is_piv else None
    _sub_msg = None
    if _is_piv:
        _T_after = snap["T"]
        _T_before = snapshots[s["snap_idx"] - 1]["T"]
        _basis_after = snap["basis"]
        _basis_before = snapshots[s["snap_idx"] - 1]["basis"]
        _piv_r, _piv_c = snap["pivot"]
        _enter_lbl = var_labels[_piv_c]
        _leave_lbl = var_labels[_basis_before[_piv_r]]

        if _sub == 0:
            _T_show = np.array(_T_before, dtype=float)
            _basis_show = list(_basis_before)
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _piv_r, False
            _sub_msg = (rf"**Sub 0 — vor Pivotieren**: Pivotelement = "
                        rf"$a_{{{_piv_r+1},{_piv_c+1}}}$, Pivotzeile ${_leave_lbl}$, "
                        rf"Pivotspalte ${_enter_lbl}$.")
        elif _sub == 1:
            _T_show = np.array(_T_before, dtype=float).copy()
            _T_show[_piv_r] = _T_after[_piv_r]
            _basis_show = list(_basis_before)
            _basis_show[_piv_r] = _basis_after[_piv_r]
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _piv_r, False
            _sub_msg = (rf"**Sub 1 — Normieren**: Pivotzeile $\div$ Pivotelement. "
                        rf"Basisvariable wechselt ${_leave_lbl} \to {_enter_lbl}$.")
        elif _sub == 2:
            _T_show = np.array(_T_before, dtype=float).copy()
            _T_show[_piv_r] = _T_after[_piv_r]
            _m = _T_show.shape[0] - 1
            for _ii in range(_m):
                if _ii != _piv_r:
                    _T_show[_ii] = _T_after[_ii]
            _basis_show = list(_basis_after)
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, None, False
            _sub_msg = (r"**Sub 2 — Eliminieren**: alle anderen Datenzeilen so updaten, "
                        r"dass die Pivotspalte zum Einheitsvektor wird "
                        r"($\text{Zeile}_i \mathrel{-}= a_{i,j} \cdot \text{Pivotzeile}$).")
        else:  # _sub == 3
            _T_show = _T_after
            _basis_show = list(_basis_after)
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, None, True
            _sub_msg = (r"**Sub 3 — z-Zeile aktualisieren**: $z$-Zeile so eliminieren, "
                        r"dass die Pivotspalte auch dort 0 wird → **Pivot fertig**.")
    else:
        _T_show = snap["T"]
        _basis_show = snap["basis"]
        _piv_col_show, _piv_row_show, _hl_z_show = (
            s["piv_col"], s["piv_row"], s["highlight_z"]
        )

    # ============== Tableau ==============
    _basis_lbls = [var_labels[idx] for idx in _basis_show]
    _col_lbls = var_labels[:np.array(_T_show).shape[1] - 1]
    tab_fig = mo.as_html(render_tableau_fig(
        _T_show, _basis_lbls, _col_lbls,
        pivot_row=_piv_row_show, pivot_col=_piv_col_show,
        highlight_z=_hl_z_show,
        cell_w=1.0, cell_h=0.75, fontsize=12,
    ))

    # ============== Flowchart ==============
    flow_fig = mo.as_html(render_flowchart(current=s["flow"]))

    # ============== Mitte: Tableau + Stage-Title + Detail ==============
    _detail_text = _sub_msg if _sub_msg is not None else s["msg"]
    _tab_col = mo.vstack([
        mo.md(s["title"]),
        tab_fig,
        mo.md(_detail_text),
    ], gap=0.4)

    # Substep-Slider sichtbar nur in Pivotieren-Stages
    if _is_piv:
        _substep_widget = sp_substep_slider
    else:
        _substep_widget = mo.md(
            "_(Pivotieren-Substep-Slider wird in Pivot-Schritten aktiv —"
            " dort kannst du Normieren / Eliminieren / z-Zeile einzeln nachvollziehen.)_"
        )

    # ============== Rechts: Problem-Settings ==============
    _zf_label = mo.md(f"#### Zielfunktion ({sense})")
    _zf_row = mo.hstack(list(c_arr), gap=0.5, widths="equal")
    _nb_label = mo.md("#### Nebenbedingungen")
    _nb_rows = []
    for _i in range(n_c):
        _nb_rows.append(
            mo.hstack(
                [*list(A_arr[_i]), signs_arr[_i], b_arr[_i]],
                gap=0.4, widths="equal", align="center",
            )
        )
    _preset_label = mo.md("### Presets *(Sonderfälle aus VL 05)*")
    _preset_row1 = mo.hstack([btn_km, btn_br, btn_reset_step],
                              gap=0.4, widths="equal", justify="start")
    _preset_row2 = mo.hstack([btn_un, btn_in, btn_de, btn_eq],
                              gap=0.4, widths="equal", justify="start")

    _controls_col = mo.vstack([
        _preset_label, _preset_row1, _preset_row2,
        mo.md("### LP-Spezifikation"),
        mo.hstack([n_var_slider, n_con_slider],
                  gap=0.6, widths="equal"),
        _zf_label, _zf_row,
        _nb_label, *_nb_rows,
        banner,
    ], gap=0.4)

    _suggestions = mo.accordion({
        "🧪 Vorschläge zum Ausprobieren": mo.md(r"""
        **A — Klee-Minty-Würfel (Default)** *(„worst case" für Simplex)*
        - $(2, 2)$: $\max\,8x_1 + 4x_2$, NB $x_1 \leq 5$, $4x_1 + x_2 \leq 25$ → **3 Pivots**
        - $(3, 3)$: 3-D-Variante → **7 Pivots**
        - $(4, 4)$: 4-D-Variante → **15 Pivots** $(= 2^4 - 1)$
        Mit Dantzig-Pivotregel besucht der Simplex **alle $2^n$ Ecken** des Würfels.

        **B — Unbeschränkt provozieren**
        NBs = 1, $a_{1,1} = 1$, $a_{1,2} = -1$, $\leq$, $b_1 = 5$.
        Zielfunktion $\max\ x_1 + x_2$. → 4 Schritte bis "Min-Ratio scheitert".

        **C — Unzulässig provozieren**
        DVs = 2, NBs = 2. NB1: $x_1 \geq 5$. NB2: $x_1 \leq 2$.
        → Banner + Stage-Titel zeigen "infeasible".

        **D — Degeneration provozieren**
        DVs = 2, NBs = 3. NB1: $x_1 + x_2 \leq 4$, NB2: $x_1 \leq 4$, NB3: $x_2 \leq 4$.
        Min-Ratio bekommt Gleichstände — Stage-Note markiert "degenerierter Schritt".

        **E — Gleichungs-NB** *(über Klausurstoff hinaus — VL 05 Zwei-Phasen)*
        Eine NB auf $=$ setzen (z. B. $x_1 + x_2 = 6$). Tableau zeigt
        künstliche Variable $y_1$ — Phase I drückt sie auf $0$, dann Phase II
        mit Originalziel.

        **F — Brauerei aus der geführten Übung**
        $c = (4, 6)$, NBs: $x_1 + 2x_2 \leq 16$ und $3x_1 + 2x_2 \leq 24$. → 2 Pivots, Optimum $(4, 6)$.
        """)
    })

    mo.vstack([
        step_slider,
        _substep_widget,
        mo.hstack([flow_fig, _tab_col, _controls_col],
                  gap=1.0, widths=[1.0, 1.2, 1.3], align="start"),
        _suggestions,
    ], gap=0.6)
    return


@app.cell(hide_code=True)
def _(mo, render_flowchart):
    _titel = mo.md(r"""
    ---

    ## 4 · Cheat Sheet — Simplex auf einer Seite
    """)

    _standardform = mo.md(r"""
    #### 1. Optimierungsproblem in **Standardform**

    Allgemeines LP:

    $$
    \begin{aligned}
    \max\ z \;=\;& \mathbf{c}^\top \mathbf{x} \\
    \text{s.t.}\;& \mathbf{A}\mathbf{x} \;\substack{\leq\\=\\\geq}\; \mathbf{b} \\
    & \mathbf{x} \geq \mathbf{0}
    \end{aligned}
    $$

    **Umformungs-Regeln** für Standardform $A\mathbf{x} = \mathbf{b}$:

    | NB-Typ | Erweiterung | Startbasis | Klausur? |
    |---|---|:---:|:---:|
    | $\sum a_{ij} x_j \leq b_i$ | $+\,s_i$ (Schlupf) | $s_i$ | ✅ |
    | $\sum a_{ij} x_j \geq b_i$ | $-\,s_i$ (Surplus) $+\,y_i$ (künstl.) | $y_i$ | ⚠️ |
    | $\sum a_{ij} x_j = b_i$ | $+\,y_i$ (künstl.) | $y_i$ | ⚠️ |

    > **Klausurrelevant:** nur ≤-NBs mit natürlicher Schlupfbasis.
    > Die beiden anderen Fälle sind in der Spielwiese illustriert (Zwei-Phasen-Verfahren),
    > werden aber **nicht in der Klausur abgefragt**.

    **Bei künstlichen Variablen — Zwei-Phasen-Verfahren**:

    - **Phase I:** $\max\ w = -\sum_i y_i$. Solange irgendein $y_i > 0$ in der Basis bleibt,
      ist das LP unzulässig.
    - **Phase II:** Originalziel $z = c^\top x$ einsetzen, $z$-Zeile mit Basis-Spalten
      konsistent machen, dann Standard-Simplex bis Optimum.

    Negative RHS? → Zeile mit $-1$ multiplizieren (Vorzeichen der NB dreht sich!).
    """)

    _flow_fig = mo.as_html(render_flowchart(current=None))
    _flow_card = mo.vstack([
        mo.md("#### 2. Simplex-**Algorithmus**"),
        _flow_fig,
        mo.md(r"""
        - **Pivotspalte**: $\arg\min_j (z_j)$ (negativster Eintrag in $z$-Zeile).
        - **Pivotzeile**: $\arg\min_i \{b_i / a_{ij} \mid a_{ij} > 0\}$ (Min-Ratio).
        - **Pivotieren**: Pivotzeile $\div$ Pivotelement; übrige Zeilen so updaten,
          dass Pivotspalte $\to$ Einheitsvektor.
        """),
    ], gap=0.3)

    _check = mo.md(r"""
    #### 3. **Endtableau** lesen

    | Größe | Wo? |
    |---|---|
    | Optimale DV-Werte $x_j^*$ | RHS der Zeilen, deren Basisvariable $x_j$ ist (sonst $x_j = 0$). |
    | Zielfunktionswert $z^*$ | RHS der $z$-Zeile. |
    | Bindende NBs | Zeilen mit Schlupf $s_i = 0$ (nicht in Basis). |
    | Schattenpreise | $z$-Zeilen-Einträge unter den Schlupfvariablen (folgt in VL 07). |
    """)

    _sonderf = mo.md(r"""
    #### 4. Sonderfall-Diagnose **am Tableau**

    | Beobachtung | Diagnose | Ursache |
    |---|---|---|
    | Pivotspalte hat **keinen** $a_{ij} > 0$ | **unbeschränkt** | ZF ohne Limit |
    | Phase I endet mit $w^\star < 0$ (irgendein $y_i > 0$) | **unzulässig** | widersprüchliche NBs |
    | Min-Ratio = 0 oder Gleichstand | **degeneriert** | mehrere NBs durch eine Ecke |

    **Pivot-Regeln im Überblick:**

    | Regel | Vorteil | Nachteil |
    |---|---|---|
    | **Dantzig** (negativster Koeffizient) | einfach, schnell in der Praxis | Zyklen möglich |
    | **Bland** (kleinster Variablenindex bei Gleichstand) | keine Zyklen | langsamer |
    | **Steepest Edge** | bessere Konvergenz | zusätzliche Rechnung pro Pivot |

    *Wir verwenden Dantzig + Bland als Anti-Cycling-Fallback.*
    """)

    mo.vstack([
        _titel,
        mo.hstack([_standardform, _flow_card],
                  gap=1.5, widths=[1.1, 1.0], align="start"),
        mo.hstack([_check, _sonderf], gap=1.5, widths="equal"),
    ], gap=0.6)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 2 · Aufgabe 2 *(2 DV, 3 NB)*

    Gegeben das LP:

    $$
    \begin{aligned}
    \max\ z \;=\;& 4\,x_1 + 3\,x_2 \\
    \text{s.t.}\;& x_1 + x_2 \leq 10 \\
    & 2\,x_1 + x_2 \leq 14 \\
    & x_1 \leq 6 \\
    & x_1,\ x_2 \geq 0
    \end{aligned}
    $$

    **(a)** Skizziert den zulässigen Bereich graphisch und findet das Optimum.

    **(b)** Stellt die Standardform auf und löst per Simplex von Hand.

    **(c)** Zeichnet den Pivot-Pfad in die Skizze ein.

    **Auf der nächsten Slide** könnt ihr die Lösung Schritt für Schritt aufbauen —
    erst den zulässigen Bereich, dann jeden Pivotschritt — und so eure
    Hand-Rechnung kontrollieren.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    aufgA_plot = mo.ui.slider(
        start=0, stop=4, step=1, value=0,
        label="Plot aufbauen (Schritt 0–4)",
        show_value=True,
    )
    return (aufgA_plot,)


@app.cell(hide_code=True)
def _(mo):
    # State für Substep, damit wir ihn bei sx-Wechsel auf 0 zurücksetzen können.
    aufgA_get_substep, aufgA_set_substep = mo.state(0)
    return aufgA_get_substep, aufgA_set_substep


@app.cell(hide_code=True)
def _(aufgA_set_substep, mo):
    # sx-Slider resettet automatisch den Substep auf 0 bei jeder Bewegung.
    aufgA_simplex = mo.ui.slider(
        start=0, stop=14, step=1, value=0,
        label="Simplex-Schritt (0–14)",
        show_value=True,
        on_change=lambda v: aufgA_set_substep(0),
    )
    return (aufgA_simplex,)


@app.cell(hide_code=True)
def _(aufgA_get_substep, aufgA_set_substep, mo):
    # 5 Substeps pro Pivot: vor / normieren / eliminate row A / eliminate row B / z-row
    aufgA_substep = mo.ui.slider(
        start=0, stop=4, step=1, value=aufgA_get_substep(),
        label="Pivot-Substep",
        show_value=False,
        on_change=aufgA_set_substep,
    )
    return (aufgA_substep,)


@app.cell(hide_code=True)
def _(
    IMSBlue,
    IMSOrange,
    aufgA_get_substep,
    aufgA_plot,
    aufgA_simplex,
    aufgA_substep,
    farbe_feasible,
    gruen,
    mo,
    render_flowchart,
    render_tableau_fig,
    rot,
):
    _plot_stage = int(aufgA_plot.value)
    _sx = int(aufgA_simplex.value)
    _sub = int(aufgA_get_substep())

    # Wenn Simplex läuft → Plot ist vollständig (alle Eckpunkte sichtbar)
    _effective_plot = 4 if _sx >= 1 else _plot_stage

    # 14 granulare Simplex-Stages (0-13), Schema wie Brauerei
    # (titel, snap_idx (T_k), piv_col, piv_row, hl_z, flow, ecke)
    _SX_STAGES = [
        (r"**Sx 0** — Initiales Tableau $T_0$ aufstellen, Basis $\{s_1, s_2, s_3\}$, Ecke $A = (0, 0)$",
         0, None, None, False, "tableau", (0, 0)),
        (r"**Sx 1** — Optimalitätsprüfung $T_0$: $z$-Zeile $(-4, -3, 0, 0, 0)$ — negative Werte → nicht optimal",
         0, None, None, True, "optimal", (0, 0)),
        (r"**Sx 2** — Pivotspalte $T_0$: $x_1$ (negativster Koeffizient $-4$)",
         0, 0, None, False, "pivotspalte", (0, 0)),
        (r"**Sx 3** — Pivotzeile via Min-Ratio: $s_1\!: 10$, $s_2\!: 7$, $s_3\!: 6$ → $s_3$ raus, Pivotelement $a_{31} = 1$",
         0, 0, 2, False, "pivotzeile", (0, 0)),
        (r"**Sx 4** — Pivotieren $T_0 \to T_1$: $x_1$ in Basis, $s_3$ raus. Ecke $B = (6, 0)$, $z = 24$",
         1, None, None, False, "pivotieren", (6, 0)),
        (r"**Sx 5** — Optimalitätsprüfung $T_1$: $z$-Zeile noch $-3$ unter $x_2$ → nicht optimal",
         1, None, None, True, "optimal", (6, 0)),
        (r"**Sx 6** — Pivotspalte $T_1$: $x_2$ (negativster Koeffizient $-3$)",
         1, 1, None, False, "pivotspalte", (6, 0)),
        (r"**Sx 7** — Pivotzeile via Min-Ratio: $s_1\!: 4$, $s_2\!: 2$, $x_1\!: -$ → $s_2$ raus, Pivotelement $a_{22} = 1$",
         1, 1, 1, False, "pivotzeile", (6, 0)),
        (r"**Sx 8** — Pivotieren $T_1 \to T_2$: $x_2$ in Basis, $s_2$ raus. Ecke $C = (6, 2)$, $z = 30$",
         2, None, None, False, "pivotieren", (6, 2)),
        (r"**Sx 9** — Optimalitätsprüfung $T_2$: $z$-Zeile noch $-2$ unter $s_3$ → nicht optimal",
         2, None, None, True, "optimal", (6, 2)),
        (r"**Sx 10** — Pivotspalte $T_2$: $s_3$ (negativster Koeffizient $-2$) — eine Schlupfvariable tritt ein!",
         2, 4, None, False, "pivotspalte", (6, 2)),
        (r"**Sx 11** — Pivotzeile via Min-Ratio: $s_1\!: 2$, $x_2\!: -$, $x_1\!: 6$ → $s_1$ raus, Pivotelement $a_{15} = 1$",
         2, 4, 0, False, "pivotzeile", (6, 2)),
        (r"**Sx 12** — Pivotieren $T_2 \to T_3$: $s_3$ in Basis, $s_1$ raus. Ecke $D = (4, 6)$, $z = 34$",
         3, None, None, False, "pivotieren", (4, 6)),
        (r"**Sx 13** — Optimalitätsprüfung $T_3$: alle $z$-Einträge $\geq 0$ → **optimal!**",
         3, None, None, True, "optimal", (4, 6)),
        (r"**Sx 14** — **STOP** — Optimum: $x_1 = 4$, $x_2 = 6$, $z^* = 34$",
         3, None, None, True, "stop", (4, 6)),
    ]

    # Tableaus (T_0 … T_3)
    _A_TABS = [
        ([[1,1,1,0,0,10],[2,1,0,1,0,14],[1,0,0,0,1,6],[-4,-3,0,0,0,0]],
         ["s_1","s_2","s_3"]),
        ([[0,1,1,0,-1,4],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,-3,0,0,4,24]],
         ["s_1","s_2","x_1"]),
        ([[0,0,1,-1,1,2],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,0,0,3,-2,30]],
         ["s_1","x_2","x_1"]),
        ([[0,0,1,-1,1,2],[0,1,2,-1,0,6],[1,0,-1,1,0,4],[0,0,2,1,0,34]],
         ["s_3","x_2","x_1"]),
    ]
    _COL_A = ["x_1", "x_2", "s_1", "s_2", "s_3"]

    # Pivotieren-Substeps für Stages 4, 8, 12 (5 Substeps pro Pivot)
    # Sub 0: vor / Sub 1: normieren / Sub 2: row A eliminieren /
    # Sub 3: row B eliminieren / Sub 4: z-Zeile
    _PIV_SUBS = {
        4: [
            {"T":[[1,1,1,0,0,10],[2,1,0,1,0,14],[1,0,0,0,1,6],[-4,-3,0,0,0,0]],
             "basis":["s_1","s_2","s_3"], "piv_row":2, "piv_col":0, "hl_z":False,
             "msg":r"**Sub 0 — vor Pivot 1**: Pivotelement $a_{31} = 1$, Pivotzeile $s_3$, Pivotspalte $x_1$."},
            {"T":[[1,1,1,0,0,10],[2,1,0,1,0,14],[1,0,0,0,1,6],[-4,-3,0,0,0,0]],
             "basis":["s_1","s_2","x_1"], "piv_row":2, "piv_col":0, "hl_z":False,
             "msg":r"**Sub 1 — Normieren**: Pivotzeile $\div 1$ (bleibt). Basisvariable $s_3 \to x_1$."},
            {"T":[[0,1,1,0,-1,4],[2,1,0,1,0,14],[1,0,0,0,1,6],[-4,-3,0,0,0,0]],
             "basis":["s_1","s_2","x_1"], "piv_row":0, "piv_col":0, "hl_z":False,
             "msg":r"**Sub 2 — $s_1$-Zeile**: $(1,1,1,0,0,10) - 1 \cdot (1,0,0,0,1,6) = (0,1,1,0,-1,4)$."},
            {"T":[[0,1,1,0,-1,4],[0,1,0,1,-2,2],[1,0,0,0,1,6],[-4,-3,0,0,0,0]],
             "basis":["s_1","s_2","x_1"], "piv_row":1, "piv_col":0, "hl_z":False,
             "msg":r"**Sub 3 — $s_2$-Zeile**: $(2,1,0,1,0,14) - 2 \cdot (1,0,0,0,1,6) = (0,1,0,1,-2,2)$."},
            {"T":[[0,1,1,0,-1,4],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,-3,0,0,4,24]],
             "basis":["s_1","s_2","x_1"], "piv_row":None, "piv_col":0, "hl_z":True,
             "msg":r"**Sub 4 — z-Zeile**: $(-4,-3,0,0,0,0) + 4 \cdot (1,0,0,0,1,6) = (0,-3,0,0,4,24)$. **Pivot 1 fertig** ($T_1$)."},
        ],
        8: [
            {"T":[[0,1,1,0,-1,4],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,-3,0,0,4,24]],
             "basis":["s_1","s_2","x_1"], "piv_row":1, "piv_col":1, "hl_z":False,
             "msg":r"**Sub 0 — vor Pivot 2**: Pivotelement $a_{22} = 1$, Pivotzeile $s_2$, Pivotspalte $x_2$."},
            {"T":[[0,1,1,0,-1,4],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,-3,0,0,4,24]],
             "basis":["s_1","x_2","x_1"], "piv_row":1, "piv_col":1, "hl_z":False,
             "msg":r"**Sub 1 — Normieren**: Pivotzeile bleibt. Basisvariable $s_2 \to x_2$."},
            {"T":[[0,0,1,-1,1,2],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,-3,0,0,4,24]],
             "basis":["s_1","x_2","x_1"], "piv_row":0, "piv_col":1, "hl_z":False,
             "msg":r"**Sub 2 — $s_1$-Zeile**: $(0,1,1,0,-1,4) - 1 \cdot (0,1,0,1,-2,2) = (0,0,1,-1,1,2)$."},
            {"T":[[0,0,1,-1,1,2],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,-3,0,0,4,24]],
             "basis":["s_1","x_2","x_1"], "piv_row":2, "piv_col":1, "hl_z":False,
             "msg":r"**Sub 3 — $x_1$-Zeile**: Koeffizient in Pivotspalte = 0, also $x_1$-Zeile bleibt unverändert."},
            {"T":[[0,0,1,-1,1,2],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,0,0,3,-2,30]],
             "basis":["s_1","x_2","x_1"], "piv_row":None, "piv_col":1, "hl_z":True,
             "msg":r"**Sub 4 — z-Zeile**: $(0,-3,0,0,4,24) + 3 \cdot (0,1,0,1,-2,2) = (0,0,0,3,-2,30)$. **Pivot 2 fertig** ($T_2$)."},
        ],
        12: [
            {"T":[[0,0,1,-1,1,2],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,0,0,3,-2,30]],
             "basis":["s_1","x_2","x_1"], "piv_row":0, "piv_col":4, "hl_z":False,
             "msg":r"**Sub 0 — vor Pivot 3**: Pivotelement $a_{15} = 1$, Pivotzeile $s_1$, Pivotspalte $s_3$."},
            {"T":[[0,0,1,-1,1,2],[0,1,0,1,-2,2],[1,0,0,0,1,6],[0,0,0,3,-2,30]],
             "basis":["s_3","x_2","x_1"], "piv_row":0, "piv_col":4, "hl_z":False,
             "msg":r"**Sub 1 — Normieren**: Pivotzeile bleibt. Basisvariable $s_1 \to s_3$."},
            {"T":[[0,0,1,-1,1,2],[0,1,2,-1,0,6],[1,0,0,0,1,6],[0,0,0,3,-2,30]],
             "basis":["s_3","x_2","x_1"], "piv_row":1, "piv_col":4, "hl_z":False,
             "msg":r"**Sub 2 — $x_2$-Zeile**: $(0,1,0,1,-2,2) + 2 \cdot (0,0,1,-1,1,2) = (0,1,2,-1,0,6)$."},
            {"T":[[0,0,1,-1,1,2],[0,1,2,-1,0,6],[1,0,-1,1,0,4],[0,0,0,3,-2,30]],
             "basis":["s_3","x_2","x_1"], "piv_row":2, "piv_col":4, "hl_z":False,
             "msg":r"**Sub 3 — $x_1$-Zeile**: $(1,0,0,0,1,6) - 1 \cdot (0,0,1,-1,1,2) = (1,0,-1,1,0,4)$."},
            {"T":[[0,0,1,-1,1,2],[0,1,2,-1,0,6],[1,0,-1,1,0,4],[0,0,2,1,0,34]],
             "basis":["s_3","x_2","x_1"], "piv_row":None, "piv_col":4, "hl_z":True,
             "msg":r"**Sub 4 — z-Zeile**: $(0,0,0,3,-2,30) + 2 \cdot (0,0,1,-1,1,2) = (0,0,2,1,0,34)$. **Optimum erreicht!**"},
        ],
    }

    # sx = 0..14, direkt _SX_STAGES[sx]. Tableau & Flowchart sind IMMER sichtbar.
    _stage_idx = max(0, min(_sx, len(_SX_STAGES) - 1))
    _titel_text, _snap_idx, _piv_col, _piv_row, _hl_z, _flow_box, _ecke = (
        _SX_STAGES[_stage_idx]
    )
    # Substep-Override in Pivotieren-Stages
    if _stage_idx in _PIV_SUBS:
        sub = _PIV_SUBS[_stage_idx][min(_sub, len(_PIV_SUBS[_stage_idx]) - 1)]
        _T_data = sub["T"]
        _basis_lbls = sub["basis"]
        _piv_col = sub["piv_col"]
        _piv_row = sub["piv_row"]
        _hl_z = sub["hl_z"]
        _detail_msg = sub["msg"]
    else:
        _T_data, _basis_lbls = _A_TABS[_snap_idx]
        _detail_msg = ""

    _titel_md = mo.md(_titel_text)
    _detail_md = mo.md(_detail_msg)

    # --- Plot ---
    def _plot_aufgA():
        import matplotlib.pyplot as plt
        plt.close("all")
        fig, ax = plt.subplots(figsize=(7.0, 6.5))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, 12)
        ax.set_ylim(-0.5, 12)
        ax.set_xlabel(r"$x_1$", fontsize=12)
        ax.set_ylabel(r"$x_2$", fontsize=12)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.axvline(x=0, color="black", linewidth=0.8)

        if _effective_plot >= 1:
            ax.plot([0, 10], [10, 0], color=IMSBlue, linewidth=2.0,
                    label=r"NB 1: $x_1 + x_2 \leq 10$")
        if _effective_plot >= 2:
            ax.plot([0, 7], [14, 0], color=gruen, linewidth=2.0,
                    label=r"NB 2: $2 x_1 + x_2 \leq 14$")
        if _effective_plot >= 3:
            ax.axvline(x=6, color="#8E44AD", linewidth=2.0, linestyle="-.",
                       label=r"NB 3: $x_1 \leq 6$")
            verts = [(0, 0), (6, 0), (6, 2), (4, 6), (0, 10)]
            ax.fill(
                [v[0] for v in verts] + [verts[0][0]],
                [v[1] for v in verts] + [verts[0][1]],
                color=farbe_feasible, alpha=0.55, zorder=1,
                label="Zulässige Menge",
            )
        if _effective_plot >= 4:
            labels = {(0, 0): "A", (6, 0): "B", (6, 2): "C",
                      (4, 6): "D", (0, 10): "E"}
            for (vx, vy), lab in labels.items():
                ax.scatter([vx], [vy], s=110, color="white",
                           edgecolor="black", linewidth=1.5, zorder=14)
                ax.annotate(lab, xy=(vx, vy),
                            xytext=(vx - 0.55, vy - 0.75),
                            fontsize=12, fontweight="bold", color="black")

        # Simplex-Pfad
        if _sx >= 1 and _ecke is not None:
            full_path = [(0, 0), (6, 0), (6, 2), (4, 6)]
            n_pfad = {(0,0):1, (6,0):2, (6,2):3, (4,6):4}.get(_ecke, 1)
            pfad = full_path[:n_pfad]
            if len(pfad) >= 2:
                ax.plot([p[0] for p in pfad], [p[1] for p in pfad],
                        color=IMSOrange, linewidth=2.6, linestyle="--",
                        marker="o", markersize=9,
                        markeredgecolor="black", markerfacecolor=IMSOrange,
                        zorder=10, label="Simplex-Pfad")
            cx, cy = _ecke
            ax.scatter([cx], [cy], s=320, color=rot, edgecolor="black",
                       linewidth=1.5, zorder=20)
            ax.annotate(f"  ({cx}, {cy})\n  $z = {4*cx + 3*cy}$",
                        xy=(cx, cy), xytext=(cx + 0.3, cy + 0.3),
                        fontsize=11, fontweight="bold",
                        color=rot, zorder=21)

        ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
        ax.set_title("Aufgabe 2 — Schrittweise lösen", fontsize=12)
        plt.tight_layout(pad=0.3)
        return fig

    _fig = mo.as_html(_plot_aufgA())

    # --- Tableau + Flowchart immer sichtbar ---
    _tab_fig = mo.as_html(render_tableau_fig(
        _T_data, _basis_lbls, _COL_A,
        pivot_col=_piv_col, pivot_row=_piv_row, highlight_z=_hl_z,
        cell_w=0.85, cell_h=0.65, fontsize=11,
    ))
    _flow_fig = mo.as_html(render_flowchart(current=_flow_box))

    # --- Substep-Slider sichtbar nur in Pivotieren-Stages 4, 8, 12 ---
    if _sx in (4, 8, 12):
        _sub_widget = aufgA_substep
    else:
        _sub_widget = mo.md("")

    # Layout: Slider-Reihe = hstack [plot | vstack(simplex, substep)]
    _slider_col = mo.vstack([aufgA_simplex, _sub_widget], gap=0.3)
    _slider_row = mo.hstack([aufgA_plot, _slider_col],
                             gap=1.0, widths="equal", align="start")

    _tab_col = mo.vstack([_tab_fig, _titel_md, _detail_md], gap=0.4)
    mo.vstack([
        _slider_row,
        mo.hstack([_fig, _tab_col, _flow_fig],
                  gap=1.0, widths=[1.2, 1.2, 1.0], align="start"),
    ], gap=0.5)
    return


@app.function(hide_code=True)
def gen_simplex_stages(snapshots, status, var_labels):
    """Liste von Stages (dict mit snap_idx, col, row, hl_z, flow, title)."""
    stages = []
    n_snap = len(snapshots)
    stages.append({
        "snap": 0, "col": None, "row": None, "hl_z": False, "flow": "tableau",
        "title": "Initiales Tableau $T_0$ aufstellen — Basis = Schlupfvariablen",
    })
    for k in range(n_snap):
        is_last = (k == n_snap - 1)
        if is_last and status == "optimal":
            stages.append({
                "snap": k, "col": None, "row": None, "hl_z": True, "flow": "optimal",
                "title": rf"Optimalitätsprüfung $T_{{{k}}}$: alle $z$-Einträge $\geq 0$ → **optimal!**",
            })
            stages.append({
                "snap": k, "col": None, "row": None, "hl_z": True, "flow": "stop",
                "title": r"**STOP** — Optimum erreicht. Lösung aus RHS-Spalte ablesen.",
            })
            break
        stages.append({
            "snap": k, "col": None, "row": None, "hl_z": True, "flow": "optimal",
            "title": rf"Optimalitätsprüfung $T_{{{k}}}$: $z$-Zeile noch nicht $\geq 0$ → weiter",
        })
        if k + 1 < n_snap:
            piv_r, piv_c = snapshots[k + 1]["pivot"]
            enter = var_labels[piv_c]
            leave = var_labels[snapshots[k]["basis"][piv_r]]
            stages.append({
                "snap": k, "col": piv_c, "row": None, "hl_z": False, "flow": "pivotspalte",
                "title": rf"Pivotspalte: ${enter}$ (negativster $z$-Koeffizient)",
            })
            stages.append({
                "snap": k, "col": piv_c, "row": piv_r, "hl_z": False, "flow": "pivotzeile",
                "title": rf"Pivotzeile via Min-Ratio: ${leave}$ tritt aus, Pivotelement rot",
            })
            stages.append({
                "snap": k + 1, "col": None, "row": None, "hl_z": False, "flow": "pivotieren",
                "title": rf"Pivotieren $T_{{{k}}} \to T_{{{k+1}}}$: ${enter}$ in Basis, ${leave}$ raus",
            })
    return stages


@app.cell(hide_code=True)
def _(mo):
    # State für Substep, damit wir ihn bei sx-Wechsel auf 0 zurücksetzen können.
    aufg3_get_substep, aufg3_set_substep = mo.state(0)
    return aufg3_get_substep, aufg3_set_substep


@app.cell(hide_code=True)
def _(aufg3_set_substep, mo):
    # sx-Slider resettet Substep auf 0 bei jeder Bewegung.
    # 15 Stages (1 init + 3*(1 opt + 3 pivot) + 2 final) → 0..14.
    aufg3_simplex = mo.ui.slider(
        start=0, stop=14, step=1, value=0,
        label="Simplex-Schritt", show_value=True,
        on_change=lambda v: aufg3_set_substep(0),
    )
    return (aufg3_simplex,)


@app.cell(hide_code=True)
def _(aufg3_get_substep, aufg3_set_substep, mo):
    # 5 Substeps pro Pivot: vor / normieren / eliminate row A / eliminate row B / z-row
    # (3 NBs → m=3 Datenzeilen → 2 Nicht-Pivot-Zeilen → m+2 = 5 Substeps).
    aufg3_substep = mo.ui.slider(
        start=0, stop=4, step=1, value=aufg3_get_substep(),
        label="Pivot-Substep", show_value=False,
        on_change=aufg3_set_substep,
    )
    return (aufg3_substep,)


@app.cell(hide_code=True)
def _(
    aufg3_get_substep,
    aufg3_simplex,
    aufg3_substep,
    loese_lp_mit_schritten,
    mo,
    np,
    render_flowchart,
    render_tableau_fig,
):
    # Dichtere LP-Matrix mit Variation der Koeffizienten ∈ {1, 2, 4} → nur
    # Halbe in den Zwischen-Tableaus. Optimum (1, 11/2, 0, 1/2), z = 65/2
    # (3 Pivots: Elemente 2, 1, 1).
    _c = [4, 5, 3, 2]
    _A = [[2, 1, 0, 1], [1, 2, 4, 0], [1, 1, 1, 1]]
    _b = [8, 12, 7]
    _snaps, _status, _vlabs = loese_lp_mit_schritten(_c, _A, ["<="] * 3, _b)
    _stages = gen_simplex_stages(_snaps, _status, _vlabs)
    _sx = max(0, min(int(aufg3_simplex.value), len(_stages) - 1))
    _s = _stages[_sx]

    # Pivotieren-Substep-Override
    _is_piv = (_s["flow"] == "pivotieren") and (_s["snap"] >= 1)
    _sub_msg = None
    if _is_piv:
        _snap_before = _snaps[_s["snap"] - 1]
        _snap_after = _snaps[_s["snap"]]
        _T_before = np.array(_snap_before["T"], dtype=float)
        _T_after = np.array(_snap_after["T"], dtype=float)
        _basis_before = _snap_before["basis"]
        _basis_after = _snap_after["basis"]
        _piv_r, _piv_c = _snap_after["pivot"]
        _enter_lbl = _vlabs[_piv_c]
        _leave_lbl = _vlabs[_basis_before[_piv_r]]
        _m = _T_before.shape[0] - 1
        _non_piv_rows = [r for r in range(_m) if r != _piv_r]
        _sub_max = _m + 1  # 0..m+1
        _sub_c = max(0, min(int(aufg3_get_substep()), _sub_max))

        if _sub_c == 0:
            _T_show = _T_before.copy()
            _basis_show = list(_basis_before)
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _piv_r, False
            _sub_msg = (rf"**Sub 0 — vor Pivotieren**: Pivotelement "
                        rf"$a_{{{_piv_r+1},{_piv_c+1}}}$, Pivotzeile ${_leave_lbl}$, "
                        rf"Pivotspalte ${_enter_lbl}$.")
        elif _sub_c == 1:
            _T_show = _T_before.copy()
            _T_show[_piv_r] = _T_after[_piv_r]
            _basis_show = list(_basis_before)
            _basis_show[_piv_r] = _basis_after[_piv_r]
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _piv_r, False
            _sub_msg = (rf"**Sub 1 — Normieren**: Pivotzeile $\div$ Pivotelement. "
                        rf"Basisvariable wechselt ${_leave_lbl} \to {_enter_lbl}$.")
        elif _sub_c <= _m:
            # Eliminiere eine Nicht-Pivot-Zeile pro Substep (in Reihenfolge).
            _T_show = _T_before.copy()
            _T_show[_piv_r] = _T_after[_piv_r]
            _basis_show = list(_basis_before)
            _basis_show[_piv_r] = _basis_after[_piv_r]
            _rows_done = _non_piv_rows[:_sub_c - 2]
            _cur_row = _non_piv_rows[_sub_c - 2]
            for _r in _rows_done:
                _T_show[_r] = _T_after[_r]
            _row_lbl = _vlabs[_basis_before[_cur_row]]
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _cur_row, False
            _sub_msg = (rf"**Sub {_sub_c} — ${_row_lbl}$-Zeile eliminieren**: "
                        rf"$\text{{Zeile}}_{{{_cur_row+1}}} \mathrel{{-}}= "
                        rf"a_{{{_cur_row+1},{_piv_c+1}}} \cdot \text{{Pivotzeile}}$.")
        else:
            _T_show = _T_after.copy()
            _basis_show = list(_basis_after)
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, None, True
            _sub_msg = (r"**Sub final — z-Zeile aktualisieren**: $z$-Zeile so "
                        r"eliminieren, dass die Pivotspalte auch dort $0$ wird → "
                        r"**Pivot fertig**.")
    else:
        _snap = _snaps[_s["snap"]]
        _T_show = _snap["T"]
        _basis_show = _snap["basis"]
        _piv_col_show = _s["col"]
        _piv_row_show = _s["row"]
        _hl_z_show = _s["hl_z"]

    _basis = [_vlabs[i] for i in _basis_show]
    _cols = _vlabs[:np.array(_T_show).shape[1] - 1]

    _tab_fig = mo.as_html(render_tableau_fig(
        _T_show, _basis, _cols,
        pivot_col=_piv_col_show, pivot_row=_piv_row_show, highlight_z=_hl_z_show,
        cell_w=0.7, cell_h=0.55, fontsize=11,
    ))
    _flow_fig = mo.as_html(render_flowchart(current=_s["flow"]))
    _titel_md = mo.md(f"**Sx {_sx}** — {_s['title']}")

    _heading = mo.md(r"### Aufgabe 3 *(4 DV, 3 NB)*")
    _lp_box = mo.md(r"""
    $$
    \begin{aligned}
    \max\ z \;=\;& 4 x_1 + 5 x_2 + 3 x_3 + 2 x_4 \\
    \text{s.t.}\;& 2 x_1 + x_2 + x_4 \leq 8 \\
    & x_1 + 2 x_2 + 4 x_3 \leq 12 \\
    & x_1 + x_2 + x_3 + x_4 \leq 7 \\
    & x_i \geq 0
    \end{aligned}
    $$
    """)

    # Slider-Reihe: kein Leer-Widget, nur die wirklich gebrauchten Slider.
    if _is_piv:
        _slider_row = mo.hstack([aufg3_simplex, aufg3_substep],
                                gap=0.5, widths="equal", align="start")
    else:
        _slider_row = aufg3_simplex

    # Detail-Text direkt unter Tableau (nur wenn vorhanden).
    _tab_parts = [_tab_fig, _titel_md]
    if _sub_msg:
        _tab_parts.append(mo.md(_sub_msg))
    _tab_col = mo.vstack(_tab_parts, gap=0.15)

    mo.vstack([
        _heading,
        _slider_row,
        mo.hstack([_lp_box, _tab_col, _flow_fig],
                  gap=0.3, widths=[0.9, 1.5, 1.0], align="start"),
    ], gap=0.15)
    return


@app.cell(hide_code=True)
def _(mo):
    aufg4_get_substep, aufg4_set_substep = mo.state(0)
    return aufg4_get_substep, aufg4_set_substep


@app.cell(hide_code=True)
def _(aufg4_set_substep, mo):
    # 4 Pivots → n_snap = 5 → 19 Stages (0..18).
    aufg4_simplex = mo.ui.slider(
        start=0, stop=18, step=1, value=0,
        label="Simplex-Schritt", show_value=True,
        on_change=lambda v: aufg4_set_substep(0),
    )
    return (aufg4_simplex,)


@app.cell(hide_code=True)
def _(aufg4_get_substep, aufg4_set_substep, mo):
    # 5 NBs → m = 5 Datenzeilen → 4 Nicht-Pivot-Zeilen → m+2 = 7 Substeps (0..6).
    aufg4_substep = mo.ui.slider(
        start=0, stop=6, step=1, value=aufg4_get_substep(),
        label="Pivot-Substep", show_value=False,
        on_change=aufg4_set_substep,
    )
    return (aufg4_substep,)


@app.cell(hide_code=True)
def _(
    aufg4_get_substep,
    aufg4_simplex,
    aufg4_substep,
    loese_lp_mit_schritten,
    mo,
    np,
    render_flowchart,
    render_tableau_fig,
):
    # Dichtere LP-Matrix mit Variation der Koeffizienten ∈ {1, 2, 4} →
    # Halbe, Viertel und vereinzelt Achtel in den Zwischen-Tableaus.
    # Optimum (5/2, 1, 9/4, 3/2), z = 139/4 (4 Pivots: Elemente 4, 2, 1, 1).
    _c = [8, 5, 3, 2]
    _A = [
        [4, 2, 0, 0],
        [1, 1, 2, 0],
        [0, 1, 1, 1],
        [0, 0, 2, 1],
        [1, 1, 0, 1],
    ]
    _b = [12, 8, 6, 6, 5]
    _snaps, _status, _vlabs = loese_lp_mit_schritten(_c, _A, ["<="] * 5, _b)
    _stages = gen_simplex_stages(_snaps, _status, _vlabs)
    _sx = max(0, min(int(aufg4_simplex.value), len(_stages) - 1))
    _s = _stages[_sx]

    _is_piv = (_s["flow"] == "pivotieren") and (_s["snap"] >= 1)
    _sub_msg = None
    if _is_piv:
        _snap_before = _snaps[_s["snap"] - 1]
        _snap_after = _snaps[_s["snap"]]
        _T_before = np.array(_snap_before["T"], dtype=float)
        _T_after = np.array(_snap_after["T"], dtype=float)
        _basis_before = _snap_before["basis"]
        _basis_after = _snap_after["basis"]
        _piv_r, _piv_c = _snap_after["pivot"]
        _enter_lbl = _vlabs[_piv_c]
        _leave_lbl = _vlabs[_basis_before[_piv_r]]
        _m = _T_before.shape[0] - 1
        _non_piv_rows = [r for r in range(_m) if r != _piv_r]
        _sub_max = _m + 1
        _sub_c = max(0, min(int(aufg4_get_substep()), _sub_max))

        if _sub_c == 0:
            _T_show = _T_before.copy()
            _basis_show = list(_basis_before)
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _piv_r, False
            _sub_msg = (rf"**Sub 0 — vor Pivotieren**: Pivotelement "
                        rf"$a_{{{_piv_r+1},{_piv_c+1}}}$, Pivotzeile ${_leave_lbl}$, "
                        rf"Pivotspalte ${_enter_lbl}$.")
        elif _sub_c == 1:
            _T_show = _T_before.copy()
            _T_show[_piv_r] = _T_after[_piv_r]
            _basis_show = list(_basis_before)
            _basis_show[_piv_r] = _basis_after[_piv_r]
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _piv_r, False
            _sub_msg = (rf"**Sub 1 — Normieren**: Pivotzeile $\div$ Pivotelement. "
                        rf"Basisvariable wechselt ${_leave_lbl} \to {_enter_lbl}$.")
        elif _sub_c <= _m:
            _T_show = _T_before.copy()
            _T_show[_piv_r] = _T_after[_piv_r]
            _basis_show = list(_basis_before)
            _basis_show[_piv_r] = _basis_after[_piv_r]
            _rows_done = _non_piv_rows[:_sub_c - 2]
            _cur_row = _non_piv_rows[_sub_c - 2]
            for _r in _rows_done:
                _T_show[_r] = _T_after[_r]
            _row_lbl = _vlabs[_basis_before[_cur_row]]
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, _cur_row, False
            _sub_msg = (rf"**Sub {_sub_c} — ${_row_lbl}$-Zeile eliminieren**: "
                        rf"$\text{{Zeile}}_{{{_cur_row+1}}} \mathrel{{-}}= "
                        rf"a_{{{_cur_row+1},{_piv_c+1}}} \cdot \text{{Pivotzeile}}$.")
        else:
            _T_show = _T_after.copy()
            _basis_show = list(_basis_after)
            _piv_col_show, _piv_row_show, _hl_z_show = _piv_c, None, True
            _sub_msg = (r"**Sub final — z-Zeile aktualisieren**: $z$-Zeile so "
                        r"eliminieren, dass die Pivotspalte auch dort $0$ wird → "
                        r"**Pivot fertig**.")
    else:
        _snap = _snaps[_s["snap"]]
        _T_show = _snap["T"]
        _basis_show = _snap["basis"]
        _piv_col_show = _s["col"]
        _piv_row_show = _s["row"]
        _hl_z_show = _s["hl_z"]

    _basis = [_vlabs[i] for i in _basis_show]
    _cols = _vlabs[:np.array(_T_show).shape[1] - 1]

    _tab_fig = mo.as_html(render_tableau_fig(
        _T_show, _basis, _cols,
        pivot_col=_piv_col_show, pivot_row=_piv_row_show, highlight_z=_hl_z_show,
        cell_w=0.62, cell_h=0.48, fontsize=10,
    ))
    _flow_fig = mo.as_html(render_flowchart(current=_s["flow"]))
    _titel_md = mo.md(f"**Sx {_sx}** — {_s['title']}")

    _heading = mo.md(r"### Aufgabe 4 *(4 DV, 5 NB)*")
    _lp_box = mo.md(r"""
    $$
    \begin{aligned}
    \max\ z \;=\;& 8 x_1 + 5 x_2 + 3 x_3 + 2 x_4 \\
    \text{s.t.}\;& 4 x_1 + 2 x_2 \leq 12 \\
    & x_1 + x_2 + 2 x_3 \leq 8 \\
    & x_2 + x_3 + x_4 \leq 6 \\
    & 2 x_3 + x_4 \leq 6 \\
    & x_1 + x_2 + x_4 \leq 5 \\
    & x_i \geq 0
    \end{aligned}
    $$
    """)

    if _is_piv:
        _slider_row = mo.hstack([aufg4_simplex, aufg4_substep],
                                gap=0.5, widths="equal", align="start")
    else:
        _slider_row = aufg4_simplex

    _tab_parts = [_tab_fig, _titel_md]
    if _sub_msg:
        _tab_parts.append(mo.md(_sub_msg))
    _tab_col = mo.vstack(_tab_parts, gap=0.15)

    mo.vstack([
        _heading,
        _slider_row,
        mo.hstack([_lp_box, _tab_col, _flow_fig],
                  gap=0.3, widths=[0.9, 1.6, 1.0], align="start"),
    ], gap=0.15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6 · Was kommt als Nächstes

    - **VL 06: Modellierung & Python (PuLP).** Wir verlassen das Tableau
      und überlassen die Pivotschritte einem Solver. Eure Aufgabe wird nur
      noch das **Modellieren** sein.
    - **UE 05: PuLP Hands-on.** Heutiges Brauerei-LP in 5 Zeilen Code,
      dann Transport- und Personalmodelle in beliebiger Größe.
    - **VL 07 (Vorausschau):** Die Brüche in der $z$-Zeile des Endtableaus
      ($\tfrac52, \tfrac12$ bei der Brauerei) sind die **Schattenpreise** —
      wir werden sie als zentralen ökonomischen Output des Simplex
      kennenlernen.

    ### Take-Aways heute

    1. Jede Basis = eine Ecke. Pivotschritt = Bewegung zur Nachbarecke.
    2. Pivotspalte (Dantzig: negativster Koeff.), Pivotzeile (Min-Ratio).
    3. Stop-Kriterium: $z$-Zeile $\geq 0$.
    4. Sonderfälle erkennt man **am Tableau** — nicht erst am Solver-Status.
    """)
    return


if __name__ == "__main__":
    app.run()
