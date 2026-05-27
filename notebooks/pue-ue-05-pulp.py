# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pulp",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.18.3"
app = marimo.App(
    width="full",
    app_title="PuE Übung 5: PuLP Hands-on",
)


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import pandas as pd
    import pulp as pl
    return mo, pd, pl


@app.cell(hide_code=True)
def helpers(mo, pl):
    import tempfile

    def _solve_with_log(prob):
        """Löst `prob` und gibt (raised_exception_or_None, cbc_log_text) zurück."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as fh:
            log_path = fh.name
        try:
            prob.solve(pl.PULP_CBC_CMD(msg=0, logPath=log_path))
            err = None
        except Exception as exc:
            err = exc
        try:
            with open(log_path) as fh:
                log_text = fh.read()
        except OSError:
            log_text = "(kein CBC-Log verfügbar)"
        return err, log_text

    def _solver_log_panel(log_text):
        return mo.accordion({
            "🔬 CBC-Solver-Output anzeigen": mo.md(
                f"```\n{log_text.strip() or '(leer)'}\n```"
            )
        })

    def check_solution(prob, expected_obj=None, label=""):
        """Solve a student-built PuLP problem and render a feedback panel.

        Robust gegen typische Anfängerfehler: keine Zielfunktion, keine
        Restriktionen, Solver-Exception, Infeasible/Unbounded. Wenn
        `expected_obj` angegeben ist, vergleicht es zusätzlich mit dem
        erwarteten Zielfunktionswert. Der CBC-Solver-Output wird in einem
        aufklappbaren Panel angezeigt.
        """
        head = f"### Auswertung — {label}" if label else "### Auswertung"

        if prob is None:
            return mo.md(f"{head}\n\n⚠️ **`prob` wurde nicht erstellt** — "
                         "stellt sicher, dass die Funktion ein `pl.LpProblem` zurückgibt.")

        if prob.objective is None:
            return mo.md(f"{head}\n\n⚠️ **Noch keine Zielfunktion** — "
                         "fügt `prob += <Ausdruck>` (ohne Vergleichsoperator) hinzu.")

        if len(prob.constraints) == 0:
            return mo.md(f"{head}\n\n⚠️ **Keine Restriktionen** — "
                         "fügt `prob += <Ausdruck> <= <Wert>, \"Name\"` hinzu.")

        err, log_text = _solve_with_log(prob)
        log_panel = _solver_log_panel(log_text)

        if err is not None:
            return mo.vstack([
                mo.md(f"{head}\n\n❌ **Solver-Fehler:** `{type(err).__name__}: {err}`"),
                log_panel,
            ])

        status = pl.LpStatus[prob.status]

        if status == "Infeasible":
            return mo.vstack([
                mo.md(f"{head}\n\n❌ **Status: Infeasible** — die Constraints widersprechen sich. "
                      "Tipp: Vorzeichen prüfen, Bounds checken, `prob.writeLP('debug.lp')`."),
                log_panel,
            ])
        if status == "Unbounded":
            return mo.vstack([
                mo.md(f"{head}\n\n❌ **Status: Unbounded** — die Zielfunktion wächst unbegrenzt. "
                      "Tipp: fehlt eine Kapazitäts-Restriktion?"),
                log_panel,
            ])
        if status != "Optimal":
            return mo.vstack([
                mo.md(f"{head}\n\n❌ **Status: {status}** — Modell wurde nicht optimal gelöst."),
                log_panel,
            ])

        obj = pl.value(prob.objective)
        if expected_obj is not None and abs(obj - expected_obj) < 1e-2:
            head_obj = (f"✅ **Status:** `Optimal` · **Zielfunktionswert:** "
                        f"`{obj:.2f}` — entspricht der Erwartung.")
        elif expected_obj is not None:
            head_obj = (f"⚠️ **Status:** `Optimal`, aber **Zielfunktionswert** `{obj:.2f}` "
                        f"weicht von der Erwartung `{expected_obj:.2f}` ab. "
                        "Constraints / Koeffizienten nochmal prüfen.")
        else:
            head_obj = f"✅ **Status:** `Optimal` · **Zielfunktionswert:** `{obj:.2f}`"

        var_rows = "\n".join(
            f"| `{v.name}` | {v.value():.2f} |" for v in prob.variables()
        )
        con_rows = "\n".join(
            f"| `{n}` | {c.slack:.2f} | {c.pi:.2f} |"
            for n, c in prob.constraints.items()
        )

        body = "\n".join([
            head,
            "",
            head_obj,
            "",
            f"**Variablen** ({len(prob.variables())})",
            "",
            "| Name | Wert |",
            "|---|---|",
            var_rows,
            "",
            f"**Constraints** ({len(prob.constraints)}) — Slack & Schattenpreis π",
            "",
            "| Name | Slack | π |",
            "|---|---|---|",
            con_rows,
        ])
        return mo.vstack([mo.md(body), log_panel])
    return (check_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Übung 5 · PuLP Hands-on

    **Planung und Entscheidung — SS 2026 · Begleitnotebook zu VL 06**

    > **Wichtig:** Wir arbeiten in diesem Notebook im **Code-View** (oben links
    > umschalten falls nötig). Ihr sollt nicht nur lesen, sondern direkt im
    > **PuLP-Code** mitmachen — Werte verändern, Constraints hinzufügen, eigene
    > Modelle schreiben.

    Aufbau:

    1. **Teil A — Lecture-Demos:** Die Codebeispiele aus VL 06 zum selbst Laufen
       lassen und Ausprobieren.
    2. **Teil B — Eigene Modelle:** Drei Aufgaben — jeweils
       *(i)* mathematisches Modell auf Papier, dann *(ii)* PuLP implementieren,
       lösen, interpretieren.

    > Workflow: **Problemtext → Modell → PuLP-Code → Solverstatus → Interpretation**

    ---

    ### So funktionieren die Aufgaben-Cells

    Jede Aufgabe ist als **Funktion** vorbereitet — z.B. `build_moebelfabrik()`.
    Ihr schreibt euren PuLP-Code **innerhalb der Funktion**, gebt das `prob`-
    Objekt zurück, und die Cell darunter macht automatisch:

    1. Constraint- und Zielfunktions-Check (gibt es überhaupt eine Zielfunktion?)
    2. Solver-Aufruf mit Error-Handling
    3. Anzeige von Status, Variablenwerten, Slack und Schattenpreisen
    4. Vergleich mit dem erwarteten Optimalwert (✅ / ⚠️ / ❌)

    **Variablennamen sind frei wählbar** — alles innerhalb der Funktion ist
    lokal, kann nicht mit anderen Cells kollidieren. Einfach Code schreiben,
    Cell ausführen, Feedback unten lesen.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Teil A · Lecture-Demos

    ## A.1 · Minimalbeispiel — City vs. EBike

    $$
    \begin{aligned}
    \max\ z \;=\;& 90\, x_C + 180\, x_E \\
    \text{s.t.}\;& 2\, x_C + 3\, x_E \leq 360 && \text{(Montage)} \\
    & x_E \leq 80 && \text{(Akku)} \\
    & x_C \leq 120 && \text{(Nachfrage City)} \\
    & x_C,\ x_E \geq 0
    \end{aligned}
    $$

    Vier Schritte: **Problem** → **Variablen** → **Zielfunktion + Constraints** → **Solve**.
    """)
    return


@app.cell
def demo_city_ebike(mo, pl):
    ce_modell = pl.LpProblem("City_vs_EBike", pl.LpMaximize)

    ce_x_C = pl.LpVariable("x_C", lowBound=0)
    ce_x_E = pl.LpVariable("x_E", lowBound=0)

    ce_modell += 90 * ce_x_C + 180 * ce_x_E, "Deckungsbeitrag"
    ce_modell += 2 * ce_x_C + 3 * ce_x_E <= 360, "Montage"
    ce_modell += ce_x_E <= 80, "Akku"
    ce_modell += ce_x_C <= 120, "Nachfrage_City"

    ce_modell.solve(pl.PULP_CBC_CMD(msg=0))

    mo.md(f"""
    **Solverstatus:** `{pl.LpStatus[ce_modell.status]}`

    | Variable | Wert |
    |---|---|
    | $x_C$ (City) | **{pl.value(ce_x_C):.2f}** |
    | $x_E$ (EBike) | **{pl.value(ce_x_E):.2f}** |
    | $z$ (Deckungsbeitrag) | **{pl.value(ce_modell.objective):.2f} €** |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## A.2 · Variablentypen (Kurzüberblick)

    PuLP unterstützt drei Typen via `cat=`:

    | Typ | `cat=` | Wertebereich | Anwendung |
    |---|---|---|---|
    | Stetig *(Default)* | `"Continuous"` | $x \in \mathbb{R}_{\geq 0}$ | Mengen (Liter, kg, hl) |
    | Ganzzahlig | `"Integer"` | $x \in \mathbb{Z}_{\geq 0}$ | Stück (Bikes, Mitarbeiter) |
    | Binär | `"Binary"` | $x \in \{0, 1\}$ | Ja/Nein-Entscheidung |

    ```python
    pl.LpVariable("Menge",  lowBound=0)                  # stetig
    pl.LpVariable("Anzahl", lowBound=0, cat="Integer")   # ganzzahlig
    pl.LpVariable("Bau",    cat="Binary")                # 0/1
    ```

    **Heute fokussieren wir auf stetige LPs.** Modelle mit `Integer` / `Binary`
    (Branch-and-Bound, MIP) kommen in **VL 10 / VL 11** und in den späteren
    Übungen ausführlich dran.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## A.3 · Produktionsplanung mit Daten-Dicts

    Statt jede Variable einzeln zu deklarieren, packen wir die Parameter in ein
    Dictionary und bauen das Modell per Schleife + `lpSum` auf — **skaliert
    sofort**, wenn ein weiteres Produkt dazukommt.
    """)
    return


@app.cell
def bp_data():
    bp_produkte = {
        "City":  {"gewinn":  90, "montage": 2, "akku": 0, "max": 120},
        "EBike": {"gewinn": 180, "montage": 3, "akku": 1, "max": None},
    }
    bp_kapa = {"montage": 360, "akku": 80}
    return bp_kapa, bp_produkte


@app.cell
def bp_model(bp_kapa, bp_produkte, mo, pl):
    bp_modell = pl.LpProblem("BikeProd", pl.LpMaximize)
    bp_x = {p: pl.LpVariable(p, lowBound=0) for p in bp_produkte}

    bp_modell += pl.lpSum(bp_produkte[p]["gewinn"] * bp_x[p] for p in bp_produkte), "Gewinn"

    bp_modell += (
        pl.lpSum(bp_produkte[p]["montage"] * bp_x[p] for p in bp_produkte)
        <= bp_kapa["montage"]
    ), "Montagekapazitaet"

    bp_modell += (
        pl.lpSum(bp_produkte[p]["akku"] * bp_x[p] for p in bp_produkte)
        <= bp_kapa["akku"]
    ), "Akku"

    for bp_p, bp_d in bp_produkte.items():
        if bp_d["max"] is not None:
            bp_modell += bp_x[bp_p] <= bp_d["max"], f"Nachfrage_{bp_p}"

    bp_modell.solve(pl.PULP_CBC_CMD(msg=0))

    bp_zeilen = "\n".join(f"| {p} | {bp_x[p].value():.2f} |" for p in bp_produkte)
    mo.md(f"""
    **Status:** `{pl.LpStatus[bp_modell.status]}` · **Gewinn:** **{pl.value(bp_modell.objective):.2f} €**

    | Produkt | Menge |
    |---|---|
    {bp_zeilen}
    """)
    return (bp_modell,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## A.4 · Auswertung — Status, Slack, Schattenpreise

    Pro Constraint liefert PuLP nach `solve()`:

    - **`c.slack`** — wieviel Kapazität bleibt *unbenutzt*? `slack > 0` ⇒ nicht
      bindend (Reserve).
    - **`c.pi`** — Schattenpreis: Gewinnzuwachs pro zusätzlicher Einheit der
      Ressource.
    """)
    return


@app.cell
def bp_auswertung(bp_modell, mo):
    bp_aus_zeilen = []
    for bp_name, bp_c in bp_modell.constraints.items():
        bp_aus_zeilen.append(f"| `{bp_name}` | {bp_c.slack:.2f} | {bp_c.pi:.2f} |")
    mo.md(f"""
    | Constraint | Slack | Schattenpreis (π) |
    |---|---|---|
    {chr(10).join(bp_aus_zeilen)}

    *Lesart:* π > 0 ⇒ bindende Restriktion. Eine zusätzliche Einheit dieser
    Ressource bringt exakt π € extra Gewinn.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## A.5 · Kostenminimierung mit Service-Obligation

    Wir **müssen** ≥ 95 % der Nachfrage bedienen; jede fehlende Einheit kostet
    **200 € Strafe**. Modelliert über Unterdeckungs-Variablen `sv_short[p]`.

    $$
    \begin{aligned}
    \min\ & \sum_{p} \big( k_p \cdot x_p + 200 \cdot \text{short}_p \big) \\
    \text{s.t.}\;& \sum_p m_p \cdot x_p \leq 320 && \text{(Montage)} \\
    & x_p + \text{short}_p \geq 0{,}95 \cdot d_p && \forall p \\
    & x_p,\ \text{short}_p \geq 0
    \end{aligned}
    $$
    """)
    return


@app.cell
def sv_model(mo, pl):
    sv_prod = {
        "City":  {"kosten": 55, "montage": 2, "nachfrage": 90},
        "EBike": {"kosten": 95, "montage": 3, "nachfrage": 60},
    }
    sv_montagekap = 320
    sv_servicequote = 0.95
    sv_penalty = 200

    sv_modell = pl.LpProblem("BikeCostMin", pl.LpMinimize)
    sv_x = {p: pl.LpVariable(f"Prod_{p}",  lowBound=0) for p in sv_prod}
    sv_short = {p: pl.LpVariable(f"Short_{p}", lowBound=0) for p in sv_prod}

    sv_modell += pl.lpSum(
        sv_prod[p]["kosten"] * sv_x[p] + sv_penalty * sv_short[p] for p in sv_prod
    ), "Gesamtkosten"

    sv_modell += pl.lpSum(sv_prod[p]["montage"] * sv_x[p] for p in sv_prod) <= sv_montagekap, "Montage"

    for sv_p, sv_daten in sv_prod.items():
        sv_modell += sv_x[sv_p] + sv_short[sv_p] >= sv_servicequote * sv_daten["nachfrage"], f"Service_{sv_p}"

    sv_modell.solve(pl.PULP_CBC_CMD(msg=0))

    sv_zeilen = "\n".join(
        f"| {p} | {sv_x[p].value():.2f} | {sv_short[p].value():.2f} | {sv_prod[p]['nachfrage']} |"
        for p in sv_prod
    )
    mo.md(f"""
    **Status:** `{pl.LpStatus[sv_modell.status]}` · **Gesamtkosten:** **{pl.value(sv_modell.objective):.2f} €**

    | Produkt | Produktion $x_p$ | Unterdeckung $\\text{{short}}_p$ | Nachfrage $d_p$ |
    |---|---|---|---|
    {sv_zeilen}

    *Trade-off:* Reicht die Montagekapazität, geht `short → 0`. Wird sie knapp,
    kauft der Solver Unterdeckung gegen die 200 €-Strafe ein.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## A.6 · Debugging — `writeLP` und Solverstatus

    Häufigster Schritt beim Debuggen: Modell als `.lp`-Datei dumpen und prüfen,
    ob Vorzeichen, Schleifen und Constraints korrekt sind.
    """)
    return


@app.cell
def bp_writelp(bp_modell, mo):
    bp_lp_path = "/tmp/pue_demo_bikeprod.lp"
    bp_modell.writeLP(bp_lp_path)
    with open(bp_lp_path) as bp_fh:
        bp_lp_text = bp_fh.read()
    mo.md(f"""
    Inhalt von `{bp_lp_path}`:

    ```
    {bp_lp_text}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sonderfälle live ausprobieren
    """)
    return


@app.cell(hide_code=True)
def sf_widget(mo):
    sf_fall = mo.ui.dropdown(
        options=["Optimal", "Infeasible", "Unbounded"],
        value="Optimal",
        label="Modellfall",
    )
    sf_fall
    return (sf_fall,)


@app.cell
def sf_solve(mo, pl, sf_fall):
    sf_m = pl.LpProblem("Fall_" + sf_fall.value, pl.LpMaximize)
    sf_x = pl.LpVariable("x", lowBound=0)
    sf_y = pl.LpVariable("y", lowBound=0)

    if sf_fall.value == "Optimal":
        sf_m += 3 * sf_x + 2 * sf_y
        sf_m += sf_x + sf_y <= 10
        sf_m += sf_x <= 6
        sf_kommentar = "Klassisches LP mit endlicher Optimallösung."
    elif sf_fall.value == "Infeasible":
        sf_m += sf_x + sf_y
        sf_m += sf_x + sf_y >= 10
        sf_m += sf_x + sf_y <= 5
        sf_kommentar = "Widerspruch: `x+y ≥ 10` **und** `x+y ≤ 5` — kein zulässiger Punkt."
    else:  # Unbounded
        sf_m += sf_x + sf_y
        sf_m += sf_x >= 0
        sf_kommentar = "Maximierung ohne Obergrenze → Zielfunktion → $\\infty$."

    sf_m.solve(pl.PULP_CBC_CMD(msg=0))
    mo.md(f"""
    **Solverstatus:** `{pl.LpStatus[sf_m.status]}`

    {sf_kommentar}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Teil B · Eigene Modelle

    Drei Aufgaben — jeweils:

    1. **Modellieren:** Variablen, Zielfunktion, NB auf Papier (oder im Kopf)
       festhalten.
    2. **Implementieren:** Funktionsrumpf ausfüllen — Variablennamen frei wählbar,
       `prob` zurückgeben.
    3. **Auswerten:** Die Check-Cell darunter solved automatisch und zeigt Status,
       Werte, Slack und einen Soll/Ist-Vergleich.
    4. **Musterlösung** als ausklappbares Accordion zum Vergleich.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Aufgabe 1 · Möbelfabrik

    > Eine kleine Möbelfabrik fertigt **Stühle** ($x_S$) und **Tische** ($x_T$).
    > Pro Stuhl: **3 h** Bearbeitungszeit und **2 m²** Holz, Deckungsbeitrag
    > **40 €**. Pro Tisch: **5 h** und **4 m²**, Deckungsbeitrag **70 €**.
    > Pro Woche: **120 h** Bearbeitungszeit und **90 m²** Holz verfügbar.

    **(1) Mathematisches Modell aufstellen** (Notation: $x_S, x_T \geq 0$, max
    Deckungsbeitrag).

    **(2) Implementierung** — füllt die Cell unten aus:
    """)
    return


@app.cell
def mb_aufgabe(pl):
    def build_moebelfabrik():
        """Aufgabe 1 · Möbelfabrik — euer Code in dieser Funktion.

        Variablennamen sind frei wählbar (alles lokal zur Funktion).
        Wichtig: am Ende `return prob`.
        """
        prob = pl.LpProblem("Moebelfabrik", pl.LpMaximize)

        # TODO: Variablen, Zielfunktion, Constraints hinzufügen


        return prob

    mb_prob = build_moebelfabrik()
    return (mb_prob,)


@app.cell(hide_code=True)
def mb_check(check_solution, mb_prob):
    check_solution(mb_prob, expected_obj=1650.0, label="Aufgabe 1 · Möbelfabrik")
    return


@app.cell(hide_code=True)
def mb_musterloesung(mo):
    mo.accordion({
        "Musterlösung Aufgabe 1 anzeigen": mo.md(r"""
    **Mathematisches Modell**

    $$
    \begin{aligned}
    \max\ z \;=\;& 40\, x_S + 70\, x_T \\
    \text{s.t.}\;& 3\, x_S + 5\, x_T \leq 120 && \text{(Zeit)} \\
    & 2\, x_S + 4\, x_T \leq 90 && \text{(Holz)} \\
    & x_S,\ x_T \geq 0
    \end{aligned}
    $$

    **PuLP**

    ```python
    mb_modell = pl.LpProblem("Moebelfabrik", pl.LpMaximize)
    mb_x_S = pl.LpVariable("Stuhl", lowBound=0)
    mb_x_T = pl.LpVariable("Tisch", lowBound=0)

    mb_modell += 40 * mb_x_S + 70 * mb_x_T, "Deckungsbeitrag"
    mb_modell += 3 * mb_x_S + 5 * mb_x_T <= 120, "Zeit"
    mb_modell += 2 * mb_x_S + 4 * mb_x_T <= 90,  "Holz"

    mb_modell.solve(pl.PULP_CBC_CMD(msg=0))
    ```

    **Lösung:** $x_S = 15$, $x_T = 15$, $z = 1650$ €. Beide Restriktionen bindend.
    """)
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Aufgabe 2 · Müsli-Mischung

    > Eine Bäckerei mischt **100 kg** Granola aus **Hafer** ($x_H$),
    > **Nüssen** ($x_N$) und **Trockenfrüchten** ($x_F$).
    > Preise pro kg: 2 € (Hafer), 8 € (Nüsse), 5 € (Früchte).
    >
    > Für die 100 kg gelten die Nährwert-Mindestwerte:
    >
    > | Nährstoff | Hafer | Nüsse | Früchte | Mindestmenge |
    > |---|---|---|---|---|
    > | Protein (g/kg) | 130 | 200 | 30  | **12 000 g** |
    > | Ballaststoffe (g/kg) | 100 | 60 | 200 | **11 000 g** |
    > | Fett (g/kg) | 70 | 600 | 5   | **12 000 g** |
    >
    > Außerdem: Nüsse ≤ 30 % der Mischung. Ziel: **Kosten minimieren**.

    **(1) Modell aufstellen** — drei Nährstoff-NB, eine Mengen-Gleichung, eine Nuss-Obergrenze.

    **(2) Implementieren** — nutzt das Dict-Pattern aus Demo A.3:
    """)
    return


@app.cell
def ms_aufgabe(pl):
    ms_komp = {
        "Hafer":   {"preis": 2, "protein": 130, "ballast": 100, "fett":  70},
        "Nuesse":  {"preis": 8, "protein": 200, "ballast":  60, "fett": 600},
        "Fruechte":{"preis": 5, "protein":  30, "ballast": 200, "fett":   5},
    }

    def build_muesli(komp):
        """Aufgabe 2 · Müsli-Mischung — Daten kommen als `komp` rein.

        Variablennamen frei wählbar. `prob` am Ende zurückgeben.
        """
        prob = pl.LpProblem("Muesli", pl.LpMinimize)

        # TODO: Variablen, Zielfunktion, Constraints hinzufügen

        return prob

    ms_prob = build_muesli(ms_komp)
    return ms_komp, ms_prob


@app.cell(hide_code=True)
def ms_check(check_solution, ms_prob):
    check_solution(ms_prob, expected_obj=310.71, label="Aufgabe 2 · Müsli-Mischung")
    return


@app.cell(hide_code=True)
def ms_musterloesung(mo):
    mo.accordion({
        "Musterlösung Aufgabe 2 anzeigen": mo.md(r"""
    **Mathematisches Modell**

    Mit $K = \{\text{Hafer, Nüsse, Früchte}\}$ und Parametern $p_k$, $\text{prot}_k$, $\text{ball}_k$, $\text{fett}_k$:

    $$
    \begin{aligned}
    \min\ & \sum_{k \in K} p_k \cdot x_k \\
    \text{s.t.}\;& \sum_k \text{prot}_k \cdot x_k \geq 12000 \\
    & \sum_k \text{ball}_k \cdot x_k \geq 11000 \\
    & \sum_k \text{fett}_k \cdot x_k \geq 12000 \\
    & \sum_k x_k = 100 \\
    & x_{\text{Nüsse}} \leq 30 \\
    & x_k \geq 0
    \end{aligned}
    $$

    **PuLP**

    ```python
    ms_modell = pl.LpProblem("Muesli", pl.LpMinimize)
    ms_x = {k: pl.LpVariable(k, lowBound=0) for k in ms_komp}

    ms_modell += pl.lpSum(ms_komp[k]["preis"] * ms_x[k] for k in ms_komp)
    ms_modell += pl.lpSum(ms_komp[k]["protein"] * ms_x[k] for k in ms_komp) >= 12000, "Protein"
    ms_modell += pl.lpSum(ms_komp[k]["ballast"] * ms_x[k] for k in ms_komp) >= 11000, "Ballast"
    ms_modell += pl.lpSum(ms_komp[k]["fett"]    * ms_x[k] for k in ms_komp) >= 12000, "Fett"
    ms_modell += pl.lpSum(ms_x[k] for k in ms_komp) == 100, "Menge"
    ms_modell += ms_x["Nuesse"] <= 30, "Nussgrenze"

    ms_modell.solve(pl.PULP_CBC_CMD(msg=0))
    ```

    **Lösung (gerundet):** $x_H \approx 74{,}3$ kg, $x_N \approx 11{,}2$ kg,
    $x_F \approx 14{,}5$ kg, **Kosten ≈ 310{,}71 €**. Nuss-Obergrenze ist nicht
    bindend — Nüsse sind zu teuer, der Solver bleibt freiwillig unter 30 kg.
    """)
    })
    return


@app.cell(hide_code=True)
def ms_referenz(mo, ms_komp, pl):
    ms_ref = pl.LpProblem("Muesli_Ref", pl.LpMinimize)
    ms_xref = {k: pl.LpVariable(k + "_ref", lowBound=0) for k in ms_komp}

    ms_ref += pl.lpSum(ms_komp[k]["preis"]   * ms_xref[k] for k in ms_komp)
    ms_ref += pl.lpSum(ms_komp[k]["protein"] * ms_xref[k] for k in ms_komp) >= 12000, "Protein"
    ms_ref += pl.lpSum(ms_komp[k]["ballast"] * ms_xref[k] for k in ms_komp) >= 11000, "Ballast"
    ms_ref += pl.lpSum(ms_komp[k]["fett"]    * ms_xref[k] for k in ms_komp) >= 12000, "Fett"
    ms_ref += pl.lpSum(ms_xref[k] for k in ms_komp) == 100, "Menge"
    ms_ref += ms_xref["Nuesse"] <= 30, "Nussgrenze"
    ms_ref.solve(pl.PULP_CBC_CMD(msg=0))

    ms_ref_zeilen = "\n".join(f"| {k} | {ms_xref[k].value():.2f} |" for k in ms_komp)
    mo.md(f"""
    **Referenz-Solverlauf:** Status `{pl.LpStatus[ms_ref.status]}` ·
    Kosten **{pl.value(ms_ref.objective):.2f} €**

    | Komponente | kg |
    |---|---|
    {ms_ref_zeilen}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Aufgabe 3 · Transportproblem

    > Eine Brauerei beliefert **3 Märkte** ($M_1, M_2, M_3$) aus **2 Werken**
    > ($W_1, W_2$).
    >
    > | | $M_1$ | $M_2$ | $M_3$ | **Kapazität** |
    > |---|---|---|---|---|
    > | $W_1$ | 4 € | 6 € | 9 € | 200 hl |
    > | $W_2$ | 5 € | 4 € | 7 € | 300 hl |
    > | **Nachfrage** | 150 hl | 180 hl | 170 hl | |
    >
    > Die Tabelle enthält die **Transportkosten pro hl** zwischen Werk und Markt.
    > Ziel: **Gesamtkosten minimieren**, alle Nachfragen decken, keine
    > Werks-Kapazität überschreiten.

    **(1) Modell aufstellen**

    Entscheidungsvariablen: $x_{ij} \geq 0$ = transportierte Menge (hl) von Werk $i$
    nach Markt $j$, mit $i \in I = \{W_1, W_2\}$, $j \in J = \{M_1, M_2, M_3\}$.

    Restriktions-Typen:

    - **Kapazität pro Werk:** $\sum_j x_{ij} \leq c_i$ für alle $i$
    - **Nachfrage pro Markt:** $\sum_i x_{ij} \geq d_j$ für alle $j$

    **(2) Implementieren** — Tipp: verschachtelte Comprehensions
    `tp_x = {(i, j): pl.LpVariable(f"x_{i}_{j}", lowBound=0) for i in tp_werke for j in tp_maerkte}`:
    """)
    return


@app.cell
def tp_aufgabe(pl):
    tp_werke   = ["W1", "W2"]
    tp_maerkte = ["M1", "M2", "M3"]
    tp_kosten = {
        ("W1", "M1"): 4, ("W1", "M2"): 6, ("W1", "M3"): 9,
        ("W2", "M1"): 5, ("W2", "M2"): 4, ("W2", "M3"): 7,
    }
    tp_kapa   = {"W1": 200, "W2": 300}
    tp_bedarf = {"M1": 150, "M2": 180, "M3": 170}

    def build_transport(werke, maerkte, kosten, kapa, bedarf):
        """Aufgabe 3 · Transportproblem — Daten kommen als Parameter rein.

        Variablennamen frei wählbar. `prob` am Ende zurückgeben.
        """
        prob = pl.LpProblem("Transport", pl.LpMinimize)

        # TODO: Variablen, Zielfunktion, Constraints hinzufügen

        return prob

    tp_prob = build_transport(tp_werke, tp_maerkte, tp_kosten, tp_kapa, tp_bedarf)
    return tp_bedarf, tp_kapa, tp_kosten, tp_maerkte, tp_prob, tp_werke


@app.cell(hide_code=True)
def tp_check(check_solution, tp_prob):
    check_solution(tp_prob, expected_obj=2610.0, label="Aufgabe 3 · Transportproblem")
    return


@app.cell(hide_code=True)
def tp_musterloesung(mo):
    mo.accordion({
        "Musterlösung Aufgabe 3 anzeigen": mo.md(r"""
    **Mathematisches Modell**

    Mit $I = \{W_1, W_2\}$, $J = \{M_1, M_2, M_3\}$, Kosten $c_{ij}$,
    Kapazitäten $K_i$, Bedarfen $D_j$:

    $$
    \begin{aligned}
    \min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} \cdot x_{ij} \\
    \text{s.t.}\;& \sum_{j \in J} x_{ij} \leq K_i && \forall i \in I \\
    & \sum_{i \in I} x_{ij} \geq D_j && \forall j \in J \\
    & x_{ij} \geq 0
    \end{aligned}
    $$

    **PuLP**

    ```python
    tp_modell = pl.LpProblem("Transport", pl.LpMinimize)
    tp_x = {(i, j): pl.LpVariable(f"x_{i}_{j}", lowBound=0)
        for i in tp_werke for j in tp_maerkte}

    tp_modell += pl.lpSum(tp_kosten[i, j] * tp_x[i, j]
                      for i in tp_werke for j in tp_maerkte)

    for i in tp_werke:
    tp_modell += pl.lpSum(tp_x[i, j] for j in tp_maerkte) <= tp_kapa[i], f"Kapa_{i}"

    for j in tp_maerkte:
    tp_modell += pl.lpSum(tp_x[i, j] for i in tp_werke) >= tp_bedarf[j], f"Bedarf_{j}"

    tp_modell.solve(pl.PULP_CBC_CMD(msg=0))
    ```

    **Lösung:** Werke arbeiten an der Kapazitätsgrenze
    (200 + 300 = 500 hl = Gesamtnachfrage). Optimales Routing:

    | | $M_1$ | $M_2$ | $M_3$ |
    |---|---|---|---|
    | $W_1$ | 150 | 50  | 0   |
    | $W_2$ | 0   | 130 | 170 |

    **Gesamtkosten = 2 610 €.**

    *Lesart:* $W_2$ ist auf $M_2$ und $M_3$ günstiger als $W_1$ → fährt diese Märkte
    voll an; $W_1$ nimmt $M_1$ komplett (mit 4 €/hl unschlagbar) und füllt den Rest
    seiner Kapazität auf $M_2$.
    """)
    })
    return


@app.cell(hide_code=True)
def tp_referenz(mo, pl, tp_bedarf, tp_kapa, tp_kosten, tp_maerkte, tp_werke):
    tp_ref = pl.LpProblem("Transport_Ref", pl.LpMinimize)
    tp_xref = {(i, j): pl.LpVariable(f"xref_{i}_{j}", lowBound=0)
               for i in tp_werke for j in tp_maerkte}

    tp_ref += pl.lpSum(tp_kosten[i, j] * tp_xref[i, j]
                       for i in tp_werke for j in tp_maerkte)

    for i in tp_werke:
        tp_ref += pl.lpSum(tp_xref[i, j] for j in tp_maerkte) <= tp_kapa[i], f"Kapa_{i}"
    for j in tp_maerkte:
        tp_ref += pl.lpSum(tp_xref[i, j] for i in tp_werke) >= tp_bedarf[j], f"Bedarf_{j}"

    tp_ref.solve(pl.PULP_CBC_CMD(msg=0))

    tp_header = "| | " + " | ".join(tp_maerkte) + " |"
    tp_sep    = "|---|" + "---|" * len(tp_maerkte)
    tp_rows   = "\n".join(
        "| " + i + " | " + " | ".join(f"{tp_xref[i, j].value():.0f}" for j in tp_maerkte) + " |"
        for i in tp_werke
    )
    mo.md(f"""
    **Referenz-Solverlauf:** Status `{pl.LpStatus[tp_ref.status]}` ·
    **Gesamtkosten {pl.value(tp_ref.objective):.0f} €**

    Transportmengen (hl):

    {tp_header}
    {tp_sep}
    {tp_rows}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Aufgabe 4 · Werbebudget-Allokation

    > Eine Agentur soll ein **Werbebudget von 100 000 €** auf vier Kanäle
    > verteilen. Bekannt sind pro Kanal: die **Reichweite je Euro Spend**
    > (in „erreichten Personen pro €"), ein **Mindestbudget** und ein
    > **Höchstbudget**:
    >
    > | Kanal | Reichweite/€ | Min-Budget (€) | Max-Budget (€) |
    > |---|---|---|---|
    > | TV     | 12 | 20 000 | 60 000 |
    > | Social | 18 | 10 000 | 40 000 |
    > | Print  |  7 |  5 000 | 30 000 |
    > | Radio  | 10 |  5 000 | 25 000 |
    >
    > Zusätzliche Vorgabe: der **digitale Kanal (Social)** muss mindestens
    > **30 % des Gesamtbudgets** ausmachen.
    >
    > **Ziel:** Reichweite maximieren.

    **Was ihr macht (alles selbst):**

    1. Mathematisches Modell aufstellen (auf Papier oder im Kopf).
    2. Daten in einem Python-Dict (oder mehreren) ablegen — Format frei wählbar.
    3. Modell in der Funktion `build_werbung()` aufbauen und `prob` zurückgeben.
    """)
    return


@app.cell
def w4_aufgabe(pl):
    def build_werbung():
        """Aufgabe 4 · Werbebudget-Allokation — alles selbst aufsetzen.

        Daten, Variablen, Zielfunktion, Constraints. `prob` zurückgeben.
        """
        prob = pl.LpProblem("Werbebudget", pl.LpMaximize)

        # TODO: Daten-Dict(s), Variablen, Zielfunktion, Constraints

        return prob

    w4_prob = build_werbung()
    return (w4_prob,)


@app.cell(hide_code=True)
def w4_check(check_solution, w4_prob):
    check_solution(w4_prob, expected_obj=1405000.0, label="Aufgabe 4 · Werbebudget")
    return


@app.cell(hide_code=True)
def w4_musterloesung(mo):
    mo.accordion({
        "Musterlösung Aufgabe 4 anzeigen": mo.md(r"""
**Mathematisches Modell**

Mit Kanälen $K = \{\text{TV, Social, Print, Radio}\}$ und Parametern
$r_k$ (Reichweite/€), $\underline{b}_k$, $\overline{b}_k$, $B = 100\,000$:

$$
\begin{aligned}
\max\ & \sum_{k \in K} r_k \cdot b_k \\
\text{s.t.}\;& \sum_{k} b_k \leq B \\
& b_{\text{Social}} \geq 0{,}3 \cdot B \\
& \underline{b}_k \leq b_k \leq \overline{b}_k && \forall k \in K
\end{aligned}
$$

**PuLP**

```python
budget = 100_000
kanal = {
    "TV":     {"reach": 12, "min": 20_000, "max": 60_000},
    "Social": {"reach": 18, "min": 10_000, "max": 40_000},
    "Print":  {"reach":  7, "min":  5_000, "max": 30_000},
    "Radio":  {"reach": 10, "min":  5_000, "max": 25_000},
}

prob = pl.LpProblem("Werbebudget", pl.LpMaximize)
b = {k: pl.LpVariable(k, lowBound=kanal[k]["min"], upBound=kanal[k]["max"]) for k in kanal}

prob += pl.lpSum(kanal[k]["reach"] * b[k] for k in kanal)
prob += pl.lpSum(b[k] for k in kanal) <= budget, "Budget"
prob += b["Social"] >= 0.3 * budget, "Digital_Anteil"

prob.solve(pl.PULP_CBC_CMD(msg=0))
```

**Lösung:** TV = 50 000 €, Social = 40 000 € (max), Print = 5 000 € (min),
Radio = 5 000 € (min). **Reichweite = 1 405 000** Personen.

*Lesart:* Social ist der attraktivste Kanal pro € → fährt auf Maximum.
TV ist Zweitbester, bekommt was übrig bleibt (50 000 €, deutlich unter
seinem Max von 60 000). Print/Radio gehen auf Minimum-Bound — beide werden
nur „bedient", weil sie eine Min-Budget-Verpflichtung haben.
""")
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Aufgabe 5 · Mehrperioden-Produktion mit Lager

    > Eine Bäckerei plant die Produktion eines Sauerteig-Brotes über **4 Wochen**.
    > Die Produktionskosten variieren (Energie- und Mehlpreise), die Nachfrage
    > ist bekannt.
    >
    > Lagerung kostet **0,10 €/Stück und Woche**, die Lagerkapazität ist
    > **150 Stück** am Wochenende. Anfangs- und Endlager sind **0**.
    >
    > **Ziel:** Gesamtkosten (Produktion + Lager) minimieren.

    Im Gegensatz zu den vorigen Aufgaben sind die Daten dieses Mal **nicht** im
    Aufgabentext als Dict aufgeschrieben, sondern liegen — wie in der Praxis —
    als **pandas DataFrame** vor. Die Cell darunter zeigt euch den DataFrame
    (Konstruktion ist versteckt — stellt euch vor, er käme aus einer Excel- oder
    CSV-Datei via `pd.read_csv(...)`).

    **Was ihr macht:**

    1. **Mathematisches Modell** aufstellen — Tipp:
       Lager-Bilanz $s_t = s_{t-1} + x_t - d_t$ mit $s_0 = 0$.
    2. **Daten aus dem DataFrame ziehen** — z.B. via
       `df["Spaltenname"].to_dict()` oder `df.loc[t, "Spalte"]`.
    3. **Modell** in `build_baeckerei(df)` aufbauen, `prob` zurückgeben.
    """)
    return


@app.cell(hide_code=True)
def w5_data(mo, pd):
    df_baeckerei = pd.DataFrame({
        "Produktionskosten": [1.80, 2.00, 1.90, 2.10],
        "Nachfrage":         [200,  300,  250,  350],
    }, index=pd.Index([1, 2, 3, 4], name="Woche"))

    mo.vstack([
        mo.md("**Datenbasis** — `df_baeckerei` (verfügbar in der Aufgaben-Cell):"),
        df_baeckerei,
    ])
    return (df_baeckerei,)


@app.cell
def w5_aufgabe(df_baeckerei, pl):
    def build_baeckerei(df):
        """Aufgabe 5 · Mehrperioden-Produktion mit Lager.

        Parameter
        ---------
        df : pandas DataFrame mit Index `Woche` und Spalten
             `Produktionskosten`, `Nachfrage`.

        Tipps zur DataFrame-Nutzung:
        - `df.index.to_list()` liefert die Wochen-Indizes
        - `df["Produktionskosten"].to_dict()` macht ein {Woche: Kosten}-Dict
        - oder direkt iterieren: `for t in df.index: ... df.loc[t, "Nachfrage"]`

        Lagerkosten = 0.10 €/Stück·Woche, Lagerkapazität = 150 Stück (fest).
        """
        prob = pl.LpProblem("Mehrperioden", pl.LpMinimize)

        # TODO: Daten aus df ziehen, Variablen, Zielfunktion,
        #       Lager-Bilanz, Lagerkapazität, Endlager-Constraint

        return prob

    w5_prob = build_baeckerei(df_baeckerei)
    return (w5_prob,)


@app.cell(hide_code=True)
def w5_check(check_solution, w5_prob):
    check_solution(w5_prob, expected_obj=2140.0, label="Aufgabe 5 · Mehrperioden-Produktion")
    return


@app.cell(hide_code=True)
def w5_musterloesung(mo):
    mo.accordion({
        "Musterlösung Aufgabe 5 anzeigen": mo.md(r"""
**Mathematisches Modell**

Mit Wochen $T = \{1, 2, 3, 4\}$, Produktionskosten $c_t$, Nachfrage $d_t$,
Lagerkosten $h = 0{,}10$ €/Stück·Woche, Lagerkapazität $K = 150$:

$$
\begin{aligned}
\min\ & \sum_{t \in T} \big( c_t \cdot x_t + h \cdot s_t \big) \\
\text{s.t.}\;& s_t = s_{t-1} + x_t - d_t && \forall t \in T \quad (s_0 = 0) \\
& s_t \leq K && \forall t \in T \\
& s_T = 0 && \text{(Endlager)} \\
& x_t,\ s_t \geq 0
\end{aligned}
$$

**PuLP — Daten aus dem DataFrame ziehen**

```python
def build_baeckerei(df):
    # DataFrame → Python-Datenstrukturen
    wochen    = df.index.to_list()
    kosten    = df["Produktionskosten"].to_dict()   # {1: 1.80, 2: 2.00, ...}
    nachfrage = df["Nachfrage"].to_dict()           # {1: 200,  2: 300,  ...}
    lagerkosten = 0.10
    lagerkap    = 150

    prob = pl.LpProblem("Mehrperioden", pl.LpMinimize)
    x = {t: pl.LpVariable(f"x_{t}", lowBound=0) for t in wochen}
    s = {t: pl.LpVariable(f"s_{t}", lowBound=0, upBound=lagerkap) for t in wochen}

    prob += pl.lpSum(kosten[t]*x[t] + lagerkosten*s[t] for t in wochen)

    for t in wochen:
        prev = 0 if t == wochen[0] else s[t-1]
        prob += s[t] == prev + x[t] - nachfrage[t], f"Bilanz_{t}"

    prob += s[wochen[-1]] == 0, "Endlager"
    return prob
```

*Pattern „Real-World Data":* die zwei Zeilen `kosten = df["..."].to_dict()`
sind die ganze Brücke zwischen pandas und PuLP. Im echten Projekt würde der
DataFrame aus `pd.read_csv("plan.csv")` oder `pd.read_excel(...)` kommen —
am Modellcode ändert sich **nichts**.

**Lösung:**

| Woche | Produktion | Lager am Ende |
|---|---|---|
| 1 | **350** | 150 |
| 2 | 150 | 0   |
| 3 | **400** | 150 |
| 4 | 200 | 0   |

**Gesamtkosten = 2 140 €.**

*Lesart:* In Woche 1 (1,80 €/St) wird auf Vorrat produziert für Woche 2
(2,00 €/St) — Ersparnis 0,20 € pro Stück, Lagerkosten 0,10 € → netto −0,10 €.
In Woche 3 (1,90 €/St) gleiches Spiel für Woche 4 (2,10 €/St). Lager läuft
in W1 und W3 jeweils voll (150 Stück = Kapazitätsgrenze) → wäre die
Kapazität größer, würde noch mehr vorgezogen.
""")
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Cheat Sheet — PuLP in 8 Zeilen

    ```python
    import pulp as pl

    m = pl.LpProblem("Name", pl.LpMaximize)         # oder LpMinimize
    x = pl.LpVariable("x", lowBound=0)              # cat="Integer"/"Binary" optional
    m += 3 * x + 2 * y                              # Zielfunktion (kein <=, ==, >=)
    m += 2 * x + y <= 10, "Ressource_A"             # Constraint mit Namen
    m.solve(pl.PULP_CBC_CMD(msg=0))                 # CBC ist Default-Solver
    print(pl.LpStatus[m.status], pl.value(m.objective), x.value())
    for name, c in m.constraints.items():
        print(name, c.slack, c.pi)                  # Slack + Schattenpreis
    ```

    **Checkliste vor dem Solve:**

    - [ ] Zielfunktion **ohne** Vergleichsoperator addiert?
    - [ ] Jede Variable mit `lowBound=0` (oder sinnvollem Bound)?
    - [ ] Constraint-Namen vergeben (für `writeLP`-Debugging)?
    - [ ] Solver-Status nach `solve()` geloggt?
    - [ ] In marimo-Aufgaben: euer Code liegt in der Funktion, `return prob` am Ende?
    """)
    return


if __name__ == "__main__":
    app.run()
