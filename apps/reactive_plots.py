# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair==5.4.1",
#     "vega-datasets==0.9.0",
# ]
# ///
import marimo

__generated_with = "0.13.5"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("# Reactive Plots Demo")
    return


@app.cell
def _(bars, mo, scatter):
    chart = mo.ui.altair_chart(scatter & bars)
    chart
    return chart,


@app.cell
def _(chart, mo):
    filtered = mo.ui.table(chart.value)
    filtered
    return filtered,


@app.cell
def _(alt, data):
    cars = data.cars()
    brush = alt.selection_interval()
    scatter = (
        alt.Chart(cars)
        .mark_point()
        .encode(x="Horsepower", y="Miles_per_Gallon", color="Origin")
        .add_params(brush)
    )
    bars = (
        alt.Chart(cars)
        .mark_bar()
        .encode(y="Origin:N", color="Origin:N", x="count(Origin):Q")
        .transform_filter(brush)
    )
    return bars, scatter


@app.cell
def _():
    import altair as alt
    from vega_datasets import data
    return alt, data


@app.cell
def _():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
