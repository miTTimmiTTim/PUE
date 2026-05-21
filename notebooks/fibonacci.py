# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
# ]
# ///
import marimo

__generated_with = "0.13.5"
app = marimo.App()

with app.setup:
    import marimo as mo


@app.cell
def _():
    mo.md(
        r"""
        # Fibonacci Calculator

        A minimal demo notebook. Replace this with your course content.
        """
    )
    return


@app.cell
def _():
    n = mo.ui.slider(1, 100, value=20, label="How many Fibonacci numbers?")
    n
    return n,


@app.cell
def _(n):
    fib = fibonacci(n.value)
    mo.md(", ".join(str(f) for f in fib))
    return


@app.function
def fibonacci(n):
    seq = [0, 1]
    for i in range(2, n):
        seq.append(seq[i - 1] + seq[i - 2])
    return seq[:n]


if __name__ == "__main__":
    app.run()
