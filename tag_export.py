import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Alma MARC tag values

    This notebook provides a way to extract tags from MARC records using versions stored in the TIMDEX dataset.
    """
    )
    return


if __name__ == "__main__":
    app.run()
