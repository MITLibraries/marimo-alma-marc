# /// script
# requires-python = ">=3.13"
#
# dependencies = [
#   "dotenv",
#   "lxml==5.4.0",
#   "marcalyx==1.0.2",
#   "marimo",
#   "pandas==2.3.0",
#   "timdex-dataset-api",
# ]
# [tool.uv.sources]
# timdex-dataset-api = { git = "https://github.com/MITLibraries/timdex-dataset-api", editable=true } # noqa: W505
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full", app_title="MARC tag values")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Alma MARC tag values

    This notebook provides a way to extract tags from MARC records using versions stored in the TIMDEX dataset.
    """
    )
    return


@app.cell
def _(mo):
    # instantiate TIMDEXDataset instance and setup logging
    with mo.status.spinner(title="Loading application..."):
        import os

        from timdex_dataset_api import TIMDEXDataset
        from timdex_dataset_api.config import configure_dev_logger

        configure_dev_logger()

        timdex_dataset = TIMDEXDataset(
            os.environ["TIMDEX_DATASET_LOCATION"],
            preload_current_records=True,
        )

    return TIMDEXDataset, timdex_dataset


@app.cell
def _(TIMDEXDataset):
    import time
    from collections.abc import Iterator

    import marcalyx
    import pandas as pd
    from lxml import etree

    class AlmaTagAnalysis:
        def __init__(self, timdex_dataset: TIMDEXDataset):
            self.timdex_dataset = timdex_dataset

        def parse_tags_from_record(
            self, record: dict, tags: list[str]
        ) -> Iterator[tuple[str, str, str]]:
            """Load MARC XML and yield requested tags."""
            record_xml = etree.fromstring(record["source_record"])
            record_marc = marcalyx.Record(record_xml)

            for tag in tags:
                for field in record_marc.field(tag):
                    yield (
                        record["timdex_record_id"],
                        record["run_date"].strftime("%Y-%m-%d"),
                        str(field),
                    )

        def run(
            self,
            tags: list[str],
            timdex_record_id_regex_input: str,
            limit: int | None = None,
        ) -> pd.DataFrame:
            """Prepare a DataFrame of timdex_record_id, tag, tag value."""
            start_time = time.perf_counter()

            # construct WHERE clause with timdex_record_id regex
            where = None
            if timdex_record_id_regex_input != "":
                where = (
                    f"""timdex_record_id similar to '{timdex_record_id_regex_input}'"""
                )

            # retrieve record dicts and prepare a list of (record,tag,tag_value) tuples
            rows = []
            for record_dict in self.timdex_dataset.read_dicts_iter(  # type: ignore[attr-defined]
                table="current_records",
                columns=[
                    "timdex_record_id",
                    "run_date",
                    "source_record",
                ],
                source="alma",
                action="index",
                where=where,
                limit=limit,
            ):
                rows.extend(list(self.parse_tags_from_record(record_dict, tags)))

            # construct a final dataframe
            df = pd.DataFrame(
                rows, columns=["timdex_record_id", "last_modified", "tag_data"]
            )
            print(f"elapsed: {time.perf_counter() - start_time}")
            return df

    return AlmaTagAnalysis, etree, marcalyx, time


@app.cell
def _(mo):
    tags_input = mo.ui.text(
        value="245", label="MARC tags (comma-separated)", full_width=True
    )

    timdex_record_id_regex_input = mo.ui.text(
        value="alma:.*", label="TIMDEX Record ID Regex", full_width=True
    )

    limit_input = mo.ui.text(
        value="50", label="Record limit (no value = no limit)", full_width=True
    )

    run_button = mo.ui.run_button(label="Run Analysis")

    mo.vstack(
        [
            mo.md("### Analysis Configuration"),
            tags_input,
            timdex_record_id_regex_input,
            limit_input,
            run_button,
        ]
    )
    return limit_input, run_button, tags_input, timdex_record_id_regex_input


@app.cell
def _():
    # init results dict with defaults
    results = {"first_run": True, "table": None, "elapsed": 0}
    return (results,)


@app.cell
def _(
    AlmaTagAnalysis,
    limit_input,
    mo,
    results,
    run_button,
    tags_input,
    timdex_dataset,
    timdex_record_id_regex_input,
    time,
):
    if results["first_run"] or run_button.value:
        with mo.status.spinner(title="Querying TIMDEX Dataset..."):
            start_time = time.perf_counter()

            # parse limit
            try:
                limit = int(limit_input.value.strip())
            except:
                limit = None

            # parse id regex
            timdex_record_id_regex_value = timdex_record_id_regex_input.value.strip()

            # parse tags
            tags = [tag.strip() for tag in tags_input.value.split(",") if tag.strip()]

            # main work
            ata = AlmaTagAnalysis(timdex_dataset)
            df = ata.run(
                tags=tags,
                timdex_record_id_regex_input=timdex_record_id_regex_value,
                limit=limit,
            )
            elapsed_time = time.perf_counter() - start_time

            # prepare results
            results["df"] = df
            results["elapsed"] = elapsed_time
            results["first_run"] = False

    results_table = mo.ui.table(results["df"], selection="single")
    mo.vstack(
        [
            mo.md("### Results"),
            mo.md(f"**Runtime: {results['elapsed']:.2f} seconds**"),
            results_table,
        ]
    )
    return (results_table,)


@app.cell
def _(mo, results_table, timdex_dataset):
    if len(results_table.value) == 0:
        mo.stop(True, mo.md("To view record details, please select a record from above."))

    result_row = results_table.value.iloc[0]
    timdex_record_id = result_row.timdex_record_id

    with mo.status.spinner(title="Retrieving record version from TIMDEX..."):
        metadata_result_df = timdex_dataset.metadata.conn.query(
            f"""
            select
                timdex_record_id,
                run_timestamp,
                run_type,
                run_id,
                action,
                run_record_offset
            from metadata.records
            where timdex_record_id = '{timdex_record_id}'
            order by run_timestamp
            ;
            """
        ).to_df()

    record_versions_table = mo.ui.table(metadata_result_df, selection="single")
    record_versions_table  # noqa: B018
    return (record_versions_table,)


@app.cell(hide_code=True)
def _(mo, record_versions_table, timdex_dataset):
    if len(record_versions_table.value) == 0:
        mo.stop(True, mo.md("Lastly, please select a record version from above."))

    selected_version = record_versions_table.value.iloc[0]

    with mo.status.spinner(title="Retrieving record..."):
        version = next(
            timdex_dataset.read_dicts_iter(
                timdex_record_id=selected_version.timdex_record_id,
                run_id=selected_version.run_id,
                run_record_offset=selected_version.run_record_offset,
            )
        )
    return (version,)


@app.cell
def _(etree, marcalyx, mo, record_versions_table, version):

    if len(record_versions_table.value) == 0:
        mo.stop(True, mo.md("Will show record upon version selection."))

    # get MARC XML and prettify
    source_record = version["source_record"]
    root = etree.fromstring(source_record)
    source_record_pretty = etree.tostring(root, pretty_print=True, encoding="unicode")

    # load MARC XML and produce string version
    record_marc = marcalyx.Record(root)
    marc_pretty = "\n".join([str(field) for field in record_marc.fields])

    mo.md(
        f"""
    # Record Details

    - TIMDEX Record ID: `{version["timdex_record_id"]}`
    - Run ID: `{version["run_id"]}`
    - Run Timestamp: `{version["run_timestamp"]}`
    - 245 Tag: `{record_marc.titleStatement()}"
    - [Alma Link](https://mit.primo.exlibrisgroup.com/permalink/01MIT_INST/1np2a75/{version["timdex_record_id"].replace(":", "")})

    ## MARC

    ```text
    {marc_pretty}
    ```

    ## MARC XML

    ```xml
    {source_record_pretty}
    ```

    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
