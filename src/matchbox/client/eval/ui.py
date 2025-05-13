"""Interface for submitting evaluation data."""

import polars as pl
import streamlit as st

st.title("Matchbox evaluation session")

original_df = pl.DataFrame(
    [
        {"id": 1, "company_name": "Pippo pluto PLC", "region": "England"},
        {"id": 2, "company_name": "Pippo pluto e paperino UK LTD", "region": "England"},
        {"id": 3, "company_name": "Pippo pluto e paperino", "region": "Devon"},
    ]
)
orig_columns = list(set(original_df.columns) - {"id"})

if "step" not in st.session_state:
    st.session_state.step = "eval"
    st.session_state.df = original_df.with_columns(select=False)

if st.session_state.step == "eval":
    edited_df = st.data_editor(
        st.session_state.df,
        disabled=orig_columns,
        column_order=("select", *orig_columns),
    )

    def splinter():
        """Endorse subset of cluster."""
        st.session_state.df = edited_df.filter(~pl.col("select"))
        selected_ids = (
            edited_df.filter(pl.col("select")).select("id").to_series().to_list()
        )
        print(f"Sent judgement to server: {selected_ids}")
        if st.session_state.df.shape[0] == 1:
            looks_good()

    def looks_good():
        """Confirm cluster as is."""
        st.session_state.step = "done"
        all_ids = st.session_state.df.select("id").to_series().to_list()
        print(f"Sent judgement to server: {all_ids}")

    if (edited_df.select("select").to_series().any()) and (
        not st.session_state.df.select("select").to_series().all()
    ):
        st.button("Splinter", icon="✂️", on_click=splinter)
    else:
        st.button("Looks good to me.", icon="✅", on_click=looks_good)

else:
    st.header("You're all done 🔥")
