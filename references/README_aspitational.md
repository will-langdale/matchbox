# 🔗 Company matching framework

A match orchestration framework to allow the comparison, validation, and orchestration of the best match methods for the company matching job.

⚠️ The below is aspirational, **unimplemented code** to help refine where we want the API to get to. By writing instructions for the end product, we'll flush out the problems with it before they occur.

A quick overview of where we're aiming:

```python
import cmf

from cmf import clean
from cmf.helpers import (
    selector, 
    selectors, 
    cleaner, 
    cleaners, 
    comparison,
    comparisons
)

from cmf.dedupers import Naive
from cmf.linkers import CMS

# Select and query the data

my_data_selector = selector(
    table="data.data_hub_statistics",
    fields=[
        "company_name",
        "data_hub_id"
    ]
)

ch_selector = selector(
    table="companieshouse.companies",
    fields=[
        "company_name"
    ]
)
dh_selector = selector(
    table="dit.data_hub__companies",
    fields=[
        "data_hub_id"
    ]
)

ch_dh_selector = selectors(
    ch_selector,
    dh_selector
)

cluster_raw = cmf.query(
    select=ch_dh_selector
)
data_raw = cmf.query(
    select=my_data_selector
)

# Clean and process the data

cleaner_dh_id = cleaner(
    function=clean.data_hub_id,
    column="data_hub_id"
)
cleaner_company_name = cleaner(
    function=clean.company_name,
    column="company_name"
)

my_cleaners = cleaners(
    cleaner_dh_id,
    cleaner_company_name
)

cluster_clean = cmf.process(
    data=cluster_raw,
    pipeline=my_cleaners
)
data_clean = cmf.process(
    data=data_raw,
    pipeline=my_cleaners
)

# Prepare and run the deduper

company_name_comparison = comparison(
    output_column="company_name",
    l_column="company_name",
    r_column="company_name",
    sql_condition="l_column = r_column"
)
dh_id_comparison = comparison(
    output_column="data_hub_id",
    l_column="data_hub_id",
    r_column="data_hub_id",
    sql_condition="l_column = r_column"
)

all_comparisons = comparisons(
    ch_selector,
    dh_selector
)

my_deduper = cmf.make_deduper(
    dedupe_run_name="data_hub_stats",
    description="""
        Clean and check company name and ID.
    """,
    deduper=Naive,
    data=data_clean,
    dedupe_settings=all_comparisons
)

dedupe_probabilities = my_deduper()

dedupe_probabilities.to_cmf() # Can now view performance stats, also .to_df() or .to_sql() to unwrap

# Prepare and run the linker

my_linker = cmf.make_linker(
    link_run_name="data_hub_stats",
    description="""
        Using company name and ID with existing cleaning methods
    """,
    linker=CMS, 
    cluster=cluster_clean, 
    data=data_clean,
    dedupe_probabilities=dedupe_probabilities, # or a dedupe_run_name
    dedupe_threshold=0.99,
    link_settings={
        company_name_comparison: 1,
        dh_id_comparison: 3
    }
)

link_probabilities = my_linker()

link_probabilities.to_cmf() # Can now view performance stats, also .to_df() or .to_sql() to unwrap

# Resolve entities

my_clusters = cmf.to_clusters(
    link_probabilities, # or a link_run_name
    link_threshold=0.95
)

my_clusters.to_cmf() # To push to final table, also .to_df() or .to_sql()

```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
