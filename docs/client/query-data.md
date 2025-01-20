## Match

Given a primary key and a source dataset, retrieves all primary keys that share its cluster in both the source and target datasets. Useful for making ad-hoc queries about specific items of data.

```python
import matchbox as mb
from matchbox.client.helpers import selector
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://')

mb.match(
    source_pk="8534735",
    source="dbt.companieshouse",
    target="hmrc.exporters",
    resolution="companies",
)
```

```shell
[
    {
        "cluster": 2354,
        "source": "dbt.companieshouse",
        "source_id": {"8534735", "8534736"},
        "target": "hmrc.exporters",
        "target_id": {"EXP123", "EXP124"}
    },
]
```

## Query

Retrieves entire data sources along with a unique entity identifier according to a point of resolution. Useful when:

* You're doing large-scale statistical analysis
* You're building a linking or deduplication pipeline 

```python
import matchbox as mb
from matchbox.client.helpers import selector
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://')

mb.query(
    selector(
        {
            "dbt.companieshouse": ["company_name"],
            "hmrc.exporters": ["year", "commodity_codes"],
        },
        engine=engine,
        resolution="companies",
    )
)
```

```shell
id      dbt_companieshouse_company_name         hmrc_exporters_year     hmrc_exporters_commodity_codes
122     Acme Ltd.                               2023                    ['85034', '85035']
122     Acme Ltd.                               2024                    ['72142', '72143']
5       Gamma Exports                           2023                    ['90328', '90329']
...
```
