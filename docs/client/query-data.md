## Match

Given a primary key and a source dataset, retrieves all primary keys that share its cluster in both the source and target datasets. Useful for making ad-hoc queries about specific items of data.

::: matchbox.client.helpers.selector.match

## Query

Retrieves entire data sources along with a unique entity identifier according to a point of resolution. Useful when:

* You're doing large-scale statistical analysis
* You're building a linking or deduplication pipeline 

::: matchbox.client.helpers.selector.query