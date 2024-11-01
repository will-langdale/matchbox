# 🔥 Matchbox PostgreSQL backend

A backend adapter for deploying Matchbox using PostgreSQL.

Currently implements the following architecture. See Confluence for [further details](https://uktrade.atlassian.net/wiki/spaces/CDL/pages/4282908700/Matchbox+0.2+architecture+ideas).

`Models.type` is one of "model", "dataset" or "human".

There are two graph-like trees in place here.

* In the models subgraph the tree is implemented as closure table, enabling quick querying of root to leaf paths at the cost of redundancy
* In the data subgraph the tree is implemented as a hierarchy, which means recursive queries are required to resolve it, but less data is stored

```mermaid
erDiagram
    Sources {
        bytes model PK,FK
        string schema
        string table
        string id
    }
    Clusters {
        bytes hash PK,FK
        bytes dataset FK
        string id
    }
    Contains {
        bytes parent PK,FK
        bytes child PK,FK
    }
    Probabilities {
        bytes model PK,FK
        bytes cluster PK,FK
        float probability
    }
    Models {
        bytes hash PK,FK
        enum type
        string name
        string description
        float truth
    }
    ModelsFrom {
        bytes parent PK,FK
        bytes child PK,FK
        int level
        float truth_cache
    }

    Sources |o--|| Models : ""
    Sources ||--o{ Clusters : ""
    Clusters ||--o{ Contains : "parent, child"
    Clusters ||--o{ Probabilities : ""
    Models ||--o{ Probabilities : ""
    Models ||--o{ ModelsFrom : "child, parent"
```
