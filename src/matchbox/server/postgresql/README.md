# 🔥 Matchbox PostgreSQL backend

A backend adapter for deploying Matchbox using PostgreSQL.

Currently implements the following architecture. See Confluence for [further details](https://uktrade.atlassian.net/wiki/spaces/CDL/pages/4282908700/Matchbox+0.2+architecture+ideas).

`Models.type` is one of "model", "dataset" or "human".

We employ the following check constraints in Models:

* When type is "model", hash MUST NOT appear in Sources and MUST appear in ModelsFrom
* When type is "dataset", hash MUST appear in Sources and MUST appear in ModelsFrom
* When type is "human", hash MUST NOT appear in Sources and MUST NOT appear in ModelsFrom

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
        type enum
        string name
        string description
        float truth
        jsonb ancestors
    }
    ModelsFrom {
        bytes parent PK,FK
        bytes child PK,FK
    }

    Sources |o--|| Models : ""
    Sources ||--o{ Clusters : ""
    Clusters ||--o{ Contains : "parent, child"
    Clusters ||--o{ Probabilities : ""
    Models ||--o{ Probabilities : ""
    Models ||--o{ ModelsFrom : "child, parent"
```
