# 🔥 Matchbox PostgreSQL backend

A backend adapter for deploying Matchbox using PostgreSQL.

Currently implements the following architecture.

```mermaid
erDiagram
    %% Data
    SourceDataset {
        uuid uuid PK
        string schema
        string table
        string id
    }
    SourceData {
        bytes sha1 PK
        string id
        string dataset FK
    }
    %% Deduplication
    Dedupes {
        bytes sha1 PK
        bytes left FK
        bytes right FK
    }
    DDupeProbabilities {
        bytes ddupe PK,FK
        bytes model PK,FK
        float probability
    }
    DDupeContains {
        bytes child PK,FK
        bytes parent PK,FK
    }
    DDupeValidation {
        uuid uuid PK
        bytes ddupe FK
        string user
        boolean valid
    }
    %% Linking
    Links {
        bytes sha1 PK
        bytes left FK
        bytes right FK
    }
    LinkProbabilities {
        bytes link PK,FK
        bytes model PK,FK
        float probability
    }
    LinkContains {
        bytes child PK,FK
        bytes parent PK,FK
    }
    LinkValidation {
        uuid uuid PK
        bytes link FK
        string user
        boolean valid
    }
    %% Clusters
    Clusters {
        bytes sha1 PK
    }
    clusters_association {
        bytes child PK,FK
        bytes parent PK,FK
    }
    ClusterValidation {
        uuid uuid PK
        bytes cluster FK
        string user
        boolean valid
    }
    %% Models
    Models {
        bytes sha1 PK
        string name
        string description
        uuid deduplicates FK
    }
    ModelsFrom {
        bytes child PK,FK
        bytes parent PK,FK
    }

    %% Data
    SourceDataset ||--o{ SourceData : ""
    SourceDataset ||--o{ Models : ""
    SourceData ||--o{ DDupeContains : ""
    SourceData ||--o{ Dedupes : "left, right"
    %% Deduplication
    Dedupes ||--o{ DDupeProbabilities : ""
    Dedupes ||--o{ DDupeValidation : ""
    DDupeContains }o--o{ Clusters : ""
    %% Linking
    Links ||--o{ LinkProbabilities : ""
    Links ||--o{ LinkValidation : ""
    %% Clusters
    Clusters ||--o{ clusters_association : ""
    Clusters ||--o{ ClusterValidation : ""
    Clusters ||--o{ Links : "left, right"
    Clusters ||--o{ LinkContains : "child, parent"
    %% Models
    Models ||--o{ DDupeProbabilities : ""
    Models ||--o{ LinkProbabilities : ""
    Models ||--o{ clusters_association : ""
    Models ||--o{ ModelsFrom: "child, parent"
```

## 🔥 Matchbox 0.2

I think this could be expressed more simply. See Confluence for [further discussion of this proposal](https://uktrade.atlassian.net/wiki/spaces/CDL/pages/4282908700/Matchbox+0.2+architecture+ideas).

Here we register every new dataset in the models table.

`Models.type` is one of "model", "dataset" or "human". This slight entity drift in the models table allows other tables to remain pure.

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
