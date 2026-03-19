# PostgreSQL

A backend adapter for deploying Matchbox with PostgreSQL.

This backend stores two connected structures:

- An execution graph in `steps` and `step_from`, covering sources, models, and resolvers.
- A data graph in `clusters`, `contains`, `model_edges`, `resolver_clusters`, and `cluster_source_key`.

Source steps index source clusters. Model steps store score edges between clusters. Resolver steps point to the clusters that form a published entity view.

```mermaid
erDiagram
    Collections {
        bigint collection_id PK
        text name
    }
    Runs {
        bigint run_id PK
        bigint collection_id FK
        boolean is_mutable
        boolean is_default
    }
    Steps {
        bigint step_id PK
        bigint run_id FK
        text name
        text description
        text type
        bytea fingerprint
        enum upload_stage
    }
    StepFrom {
        bigint parent PK,FK
        bigint child PK,FK
        integer level
    }
    SourceConfigs {
        bigint source_config_id PK
        bigint step_id FK
        text location_type
        text location_name
        text extract_transform
    }
    SourceFields {
        bigint field_id PK
        bigint source_config_id FK
        integer index
        text name
        text type
        boolean is_key
    }
    ModelConfigs {
        bigint model_config_id PK
        bigint step_id FK
        text model_class
        jsonb model_settings
        jsonb left_query
        jsonb right_query
    }
    ResolverConfigs {
        bigint resolver_config_id PK
        bigint step_id FK
        text resolver_class
        jsonb resolver_settings
    }
    Clusters {
        bigint cluster_id PK
        bytea cluster_hash
    }
    ClusterSourceKey {
        bigint key_id PK
        bigint cluster_id FK
        bigint source_config_id FK
        text key
    }
    Contains {
        bigint root PK,FK
        bigint leaf PK,FK
    }
    ModelEdges {
        bigint result_id PK
        bigint step_id FK
        bigint left_id FK
        bigint right_id FK
        real score
    }
    ResolverClusters {
        bigint step_id PK,FK
        bigint cluster_id PK,FK
    }
    Users {
        bigint user_id PK
        text name
        text email
    }
    Groups {
        bigint group_id PK
        text name
        text description
        boolean is_system
    }
    UserGroups {
        bigint user_id PK,FK
        bigint group_id PK,FK
    }
    Permissions {
        bigint permission_id PK
        text permission
        bigint group_id FK
        bigint collection_id FK
        boolean is_system
    }
    EvalJudgements {
        bigint judgement_id PK
        bigint user_id FK
        bigint endorsed_cluster_id FK
        bigint shown_cluster_id FK
        datetime timestamp
    }

    Collections ||--o{ Runs : ""
    Collections ||--o{ Permissions : ""
    Runs ||--o{ Steps : ""
    Steps ||--o{ StepFrom : "parent"
    StepFrom }o--|| Steps : "child"
    Steps |o--|| SourceConfigs : ""
    Steps |o--|| ModelConfigs : ""
    Steps |o--|| ResolverConfigs : ""
    Steps ||--o{ ModelEdges : ""
    Steps ||--o{ ResolverClusters : ""
    SourceConfigs ||--o{ SourceFields : ""
    SourceConfigs ||--o{ ClusterSourceKey : ""
    Clusters ||--o{ ClusterSourceKey : ""
    Clusters ||--o{ Contains : "root"
    Contains }o--|| Clusters : "leaf"
    Clusters ||--o{ ModelEdges : "left_id"
    Clusters ||--o{ ModelEdges : "right_id"
    Clusters ||--o{ ResolverClusters : ""
    Clusters ||--o{ EvalJudgements : "endorsed_cluster_id"
    Clusters ||--o{ EvalJudgements : "shown_cluster_id"
    Users ||--o{ UserGroups : ""
    Users ||--o{ EvalJudgements : ""
    Groups ||--o{ UserGroups : ""
    Groups ||--o{ Permissions : ""
```

::: matchbox.server.postgresql
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        show_submodules: true
        extra:
            show_root_docstring: true
