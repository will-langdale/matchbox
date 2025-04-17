# API reference

`matchbox.client` is the client used to interact with the [Matchbox server](../../server/install.md).

All names in `matchbox.client` are also accessible from the top-level `matchbox` module.

::: matchbox.client
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        extra:
            show_root_docstring: true
        filters:
            - "!^[A-Z]$"  # Excludes single-letter uppercase variables (like T, P, R)
            - "!^_"       # Excludes private attributes
            - "!_logger$"  # Excludes logger variables
            - "!_path$"    # Excludes path variables
            - "!model_config" # Excludes Pydantic configuration
            - "!app$"    # Excludes FastAPI app