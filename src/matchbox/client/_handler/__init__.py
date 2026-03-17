"""Functions abstracting the interaction with the server API."""

from matchbox.client._handler.admin import (
    count_backend_items,
    delete_orphans,
)
from matchbox.client._handler.auth import auth_status, login
from matchbox.client._handler.collections import (
    create_collection,
    create_run,
    create_step,
    delete_collection,
    delete_run,
    delete_step,
    get_collection,
    get_collection_permissions,
    get_data,
    get_run,
    get_step,
    get_step_stage,
    grant_collection_permission,
    list_collections,
    revoke_collection_permission,
    set_data,
    set_run_default,
    set_run_mutable,
    update_step,
)
from matchbox.client._handler.eval import (
    download_eval_data,
    sample_for_eval,
    send_eval_judgement,
)
from matchbox.client._handler.groups import (
    add_user_to_group,
    create_group,
    delete_group,
    get_group,
    list_groups,
    remove_user_from_group,
)
from matchbox.client._handler.main import healthcheck
from matchbox.client._handler.query import match, query

__all__ = [
    # auth
    "auth_status",
    "login",
    # admin
    "count_backend_items",
    "delete_orphans",
    # groups
    "add_user_to_group",
    "create_group",
    "delete_group",
    "get_group",
    "list_groups",
    "remove_user_from_group",
    # main
    "healthcheck",
    # eval
    "download_eval_data",
    "sample_for_eval",
    "send_eval_judgement",
    # query
    "query",
    "match",
    # collections
    "create_collection",
    "create_step",
    "create_run",
    "delete_collection",
    "delete_step",
    "delete_run",
    "get_collection",
    "get_collection_permissions",
    "get_data",
    "get_step",
    "get_step_stage",
    "get_run",
    "grant_collection_permission",
    "list_collections",
    "revoke_collection_permission",
    "set_data",
    "set_run_default",
    "set_run_mutable",
    "update_step",
]
