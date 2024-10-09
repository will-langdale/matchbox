from sqlalchemy import Engine
from sqlalchemy.orm import Session

from matchbox.data import ENGINE, Models
from matchbox.helpers.selector import get_all_parents


def delete_model(model: str, engine: Engine = ENGINE, certain: bool = False) -> None:
    """
    Deletes:

    * The model from the model table
    * The model's edges to its child models from the models_from table
    * The creates edges the model made from the clusters_association table
    * Any probability values associated with the model from the ddupe_probabilities and
        link_probabilities tables
    * All of the above for all parent models. As every model is defined by
        its children, deleting a model means cascading deletion to all ancestors

    It DOESN'T delete the raw clusters or probability nodes from the ddupes and links
    tables, which retain any validation attached to them.
    """
    with Session(engine) as session:
        target_model = session.query(Models).filter_by(name=model).first()
        all_parents = get_all_parents(target_model)
        if certain:
            for m in all_parents:
                session.delete(m)
            session.commit()
        else:
            raise ValueError(
                "This operation will delete the models "
                f"{', '.join([m.name for m in all_parents])}, as well as all "
                "references to clusters and probabilities they have created."
                "\n\n"
                "It will not delete validation associated with these "
                "clusters or probabilities."
                "\n\n"
                "If you're sure you want to continue, rerun with certain=True"
            )
