from sqlalchemy import Engine
from sqlalchemy.orm import Session

from cmf.data import ENGINE, Models
from cmf.helpers.selector import _get_all_parents


def delete_model(model: str, engine: Engine = ENGINE, certain: bool = False) -> None:
    """
    Deletes:

    * The model from the model table
    * The creates edges the model made
    * Any models that depended on this model, and their creates edges
    * Any probability values associated with the model
    * All of the above for all parent models. As every model is defined by
        its children, deleting a model means cascading deletion to all ancestors

    It DOESN'T delete the raw clusters or probability nodes, which retain
    any validation attached to them.
    """
    with Session(engine) as session:
        target_model = session.query(Models).filter_by(name=model).first()
        all_parents = _get_all_parents(target_model)
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
