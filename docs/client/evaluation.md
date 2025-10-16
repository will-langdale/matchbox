# Evaluate model performance

After running models to deduplicate and link your data, you’ll likely want to check how well they’re performing. Matchbox helps with this by using **precision** and **recall**:

* **Precision**: How many of the matches your model made are actually correct?
* **Recall**: How many of the correct matches did your model successfully find?

To calculate these, we need a **ground truth** - a set of correct matches created by people. This is called **validation data**.

## Creating validation data

Matchbox provides a terminal-based UI to help you create this validation data. Here's how it works:

1. **Launch the evaluation tool** using `matchbox eval start --resolution <resolution_name> --samples <number> --user <your_username>`
    a. You can also set yours username with the `MB__CLIENT__USER` environment variable.
2. Matchbox will **download clusters** for you to review. It avoids clusters you've already judged and focuses on ones the model is unsure about.
3. In the terminal interface, you'll review each cluster:
   * Use keyboard commands like `b+1,3,5` to assign rows 1, 3, and 5 to group B
   * Use extended groups like `aa+7` for more complex splits
   * Navigate with arrow keys between entities
   * Press `?` or `F1` for help with commands

Once enough users have reviewed clusters, this data can be used to evaluate model performance.

## How precision and recall are calculated

Validation data is created at the **cluster level**, but precision and recall are calculated using **pairs of records**.

For example, if a model links records A, B, C, and D into one cluster, it implies these pairs:

> A-B, A-C, A-D, B-C, B-D, C-D

If a user says only A, B, and C belong together, the correct pairs are:

> A-B, A-C, B-C

So, the model found all the correct pairs (good recall), but also included some incorrect ones (lower precision).

If users disagree on a pair:

* If more users approve it than reject it → it’s considered **correct**.
* If more reject it → it’s **incorrect**.
* If it’s a tie → it’s **ignored** in the calculation.

Only pairs that appear in both the model and the validation data are used in the evaluation.

!!! tip "Relative vs. absolute scores"
    If your model builds on others, it may inherit some incorrect pairs it can’t control. So, precision and recall scores aren’t absolute - they’re best used to **compare models or thresholds**.

## Tuning your model’s threshold

Choosing the right threshold for your model involves balancing precision and recall. A higher threshold usually means:

* **Higher precision** (fewer false matches)
* **Lower recall** (more missed matches)

To evaluate this in code, first you need to build and run a model outside of a DAG:

```python
from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.sources import Source
from matchbox.client.models.dedupers import NaiveDeduper
from sqlalchemy import create_engine

engine = create_engine('postgresql://')

dag = DAG(name="companies").new_run()

source = dag.source(...) # source parameters must be completed

model = source.query().deduper(
    name="model_name",
    description=f"description",
    model_class=NaiveDeduper,
    model_settings={
        "unique_fields": ["field1", "field2"],
    },
)

results = model.run()
```

Download validation data and create evaluation object, then get precision and recall at a specific threshold:

```python
from matchbox.client.cli.eval import EvalData
eval_data = EvalData()

# Get precision and recall at a specific threshold
precision, recall = eval_data.precision_recall(results, threshold=0.5)
print(f"At threshold 0.5: Precision={precision:.2f}, Recall={recall:.2f}")
```

!!! tip "Deterministic models"
    Some types of model (like the `NaiveDeduper` used in the example) only output 1s for the matches they make, hence **threshold truth tuning doesn't apply**:

    * You won't get a precision-recall curve, but a single point at threshold 1.0.
    * The precision and recall scores will be the same at all thresholds.

    On the other hand, probabilistic models (like `SplinkLinker`), can output **any value between 0.0 and 1.0**.
    