# Evaluating resolver output

After you have built a resolver, the next question is whether its clusters are useful. Matchbox evaluates resolvers with precision and recall against human-labelled validation data.

- Precision asks how many of the matches your resolver returned are correct.
- Recall asks how many of the correct matches your resolver found.

Validation data is created at cluster level, then converted into record pairs for scoring.

## Creating a sample set

Evaluation starts with a set of clusters for reviewers to inspect.

### Sampling one resolver

=== "Example"
    ```python
    dag = ...  # your DAG, defined elsewhere

    dag.get_matches().as_dump().write_parquet("samples.pq")
    ```

`dag.get_matches()` samples the default resolver. Pass `resolver="resolver_name"` if you want a specific resolver instead.

### Sampling more than one resolver

If you want to compare resolvers, merge their cluster views and review the union.

=== "Example"
    ```python
    dag = ...  # your DAG, defined elsewhere

    baseline = dag.get_matches(resolver="resolver_baseline")
    candidate = dag.get_matches(resolver="resolver_candidate")

    baseline.merge(candidate).as_dump().write_parquet("samples.pq")
    ```

## Creating validation data

Matchbox ships with a terminal UI for collecting human judgements.

=== "Server sampling"
    ```shell
    matchbox eval \
      --collection companies \
      --resolver companies_resolver \
      --warehouse postgresql://user:password@host/database \
      --tag companies__15_02_2025
    ```

=== "Local sample file"
    ```shell
    matchbox eval \
      --collection companies \
      --file samples.pq \
      --warehouse postgresql://user:password@host/database \
      --tag companies__15_02_2025
    ```

Reviewers are shown one cluster at a time and group the records they believe belong together. The resulting judgements are stored on the Matchbox server and can be filtered by `tag`.

## Measuring precision and recall

Use [`EvalData`][matchbox.client.eval.EvalData] to download the relevant judgements and score a resolver.

=== "Example"
    ```python
    from matchbox.client.eval import EvalData

    eval_data = EvalData(tag="companies__15_02_2025")
    precision, recall = eval_data.precision_recall(my_resolver.results_eval)
    ```

`resolver.results_eval` expands resolver clusters back to server leaf IDs, which is the format the evaluation helpers expect.

!!! note "results_eval is a local helper"
    `resolver.results_eval` depends on local leaf mappings from the current Python session. Run the resolver locally, keep upstream model runs at the default `low_memory=False`, and then call it. If you load a DAG from the server, rerun the relevant steps locally first.

## Comparing resolvers

Each precision-recall point belongs to a resolver configuration. If you want to compare score cut-offs or clustering strategies, create several resolvers and evaluate each one separately.

!!! note "Deterministic inputs"
    Resolvers built only from deterministic model outputs usually collapse to a single operating point, because those models only emit `1.0` scores. In that case, threshold changes do not create a smooth precision-recall trade-off: you compare discrete resolver configurations instead.

=== "Example"
    ```python
    strict_precision, strict_recall = eval_data.precision_recall(
        strict_resolver.results_eval
    )
    balanced_precision, balanced_recall = eval_data.precision_recall(
        balanced_resolver.results_eval
    )
    ```

## How Matchbox scores validation data

Validation happens on clusters, but precision and recall are computed on implied record pairs.

If a resolver puts `A`, `B`, `C`, and `D` into one cluster, it implies these pairs:

> A-B, A-C, A-D, B-C, B-D, C-D

If a reviewer decides only `A`, `B`, and `C` belong together, the endorsed pairs are:

> A-B, A-C, B-C

That raises recall if the resolver found all three endorsed pairs, and lowers precision if it also added unsupported pairs such as `A-D`.

When reviewers disagree on a pair:

- More approvals than rejections means the pair counts as correct.
- More rejections than approvals means the pair counts as incorrect.
- A tie means the pair is ignored.

## Interpreting the result

Precision and recall are most useful for comparing resolver configurations that solve the same business problem.

- A stricter resolver usually raises precision and lowers recall.
- A looser resolver usually raises recall and lowers precision.
- A resolver built on upstream models inherits the strengths and weaknesses of those models, so upstream errors can cap the best score you can achieve.

!!! note "Relative rather than absolute"
    Treat these scores as relative rather than absolute. They are most useful when two resolvers are evaluated against the same labelled sample set.

The comparison that matters is between the resolvers you are considering for publication.
