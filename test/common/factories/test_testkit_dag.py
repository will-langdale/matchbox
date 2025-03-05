import pytest

from matchbox.common.factories.dags import TestkitDAG
from matchbox.common.factories.entities import FeatureConfig
from matchbox.common.factories.models import ModelTestkit, model_factory
from matchbox.common.factories.sources import (
    SourceConfig,
    linked_sources_factory,
    source_factory,
)


@pytest.mark.parametrize(
    ("chain_config", "expected_sources_by_model"),
    [
        pytest.param(
            [
                {
                    "name": "deduper_crn",
                    "sources": ["crn"],
                    "previous_models": [],
                    "standalone": False,
                }
            ],
            {
                "deduper_crn": ["crn"],
            },
            id="simple_deduper_finds_linked",
        ),
        pytest.param(
            [
                {
                    "name": "deduper_cdms",
                    "sources": ["cdms"],
                    "previous_models": [],
                    "standalone": False,
                }
            ],
            {
                "deduper_cdms": ["cdms"],
            },
            id="another_deduper_finds_linked",
        ),
        pytest.param(
            [
                {
                    "name": "standalone_deduper",
                    "sources": ["standalone_source"],
                    "previous_models": [],
                    "standalone": True,
                }
            ],
            {
                "standalone_deduper": [],
            },
            id="standalone_source_no_linked",
        ),
        pytest.param(
            [
                {
                    "name": "deduper_crn",
                    "sources": ["crn"],
                    "previous_models": [],
                    "standalone": False,
                },
                {
                    "name": "deduper_cdms",
                    "sources": ["cdms"],
                    "previous_models": [],
                    "standalone": False,
                },
                {
                    "name": "linker_models",
                    "sources": [],
                    "previous_models": ["deduper_crn", "deduper_cdms"],
                    "standalone": False,
                },
            ],
            {
                "deduper_crn": ["crn"],
                "deduper_cdms": ["cdms"],
                "linker_models": ["cdms", "crn"],
            },
            id="deduper_deduper_linker_chain",
        ),
        pytest.param(
            [
                {
                    "name": "deduper_crn",
                    "sources": ["crn"],
                    "previous_models": [],
                    "standalone": False,
                },
                {
                    "name": "linker_model_cdms",
                    "sources": ["cdms"],
                    "previous_models": ["deduper_crn"],
                    "standalone": False,
                },
            ],
            {
                "deduper_crn": ["crn"],
                "linker_model_cdms": ["cdms", "crn"],
            },
            id="deduper_linker_chain",
        ),
        pytest.param(
            [
                {
                    "name": "linker_crn_cdms",
                    "sources": ["crn", "cdms"],
                    "previous_models": [],
                    "standalone": False,
                }
            ],
            {
                "linker_crn_cdms": ["cdms", "crn"],
            },
            id="linker_finds_linked",
        ),
    ],
)
def test_testkit_dag_model_chain(
    chain_config: list[dict],
    expected_sources_by_model: dict[str, list[str]],
) -> None:
    """Test TestkitDAG with different model configurations including standalone sources.

    This test verifies that:
    1. We can create different types of models: single source dedupers, linkers, chains
    2. The DAG correctly tracks model dependencies
    3. We can find the original LinkedSourcesTestkit from any model in the chain
    4. We can identify the correct sources for each model
    5. Standalone sources are handled correctly
    """
    # Create linked sources
    linked = linked_sources_factory(seed=hash(str(chain_config)) % 1000000)
    all_true_sources = tuple(linked.true_entities.values())

    # Create standalone source (not part of linked)
    standalone = source_factory(
        full_name="standalone_source",
        features=[
            {"name": "name", "base_generator": "name"},
            {"name": "age", "base_generator": "random_int"},
        ],
    )

    # Create DAG
    dag = TestkitDAG()
    dag.add_source(linked)
    dag.add_source(standalone)

    # Build the chain of models
    models: dict[str, ModelTestkit] = {}

    for model_config in chain_config:
        model_name = model_config["name"]
        source_names = model_config["sources"]
        previous_model_names = model_config["previous_models"]
        is_standalone = model_config.get("standalone", False)

        # Determine left and right inputs
        left_testkit = None
        right_testkit = None

        # Set up left input (required for all models)
        if previous_model_names:
            left_testkit = models[previous_model_names[0]]
        elif is_standalone:
            left_testkit = standalone
        elif source_names:
            left_testkit = linked.sources[source_names[0]]

        # Set up right input (optional, for linkers)
        if previous_model_names and len(previous_model_names) >= 2:
            right_testkit = models[previous_model_names[1]]
        elif source_names and (
            len(source_names) >= 2 or (previous_model_names and source_names)
        ):
            source_index = 1 if len(source_names) >= 2 else 0
            right_testkit = linked.sources[source_names[source_index]]

        # Create the model
        model = model_factory(
            name=model_name,
            left_testkit=left_testkit,
            right_testkit=right_testkit,
            true_entities=all_true_sources,
        )

        # Add model to DAG and dictionary
        dag.add_model(model)
        models[model_name] = model

    # Verify all models were added
    assert len(models) == len(chain_config)
    assert len(dag.models) == len(chain_config), "DAG should have all models"

    # Test each model in the chain
    for model_name, expected_sources in expected_sources_by_model.items():
        model = models[model_name]
        linked_testkit, sources = dag.get_sources_for_model(model.name)

        # Check if we expect a linked testkit for this model
        if not expected_sources:  # Standalone model case
            assert linked_testkit is None, (
                f"Model {model_name} should not find linked sources testkit"
            )
        else:
            assert linked_testkit is not None, (
                f"Model {model_name} should find linked sources testkit"
            )
            assert linked_testkit is linked, (
                f"Model {model_name} should find the original linked testkit"
            )

        # Verify sources match expected
        assert sorted(sources) == sorted(expected_sources), (
            f"Model {model_name} should have sources {expected_sources} "
            f"but got {sources}"
        )


def test_testkit_dag_multiple_linked_sources():
    """Test handling of multiple LinkedSourcesTestkit objects."""
    # Create two separate linked sources testkits
    features = [
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="email", base_generator="email"),
        FeatureConfig(name="address", base_generator="address"),
    ]

    configs1 = (
        SourceConfig(full_name="foo1", features=tuple(features[:1])),
        SourceConfig(full_name="foo2", features=tuple(features[:1])),
    )
    configs2 = (
        SourceConfig(full_name="bar1", features=tuple(features[1:])),
        SourceConfig(full_name="bar2", features=tuple(features[1:])),
    )

    linked1 = linked_sources_factory(source_configs=configs1, n_true_entities=10)
    linked2 = linked_sources_factory(source_configs=configs2, n_true_entities=10)

    # Create DAG and add both linked testkits
    dag = TestkitDAG()
    dag.add_source(linked1)
    dag.add_source(linked2)

    # Create models from each linked testkit
    model1 = model_factory(
        left_testkit=linked1.sources["foo1"],
        true_entities=tuple(linked1.true_entities.values()),
        seed=42,
    )

    model2 = model_factory(
        left_testkit=linked2.sources["bar1"],
        true_entities=tuple(linked2.true_entities.values()),
        seed=43,
    )

    # Add models to DAG
    dag.add_model(model1)
    dag.add_model(model2)

    # Test sources for each model to find their linked testkits
    linked_for_model1, sources_model1 = dag.get_sources_for_model(model1.name)
    linked_for_model2, sources_model2 = dag.get_sources_for_model(model2.name)

    assert linked_for_model1 is linked1, "Model1 should be linked to linked1"
    assert linked_for_model2 is linked2, "Model2 should be linked to linked2"
    assert linked_for_model1 is not linked_for_model2, (
        "Should not mix up the linked testkits"
    )

    # Verify sources
    assert sources_model1 == ["foo1"], (
        f"Model1 should use source 'foo1' but got {sources_model1}"
    )
    assert sources_model2 == ["bar1"], (
        f"Model2 should use source 'bar1' but got {sources_model2}"
    )

    # Create a linker model that combines both sources from linked1
    model3 = model_factory(
        left_testkit=linked1.sources["foo1"],
        right_testkit=linked1.sources["foo2"],
        true_entities=tuple(linked1.true_entities.values()),
        seed=44,
    )

    # Add the linker model to DAG
    dag.add_model(model3)

    # Test sources for linker model
    _, sources_model3 = dag.get_sources_for_model(model3.name)
    assert sorted(sources_model3) == ["foo1", "foo2"], (
        f"Linker model should use both 'foo1' and 'foo2' but got {sources_model3}"
    )
