"""Contracts that keep the orchestration layer from failing silently.

These tests pin the failure-mode guarantees added across the registry, Auto
factories, pipeline entry points, workflow engine, and simulation adapters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sciai.auto.factory import AutoModel
from sciai.auto.providers import ModelProvider
from sciai.domains import load_builtin_plugins
from sciai.domains.base import DomainPlugin, discover_plugins
from sciai.domains.fluid.data import FlowField
from sciai.domains.fluid.model import IncompressibilityModel, PINNSolver
from sciai.exceptions import ArtifactNotFoundError, RegistryError, ValidationError
from sciai.models import ScienceModel
from sciai.pipelines import pipeline
from sciai.registry import Registry, model_registry
from sciai.simulation import CoupledSimulator, InMemoryAdapter
from sciai.simulation.base import SimulationAdapter
from sciai.workflow import ScienceWorkflow

# --------------------------------------------------------------------------- #
# Registry / DomainPlugin
# --------------------------------------------------------------------------- #


def test_registry_conflict_raises() -> None:
    registry: Registry[int] = Registry("test-reg")
    registry.register("a", 1)
    with pytest.raises(RegistryError):
        registry.register("a", 2)


def test_plugin_register_is_idempotent_for_same_object() -> None:
    load_builtin_plugins()
    plugin = DomainPlugin(
        name="idem-plugin",
        version="0.0.1",
        description="idempotent plugin",
        models={
            "idem-model": ModelProvider(
                factory=IncompressibilityModel, domain="fluid", tasks=("t",)
            )
        },
    )
    plugin.register()
    plugin.register()  # same object -> no error
    assert model_registry.contains("idem-model")


def test_plugin_register_conflicts_on_different_object() -> None:
    load_builtin_plugins()
    first = DomainPlugin(
        name="conflict-plugin",
        version="0.0.1",
        description="x",
        models={
            "conflict-model": ModelProvider(
                factory=IncompressibilityModel, domain="fluid", tasks=("t",)
            )
        },
    )
    first.register()
    second = DomainPlugin(
        name="conflict-plugin",
        version="0.0.1",
        description="x",
        models={
            "conflict-model": ModelProvider(
                factory=IncompressibilityModel, domain="fluid", tasks=("t",)
            )
        },
    )
    with pytest.raises(RegistryError):
        second.register()


def test_plugin_register_preflight_avoids_partial_write() -> None:
    load_builtin_plugins()
    unique_key = "preflight-unique-model"
    assert not model_registry.contains(unique_key)
    plugin = DomainPlugin(
        name="preflight-plugin",
        version="0.0.1",
        description="x",
        models={
            unique_key: ModelProvider(
                factory=IncompressibilityModel, domain="fluid", tasks=("t",)
            ),
            # Already registered by the builtin fluid plugin -> must abort before writes.
            "incompressibility-residual": ModelProvider(
                factory=IncompressibilityModel, domain="fluid", tasks=("t",)
            ),
        },
    )
    with pytest.raises(RegistryError):
        plugin.register()
    # The preflight pass must have prevented the non-conflicting key from being written.
    assert not model_registry.contains(unique_key)
    assert not model_registry.contains("preflight-plugin")


def test_discover_plugins_surfaces_entry_name(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenEntryPoint:
        name = "broken-plugin"

        @staticmethod
        def load() -> object:
            raise RuntimeError("boom")

    class _EntryPoints:
        def __init__(self, points: list[object]) -> None:
            self._points = points

        def __iter__(self) -> object:
            return iter(self._points)

    monkeypatch.setattr(
        "sciai.domains.base.entry_points",
        lambda *, group: _EntryPoints([_BrokenEntryPoint()]),
    )
    with pytest.raises(RegistryError) as exc_info:
        discover_plugins()
    assert "broken-plugin" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert "boom" in str(exc_info.value.__cause__)


# --------------------------------------------------------------------------- #
# AutoModel local artifacts
# --------------------------------------------------------------------------- #


def test_automodel_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "model.json"
    file_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactNotFoundError):
        AutoModel.from_pretrained(file_path)


@pytest.mark.parametrize(
    "manifest",
    [
        "not json at all",
        "[1, 2, 3]",
        json.dumps({"config": {}}),
        json.dumps({"model_type": 5, "config": {}}),
        json.dumps({"model_type": "x", "config": [1, 2]}),
    ],
)
def test_automodel_rejects_invalid_manifest(tmp_path: Path, manifest: str) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValidationError):
        AutoModel.from_pretrained(artifact)


def test_automodel_loads_valid_local_artifact(tmp_path: Path) -> None:
    load_builtin_plugins()
    artifact = tmp_path / "artifact"
    IncompressibilityModel(reduction="mean").save(artifact)
    model = AutoModel.from_pretrained(artifact)
    assert isinstance(model, ScienceModel)
    assert model.model_type == "incompressibility-residual"


# --------------------------------------------------------------------------- #
# pipeline entry points
# --------------------------------------------------------------------------- #


def test_pipeline_rejects_empty_task() -> None:
    with pytest.raises(ArtifactNotFoundError):
        pipeline("", model="incompressibility-residual")


def test_pipeline_requires_science_model() -> None:
    with pytest.raises(ValidationError):
        pipeline("conservation-check", model=object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# workflow validation
# --------------------------------------------------------------------------- #


def test_workflow_rejects_self_reference() -> None:
    workflow = ScienceWorkflow("self")
    workflow.add_node("a", lambda x: x, inputs={"x": "a"})
    with pytest.raises(ValidationError):
        workflow.execution_order()


def test_workflow_rejects_invalid_input_path() -> None:
    workflow = ScienceWorkflow("paths")
    with pytest.raises(ValidationError):
        workflow.add_node("a", lambda x: x, inputs={"x": "input."})


def test_workflow_forbids_parameters_overriding_inputs() -> None:
    workflow = ScienceWorkflow("override")
    workflow.add_node(
        "double",
        lambda value: value * 2,
        inputs={"value": "input.value"},
        parameters={"value": 99},
    )
    with pytest.raises(ValidationError):
        workflow.run(value=5)


def test_workflow_to_dict_rejects_non_json_parameters() -> None:
    workflow = ScienceWorkflow("json")
    workflow.add_node("a", lambda x: x, inputs={"x": "input.value"}, parameters={"obj": object()})
    with pytest.raises(ValidationError):
        workflow.to_dict()


def test_workflow_wraps_nested_resolution_errors() -> None:
    workflow = ScienceWorkflow("nested")
    workflow.add_node("a", lambda x: x, inputs={"x": "input.value.missing"})
    with pytest.raises(ValidationError):
        workflow.run(value={"value": {}})


# --------------------------------------------------------------------------- #
# simulation adapters
# --------------------------------------------------------------------------- #


class _FailingInitAdapter(SimulationAdapter):
    def __init__(self, name: str = "fail") -> None:
        super().__init__()
        self.name = name
        self.domain = "general"

    def initialize(self, context: object = None) -> None:  # noqa: ANN401
        raise ValidationError("init boom")

    def step(self, step: int, time: object = None) -> None:  # noqa: ANN401
        pass

    def read(self, variables: tuple[str, ...]) -> dict[str, object]:
        return {}

    def write(self, values: dict[str, object]) -> None:  # noqa: ANN401
        pass


def test_simulator_finalizes_initialized_adapters_on_init_failure() -> None:
    good = InMemoryAdapter("good", "general", {"x": 1.0})
    failing = _FailingInitAdapter("fail")
    simulator = CoupledSimulator([good, failing])
    with pytest.raises(ValidationError) as exc_info:
        simulator.run(steps=1)
    assert "init boom" in str(exc_info.value)
    # Only the adapter that actually initialized is finalized; the failed one is untouched.
    assert good.state.value == "finished"
    assert failing.state.value == "created"


def test_simulator_steps_and_timestep_validation() -> None:
    adapter = InMemoryAdapter("a", "general", {"x": 1.0})

    def _run(steps: object = 1, time_step: object = None) -> None:  # noqa: ANN401
        CoupledSimulator([adapter]).run(steps=steps, time_step=time_step)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        _run(steps=0)
    with pytest.raises(ValidationError):
        _run(steps="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _run(time_step=-1.0)
    with pytest.raises(ValidationError):
        _run(time_step=0.0)
    with pytest.raises(ValidationError):
        _run(time_step="x")  # type: ignore[arg-type]


def test_simulator_exchange_is_order_invariant() -> None:
    a = InMemoryAdapter("a", "general", {"x": 1.0})
    b = InMemoryAdapter("b", "general", {"y": 2.0})

    forward = CoupledSimulator([a, b])
    forward.couple("a", "b", variables=["x"])
    forward.couple("b", "a", variables=["y"])
    result_forward = forward.run(steps=2)

    c = InMemoryAdapter("a", "general", {"x": 1.0})
    d = InMemoryAdapter("b", "general", {"y": 2.0})
    reverse = CoupledSimulator([c, d])
    reverse.couple("b", "a", variables=["y"])
    reverse.couple("a", "b", variables=["x"])
    result_reverse = reverse.run(steps=2)

    assert result_forward.adapter_states == result_reverse.adapter_states
    assert result_forward.adapter_states["b"]["x"] == 1.0
    assert result_forward.adapter_states["a"]["y"] == 2.0


def test_simulator_supports_gauss_seidel_scheme() -> None:
    a = InMemoryAdapter("a", "general", {"x": 1.0})
    b = InMemoryAdapter("b", "general", {"y": 2.0})
    simulator = CoupledSimulator([a, b])
    simulator.couple("a", "b", variables=["x"])
    simulator.couple("b", "a", variables=["y"])
    result = simulator.run(steps=1, scheme="gauss-seidel")
    assert result.steps == 1


def test_adapter_read_write_require_active_state() -> None:
    adapter = InMemoryAdapter("a", "general", {"x": 1.0})
    with pytest.raises(ValidationError):
        adapter.read(("x",))
    with pytest.raises(ValidationError):
        adapter.write({"x": 2.0})


def test_generic_adapter_snapshot_is_empty() -> None:
    adapter = _FailingInitAdapter("generic")
    assert dict(adapter.snapshot()) == {}


def test_in_memory_snapshot_returns_copy() -> None:
    adapter = InMemoryAdapter("a", "general", {"x": 1.0})
    snapshot = adapter.snapshot()
    snapshot["x"] = 999.0
    assert adapter.values["x"] == 1.0


# --------------------------------------------------------------------------- #
# FlowField
# --------------------------------------------------------------------------- #


def test_flowfield_component_axis_not_at_end() -> None:
    velocity = np.zeros((3, 9, 9))
    pressure = np.zeros((9, 9))
    field = FlowField.from_arrays(velocity, pressure=pressure, component_axis=0)
    assert field.component_axis == 0
    # A mismatched pressure grid must be rejected.
    with pytest.raises(ValidationError):
        FlowField.from_arrays(velocity, pressure=np.zeros((3, 9)), component_axis=0)


def test_flowfield_from_file_rejects_non_npz(tmp_path: Path) -> None:
    bad = tmp_path / "flow.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValidationError):
        FlowField.from_file(bad)


def test_flowfield_from_file_requires_consecutive_coords(tmp_path: Path) -> None:
    axis = np.linspace(0.0, 1.0, 9)
    velocity = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    good = tmp_path / "good.npz"
    np.savez(good, velocity=velocity, coord_0=axis, coord_1=axis)

    gapped = tmp_path / "gapped.npz"
    np.savez(gapped, velocity=velocity, coord_0=axis, coord_2=axis)
    with pytest.raises(ValidationError):
        FlowField.from_file(gapped)

    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, velocity=velocity, coord_x=axis)
    with pytest.raises(ValidationError):
        FlowField.from_file(malformed)

    field = FlowField.from_file(good)
    assert len(field.coordinates) == 2


# --------------------------------------------------------------------------- #
# PINNSolver
# --------------------------------------------------------------------------- #


def test_pinnsolver_requires_non_empty_physics() -> None:
    with pytest.raises(ValidationError):
        PINNSolver(physics="")


def test_pinnsolver_rejects_non_flowfield_backend_result() -> None:
    flow = FlowField.from_arrays(np.zeros((9, 9, 2)), coordinates=(np.arange(9), np.arange(9)))

    class _BadBackend:
        def solve(self, flow: FlowField, *, physics: str, boundary_conditions: object) -> object:  # noqa: ANN401
            return {"not": "a flow field"}

    solver = PINNSolver(physics="navier-stokes", backend=_BadBackend())
    with pytest.raises(ValidationError):
        solver.solve(flow, boundary_conditions={})


def test_pinnsolver_accepts_flowfield_backend_result() -> None:
    flow = FlowField.from_arrays(np.zeros((9, 9, 2)), coordinates=(np.arange(9), np.arange(9)))

    class _GoodBackend:
        def solve(self, flow: FlowField, *, physics: str, boundary_conditions: object) -> FlowField:  # noqa: ANN401
            return flow

    solver = PINNSolver(physics="navier-stokes", backend=_GoodBackend())
    assert solver.solve(flow, boundary_conditions={}) is flow
