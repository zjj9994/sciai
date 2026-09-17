import pytest

from sciai import ScienceWorkflow
from sciai.exceptions import ValidationError
from sciai.simulation import CoupledSimulator, InMemoryAdapter


def test_multiphysics_exchange() -> None:
    fluid = InMemoryAdapter("fluid", "fluid", {"pressure": 2.0})
    solid = InMemoryAdapter(
        "solid",
        "solid",
        {"displacement": 0.0},
        update=lambda values, **_: {"displacement": values["displacement"] + 1.0},
    )
    simulation = CoupledSimulator([fluid, solid])
    simulation.couple("fluid", "solid", variables=["pressure"])
    simulation.couple("solid", "fluid", variables=["displacement"])
    result = simulation.run(steps=3)
    assert result.steps == 3
    assert result.adapter_states["solid"]["pressure"] == 2.0
    assert result.adapter_states["fluid"]["displacement"] == 3.0


def test_workflow_cross_domain_dag() -> None:
    workflow = ScienceWorkflow("demo")
    workflow.add_node(
        "double",
        lambda value: value * 2,
        inputs={"value": "input.value"},
        domain="math-physics",
    )
    workflow.add_node(
        "offset",
        lambda value, amount: value + amount,
        inputs={"value": "double"},
        parameters={"amount": 3},
        domain="engineering",
    )
    result = workflow.run(value=5)
    assert result.outputs["offset"] == 13
    assert result.execution_order == ("double", "offset")


def test_workflow_cycle_is_rejected() -> None:
    workflow = ScienceWorkflow("cycle")
    workflow.add_node("a", lambda value: value, inputs={"value": "b"})
    workflow.add_node("b", lambda value: value, inputs={"value": "a"})
    with pytest.raises(ValidationError):
        workflow.execution_order()
