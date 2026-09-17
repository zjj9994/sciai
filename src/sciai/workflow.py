"""Directed acyclic workflows for cross-domain research tasks.

The graph is inspectable (see :meth:`ScienceWorkflow.to_dict`) but is *not*
considered serializable: node ``operation`` callables are never deserialized from
untrusted data, and rebuilding a workflow from its spec requires reconstructing those
callables explicitly.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from sciai.exceptions import ValidationError


@dataclass(frozen=True)
class WorkflowNode:
    name: str
    operation: Callable[..., Any]
    inputs: Mapping[str, str] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    domain: str = "general"


@dataclass(frozen=True)
class WorkflowResult:
    outputs: dict[str, Any]
    execution_order: tuple[str, ...]


class ScienceWorkflow:
    """A small DAG engine whose graph can back a future visual editor."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._nodes: dict[str, WorkflowNode] = {}

    def add_node(
        self,
        name: str,
        operation: Callable[..., Any],
        *,
        inputs: Mapping[str, str] | None = None,
        parameters: Mapping[str, Any] | None = None,
        domain: str = "general",
    ) -> ScienceWorkflow:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Workflow node name must be a non-empty string")
        name = name.strip()
        if not isinstance(domain, str) or not domain.strip():
            raise ValidationError(f"Workflow node {name!r} requires a non-empty domain")
        domain = domain.strip()
        if not callable(operation):
            raise ValidationError(f"Workflow node {name!r} requires a callable operation")
        node_inputs = dict(inputs or {})
        node_parameters = dict(parameters or {})
        self._validate_input_paths(name, node_inputs)
        if name in self._nodes:
            raise ValidationError(f"Workflow node {name!r} already exists")
        self._nodes[name] = WorkflowNode(
            name=name,
            operation=operation,
            inputs=node_inputs,
            parameters=node_parameters,
            domain=domain,
        )
        return self

    @staticmethod
    def _validate_input_paths(node: str, inputs: Mapping[str, str]) -> None:
        for argument, source in inputs.items():
            if not isinstance(source, str) or not source.strip():
                raise ValidationError(
                    f"Workflow node {node!r} input {argument!r} needs a non-empty source path"
                )
            if any(part == "" for part in source.split(".")):
                raise ValidationError(
                    f"Workflow node {node!r} input {argument!r} has an invalid path {source!r}; "
                    "segments must be non-empty"
                )

    def _dependencies(self) -> dict[str, set[str]]:
        dependencies: dict[str, set[str]] = {name: set() for name in self._nodes}
        for name, node in self._nodes.items():
            for source in node.inputs.values():
                dependency = source.split(".", 1)[0]
                if dependency == name:
                    raise ValidationError(f"Workflow node {name!r} references itself")
                if dependency != "input":
                    if dependency not in self._nodes:
                        raise ValidationError(
                            f"Workflow node {name!r} depends on unknown node {dependency!r}"
                        )
                    dependencies[name].add(dependency)
        return dependencies

    def execution_order(self) -> tuple[str, ...]:
        dependencies = self._dependencies()
        dependents: dict[str, set[str]] = defaultdict(set)
        for name, requirements in dependencies.items():
            for requirement in requirements:
                dependents[requirement].add(name)
        ready = deque(sorted(name for name, values in dependencies.items() if not values))
        order: list[str] = []
        while ready:
            name = ready.popleft()
            order.append(name)
            for dependent in sorted(dependents[name]):
                dependencies[dependent].remove(name)
                if not dependencies[dependent]:
                    ready.append(dependent)
        if len(order) != len(self._nodes):
            cyclic = sorted(name for name, values in dependencies.items() if values)
            raise ValidationError(f"Workflow contains a cycle involving {cyclic}")
        return tuple(order)

    @staticmethod
    def _resolve(source: str, context: Mapping[str, Any]) -> Any:
        parts = source.split(".")
        if parts[0] not in context:
            raise ValidationError(f"Workflow input source {source!r} is unavailable")
        value = context[parts[0]]
        try:
            for part in parts[1:]:
                value = value[part] if isinstance(value, Mapping) else getattr(value, part)
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValidationError(
                f"Workflow input source {source!r} cannot be resolved: {exc}"
            ) from exc
        return value

    def run(self, **inputs: Any) -> WorkflowResult:
        context: dict[str, Any] = {"input": inputs}
        order = self.execution_order()
        for name in order:
            node = self._nodes[name]
            overlap = set(node.parameters) & set(node.inputs)
            if overlap:
                raise ValidationError(
                    f"Workflow node {name!r} parameters {sorted(overlap)} would override inputs"
                )
            arguments = {
                argument: self._resolve(source, context)
                for argument, source in node.inputs.items()
            }
            arguments.update(node.parameters)
            context[name] = node.operation(**arguments)
        return WorkflowResult(
            outputs={name: context[name] for name in order},
            execution_order=order,
        )

    def to_dict(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        for node in self._nodes.values():
            try:
                json.dumps(dict(node.parameters))
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Workflow node {node.name!r} parameters are not JSON-encodable: {exc}"
                ) from exc
            nodes.append(
                {
                    "name": node.name,
                    "operation": f"{node.operation.__module__}:{node.operation.__qualname__}",
                    "inputs": dict(node.inputs),
                    "parameters": dict(node.parameters),
                    "domain": node.domain,
                }
            )
        return {"name": self.name, "nodes": nodes}
