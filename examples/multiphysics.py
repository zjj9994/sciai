"""Minimal fluid-solid variable exchange through the coupling protocol."""

from sciai.multiphysics import CoupledSimulator, InMemoryAdapter

fluid = InMemoryAdapter("fluid", "fluid", {"pressure": 101325.0})
solid = InMemoryAdapter("solid", "solid", {"displacement": 0.0})

simulation = CoupledSimulator([fluid, solid])
simulation.couple("fluid", "solid", variables=["pressure"])
result = simulation.run(steps=3)
print(result.adapter_states)
