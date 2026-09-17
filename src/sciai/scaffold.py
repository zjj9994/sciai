"""Community domain-plugin project scaffolding."""

from __future__ import annotations

import re
from pathlib import Path

from sciai.exceptions import ValidationError

_NAME = re.compile(r"^[a-z][a-z0-9-]*$")

_PYPROJECT = '''[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sciai-{distribution}"
version = "0.1.0"
description = "{display} domain plugin for sciai"
requires-python = ">=3.10"
dependencies = ["sciai-core>=0.1.0a1"]

[project.entry-points."sciai.domains"]
{entry_key} = "sciai_{module}:plugin"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools]
package-dir = {{"" = "src"}}
'''

_INIT = '''"""{display} domain plugin for sciai."""

from sciai.domains import DomainPlugin

plugin = DomainPlugin(
    name="{entry_key}",
    version="0.1.0",
    description="{display} domain extension.",
    subdomains=(),
    data_types=(),
    models={{}},
    datasets={{}},
    operators={{}},
    pipelines={{}},
    simulators={{}},
)

__all__ = ["plugin"]
'''

_TEST = '''from sciai_{module} import plugin


def test_plugin_identity() -> None:
    assert plugin.name == "{entry_key}"
    assert plugin.version == "0.1.0"
'''


def scaffold_domain(name: str, destination: str | Path = ".") -> Path:
    """Create a minimal separately distributable domain plugin."""
    normalized = name.strip().lower().replace("_", "-")
    if not _NAME.fullmatch(normalized):
        raise ValidationError(
            "Domain names must start with a letter and contain lowercase letters, "
            "digits, or hyphens"
        )
    root = Path(destination).expanduser().resolve() / f"sciai-{normalized}"
    if root.exists():
        raise FileExistsError(root)
    module = normalized.replace("-", "_")
    display = normalized.replace("-", " ").title()
    package = root / "src" / f"sciai_{module}"
    tests = root / "tests"
    package.mkdir(parents=True)
    tests.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        _PYPROJECT.format(
            distribution=normalized,
            display=display,
            entry_key=normalized,
            module=module,
        ),
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        _INIT.format(display=display, entry_key=normalized),
        encoding="utf-8",
    )
    (tests / "test_plugin.py").write_text(
        _TEST.format(module=module, entry_key=normalized),
        encoding="utf-8",
    )
    return root
