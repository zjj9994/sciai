"""Scientific domain plugins and capability catalog."""

import threading

from sciai.domains.base import DomainPlugin, discover_plugins
from sciai.domains.catalog import (
    DOMAIN_CATALOG,
    CapabilityStatus,
    DomainFamily,
    get_domain_family,
)

_BUILTINS_LOADED = False
_LOAD_LOCK = threading.RLock()


def load_builtin_plugins() -> None:
    global _BUILTINS_LOADED
    with _LOAD_LOCK:
        if _BUILTINS_LOADED:
            return
        from sciai.domains.fluid import plugin as fluid_plugin
        from sciai.domains.molecule import plugin as molecule_plugin

        molecule_plugin.register()
        fluid_plugin.register()
        _BUILTINS_LOADED = True


__all__ = [
    "DOMAIN_CATALOG",
    "CapabilityStatus",
    "DomainFamily",
    "DomainPlugin",
    "discover_plugins",
    "get_domain_family",
    "load_builtin_plugins",
]
