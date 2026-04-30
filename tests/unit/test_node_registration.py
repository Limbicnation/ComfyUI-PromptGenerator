"""Contract tests for ComfyUI node registration.

Catches the v1.1.6-class packaging gap: a new node file added under nodes/ but
forgotten in the top-level __init__.py NODE_CLASS_MAPPINGS dict, which would
silently drop the node from the ComfyUI menu at runtime.
"""

from __future__ import annotations

import importlib.util
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import nodes

EXPECTED_NODES: frozenset[str] = frozenset(
    {
        "Limbicnation_PromptGenerator",
        "Limbicnation_StyleApplier",
        "Limbicnation_PromptRefiner",
        "Limbicnation_NegativePrompt",
        "Limbicnation_PromptCombiner",
    }
)


def _load_top_level_init() -> ModuleType:
    """Load the package's top-level __init__.py directly via importlib.

    `import __init__` is not valid as a regular import; this lets the test
    inspect the canonical NODE_CLASS_MAPPINGS without changing sys.path
    semantics for the rest of the suite.
    """
    init_path = Path(__file__).resolve().parents[2] / "__init__.py"
    spec = importlib.util.spec_from_file_location("comfyui_prompt_generator_root", init_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for {init_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def root() -> ModuleType:
    return _load_top_level_init()


@pytest.fixture(scope="module")
def class_mappings(root: ModuleType) -> dict[str, type]:
    return root.NODE_CLASS_MAPPINGS


@pytest.fixture(scope="module")
def display_mappings(root: ModuleType) -> dict[str, str]:
    return root.NODE_DISPLAY_NAME_MAPPINGS


def test_class_mappings_match_expected(class_mappings: dict[str, type]) -> None:
    assert set(class_mappings) == EXPECTED_NODES


def test_display_names_cover_all_classes(class_mappings: dict[str, type], display_mappings: dict[str, str]) -> None:
    assert set(display_mappings) == set(class_mappings), (
        "NODE_DISPLAY_NAME_MAPPINGS keys must mirror NODE_CLASS_MAPPINGS keys"
    )


@pytest.mark.parametrize("key", sorted(EXPECTED_NODES))
def test_each_node_has_required_attrs(class_mappings: dict[str, type], key: str) -> None:
    """Every registered class must satisfy the ComfyUI node interface."""
    cls: Any = class_mappings[key]
    assert callable(getattr(cls, "INPUT_TYPES", None)), f"{key} missing INPUT_TYPES classmethod"
    assert hasattr(cls, "RETURN_TYPES"), f"{key} missing RETURN_TYPES"
    assert hasattr(cls, "FUNCTION"), f"{key} missing FUNCTION"
    assert hasattr(cls, "CATEGORY"), f"{key} missing CATEGORY"

    method_name = cls.FUNCTION
    assert callable(getattr(cls, method_name, None)), (
        f"{key} declares FUNCTION='{method_name}' but no such method exists"
    )


def test_every_nodes_module_is_registered(class_mappings: dict[str, type]) -> None:
    """Every nodes/*_node.py on disk must be wired into NODE_CLASS_MAPPINGS.

    This is the regression guard for the v1.1.6 issue where chain nodes were
    added to the nodes/ directory but never registered, so they didn't ship.
    """
    discovered_modules: set[str] = {name for _, name, ispkg in pkgutil.iter_modules(nodes.__path__) if not ispkg}
    discovered_node_modules = {n for n in discovered_modules if n.endswith("_node")}

    registered_modules = {cls.__module__.rsplit(".", 1)[-1] for cls in class_mappings.values()}

    missing = discovered_node_modules - registered_modules
    assert not missing, f"Node modules present on disk but not registered in NODE_CLASS_MAPPINGS: {sorted(missing)}"
