"""Strict, immutable base model for every contract in the engine."""

import hashlib
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Frozen, extra-forbidding, strictly typed model.

    * ``strict=True``: no silent coercion; custom primitives parse canonical text explicitly.
    * ``frozen=True``: state changes produce new, fully re-validated instances (:meth:`evolve`).
    * ``hide_input_in_errors=True``: validation errors never echo financial values or secrets.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
        use_enum_values=False,
        revalidate_instances="never",
    )

    def evolve(self, **changes: Any) -> Self:  # noqa: ANN401 - field values are heterogeneous
        """Return a copy with ``changes`` applied, re-running every validator.

        Unlike ``model_copy(update=...)``, invalid updates raise instead of producing a model
        that violates its own invariants.
        """
        data: dict[str, object] = {name: getattr(self, name) for name in type(self).model_fields}
        unknown = set(changes) - set(data)
        if unknown:
            raise ValueError(f"unknown fields for {type(self).__name__}: {sorted(unknown)}")
        data.update(changes)
        return type(self).model_validate(data)

    def canonical_json(self) -> str:
        """Deterministic JSON (field order fixed, sets sorted, decimals as exact text)."""
        return self.model_dump_json()

    def digest(self) -> str:
        """SHA-256 of :meth:`canonical_json`, for reproducibility checks and PII-free logging."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
