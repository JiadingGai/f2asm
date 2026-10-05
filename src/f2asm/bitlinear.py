"""Exact GF(2) bit-linear instruction encoding.

This module deliberately has no dependency on the existing learned assembler.
It models each output bit as a linear function over explicit binary features:
operand bits, one-hot modifier choices, sentinel flags, and selected pairwise
interactions.  Training and support checks use Python integers as bitsets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


WORD_BITS = 128
WORD_MASK = (1 << WORD_BITS) - 1
FORMAT_VERSION = 1

# Standalone model JSON is an external input.  Bound its object graph before
# constructing FeatureSchema, whose cached feature-name tuple scales with the
# declared widths and field counts.
MAX_MODEL_JSON_CHARS = 64 * 1024 * 1024
_MAX_SERIALIZED_SCHEMA_ITEMS = 4096
_MAX_SERIALIZED_FEATURES = 131072
_MAX_SERIALIZED_SAMPLE_COUNT = (1 << 63) - 1
_MAX_SERIALIZED_NAME_CHARS = 4096


class BitLinearError(ValueError):
    """Base class for bit-linear model errors."""


class FeatureValidationError(BitLinearError):
    """An operand, modifier, or feature schema is invalid."""


class ConflictingSamplesError(BitLinearError):
    """Training observations cannot be represented by one GF(2) model."""

    def __init__(
        self, sample_index: int, observed: int, implied: int, residual: int
    ) -> None:
        self.sample_index = sample_index
        self.observed = observed
        self.implied = implied
        self.residual = residual
        super().__init__(
            "training sample %d conflicts with prior samples: observed "
            "0x%032x, implied 0x%032x (xor 0x%032x)"
            % (sample_index, observed, implied, residual)
        )


class OutOfSpanError(BitLinearError):
    """An input uses a feature combination unsupported by the training span."""

    def __init__(self, residual_features: Sequence[str]) -> None:
        self.residual_features = tuple(residual_features)
        detail = ", ".join(self.residual_features) or "<unknown>"
        super().__init__(
            "feature vector is outside the training row span; unresolved "
            "features: " + detail
        )


@dataclass(frozen=True)
class BitLinearExtensionReport:
    """Deterministic statistics for one incremental model update."""

    submitted_samples: int
    independent_rows_added: int
    redundant_samples: int
    rank_before: int
    rank_after: int


@dataclass(frozen=True)
class OperandField:
    """A finite-width operand expanded into little-endian binary features."""

    name: str
    width: int
    signed: bool = False

    def __post_init__(self) -> None:
        _validate_name(self.name, "operand")
        if not isinstance(self.width, int) or isinstance(self.width, bool):
            raise FeatureValidationError("operand width must be an integer")
        if self.width <= 0:
            raise FeatureValidationError(
                "operand %r width must be positive" % self.name
            )
        if not isinstance(self.signed, bool):
            raise FeatureValidationError("operand signed must be boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "width": self.width, "signed": self.signed}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperandField":
        _require_keys(data, {"name", "width", "signed"}, "operand")
        return cls(
            name=_require_type(data["name"], str, "operand name"),
            width=_require_int(data["width"], "operand width"),
            signed=_require_bool(data["signed"], "operand signed"),
        )


@dataclass(frozen=True)
class ModifierEnum:
    """A named modifier group represented by one feature per legal choice."""

    name: str
    choices: Tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        _validate_name(self.name, "modifier")
        choices = tuple(self.choices)
        object.__setattr__(self, "choices", choices)
        if not choices:
            raise FeatureValidationError(
                "modifier %r must define at least one choice" % self.name
            )
        if any(
            not isinstance(choice, str) or not choice or ":" in choice
            for choice in choices
        ):
            raise FeatureValidationError(
                "modifier %r choices must be nonempty strings without ':'"
                % self.name
            )
        if len(set(choices)) != len(choices):
            raise FeatureValidationError(
                "modifier %r contains duplicate choices" % self.name
            )
        if not isinstance(self.required, bool):
            raise FeatureValidationError("modifier required must be boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "choices": list(self.choices),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModifierEnum":
        _require_keys(data, {"name", "choices", "required"}, "modifier")
        choices = data["choices"]
        if not isinstance(choices, list):
            raise FeatureValidationError("modifier choices must be a list")
        return cls(
            name=_require_type(data["name"], str, "modifier name"),
            choices=tuple(
                _require_type(choice, str, "modifier choice")
                for choice in choices
            ),
            required=_require_bool(data["required"], "modifier required"),
        )


@dataclass(frozen=True)
class SentinelFlag:
    """A binary feature indicating that an operand equals a special value."""

    name: str
    operand: str
    value: int

    def __post_init__(self) -> None:
        _validate_name(self.name, "sentinel")
        _validate_name(self.operand, "sentinel operand")
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            raise FeatureValidationError("sentinel value must be an integer")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "operand": self.operand, "value": self.value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SentinelFlag":
        _require_keys(data, {"name", "operand", "value"}, "sentinel")
        return cls(
            name=_require_type(data["name"], str, "sentinel name"),
            operand=_require_type(data["operand"], str, "sentinel operand"),
            value=_require_int(data["value"], "sentinel value"),
        )


@dataclass(frozen=True)
class Interaction:
    """A selected pairwise product of two already-defined binary features."""

    name: str
    left: str
    right: str

    def __post_init__(self) -> None:
        _validate_name(self.name, "interaction")
        if not isinstance(self.left, str) or not self.left:
            raise FeatureValidationError("interaction left must be a feature name")
        if not isinstance(self.right, str) or not self.right:
            raise FeatureValidationError("interaction right must be a feature name")
        if self.left == self.right:
            raise FeatureValidationError(
                "interaction %r must reference two distinct features" % self.name
            )

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "left": self.left, "right": self.right}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Interaction":
        _require_keys(data, {"name", "left", "right"}, "interaction")
        return cls(
            name=_require_type(data["name"], str, "interaction name"),
            left=_require_type(data["left"], str, "interaction left"),
            right=_require_type(data["right"], str, "interaction right"),
        )


def operand_bit(operand: str, bit: int) -> str:
    """Return the canonical feature name for one operand bit."""

    _validate_name(operand, "operand")
    if not isinstance(bit, int) or isinstance(bit, bool) or bit < 0:
        raise FeatureValidationError("operand bit index must be nonnegative")
    return "operand:%s:%d" % (operand, bit)


def modifier_choice(modifier: str, choice: str) -> str:
    """Return the canonical feature name for a modifier choice."""

    _validate_name(modifier, "modifier")
    if not isinstance(choice, str) or not choice or ":" in choice:
        raise FeatureValidationError(
            "modifier choice must be a nonempty string without ':'"
        )
    return "modifier:%s:%s" % (modifier, choice)


def sentinel_flag(name: str) -> str:
    """Return the canonical feature name for a sentinel flag."""

    _validate_name(name, "sentinel")
    return "sentinel:%s" % name


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered definition of the binary feature vector ``phi``."""

    operands: Tuple[OperandField, ...] = ()
    modifiers: Tuple[ModifierEnum, ...] = ()
    sentinels: Tuple[SentinelFlag, ...] = ()
    interactions: Tuple[Interaction, ...] = ()
    _feature_names_cache: Tuple[str, ...] = field(
        init=False, repr=False, compare=False
    )
    _operand_offsets: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _modifier_offsets: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _sentinel_offsets: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _interaction_inputs: Tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )
    _interaction_lookup: Mapping[Tuple[int, int], Tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "modifiers", tuple(self.modifiers))
        object.__setattr__(self, "sentinels", tuple(self.sentinels))
        object.__setattr__(self, "interactions", tuple(self.interactions))
        _require_instances(self.operands, OperandField, "operands")
        _require_instances(self.modifiers, ModifierEnum, "modifiers")
        _require_instances(self.sentinels, SentinelFlag, "sentinels")
        _require_instances(self.interactions, Interaction, "interactions")

        operand_names = [field.name for field in self.operands]
        modifier_names = [field.name for field in self.modifiers]
        sentinel_names = [flag.name for flag in self.sentinels]
        interaction_names = [item.name for item in self.interactions]
        _reject_duplicates(operand_names, "operand")
        _reject_duplicates(modifier_names, "modifier")
        _reject_duplicates(sentinel_names, "sentinel")
        _reject_duplicates(interaction_names, "interaction")

        operand_map = {field.name: field for field in self.operands}
        for flag in self.sentinels:
            if flag.operand not in operand_map:
                raise FeatureValidationError(
                    "sentinel %r references unknown operand %r"
                    % (flag.name, flag.operand)
                )
            _validate_operand_value(operand_map[flag.operand], flag.value)

        base_names = self._base_feature_names()
        if len(set(base_names)) != len(base_names):
            raise FeatureValidationError("generated base feature names collide")
        base_set = set(base_names)
        for item in self.interactions:
            if item.left not in base_set:
                raise FeatureValidationError(
                    "interaction %r references unknown feature %r"
                    % (item.name, item.left)
                )
            if item.right not in base_set:
                raise FeatureValidationError(
                    "interaction %r references unknown feature %r"
                    % (item.name, item.right)
                )

        feature_names = base_names + tuple(
            "interaction:%s" % item.name for item in self.interactions
        )
        if len(set(feature_names)) != len(feature_names):
            raise FeatureValidationError("generated feature names collide")
        object.__setattr__(self, "_feature_names_cache", feature_names)

        index = {name: bit for bit, name in enumerate(feature_names)}
        cursor = 1
        operand_offsets = []
        for operand in self.operands:
            operand_offsets.append(cursor)
            cursor += operand.width
        modifier_offsets = []
        for modifier in self.modifiers:
            modifier_offsets.append(cursor)
            cursor += len(modifier.choices)
        sentinel_offsets = tuple(
            range(cursor, cursor + len(self.sentinels))
        )
        object.__setattr__(
            self, "_operand_offsets", tuple(operand_offsets)
        )
        object.__setattr__(
            self, "_modifier_offsets", tuple(modifier_offsets)
        )
        object.__setattr__(self, "_sentinel_offsets", sentinel_offsets)

        interaction_lookup: Dict[Tuple[int, int], List[int]] = {}
        interaction_base = len(base_names)
        for interaction_index, item in enumerate(self.interactions):
            endpoints = tuple(sorted((index[item.left], index[item.right])))
            interaction_lookup.setdefault(endpoints, []).append(
                interaction_base + interaction_index
            )
        object.__setattr__(
            self,
            "_interaction_inputs",
            tuple(
                sorted(
                    {
                        endpoint
                        for endpoints in interaction_lookup
                        for endpoint in endpoints
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "_interaction_lookup",
            {
                endpoints: tuple(outputs)
                for endpoints, outputs in interaction_lookup.items()
            },
        )

    def _base_feature_names(self) -> Tuple[str, ...]:
        names: List[str] = ["constant"]
        for field in self.operands:
            names.extend(operand_bit(field.name, bit) for bit in range(field.width))
        for modifier in self.modifiers:
            names.extend(
                modifier_choice(modifier.name, choice)
                for choice in modifier.choices
            )
        names.extend(sentinel_flag(flag.name) for flag in self.sentinels)
        return tuple(names)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self._feature_names_cache

    @property
    def size(self) -> int:
        return len(self.feature_names)

    def vector(
        self,
        operands: Mapping[str, int],
        modifiers: Optional[Mapping[str, Optional[str]]] = None,
    ) -> int:
        """Build a feature vector represented as a Python integer bitset."""

        if not isinstance(operands, Mapping):
            raise FeatureValidationError("operands must be a mapping")
        if modifiers is None:
            modifiers = {}
        if not isinstance(modifiers, Mapping):
            raise FeatureValidationError("modifiers must be a mapping")

        expected_operands = {field.name for field in self.operands}
        supplied_operands = set(operands)
        missing_operands = expected_operands - supplied_operands
        extra_operands = supplied_operands - expected_operands
        if missing_operands:
            raise FeatureValidationError(
                "missing operands: " + ", ".join(sorted(missing_operands))
            )
        if extra_operands:
            raise FeatureValidationError(
                "unknown operands: " + ", ".join(sorted(extra_operands))
            )

        expected_modifiers = {field.name for field in self.modifiers}
        extra_modifiers = set(modifiers) - expected_modifiers
        if extra_modifiers:
            raise FeatureValidationError(
                "unknown modifier groups: " + ", ".join(sorted(extra_modifiers))
            )

        vector = 1  # constant is always feature zero

        for field, offset in zip(self.operands, self._operand_offsets):
            value = operands[field.name]
            encoded = _validate_operand_value(field, value)
            vector |= encoded << offset

        for modifier, offset in zip(
            self.modifiers, self._modifier_offsets
        ):
            choice = modifiers.get(modifier.name)
            if choice is None:
                if modifier.required:
                    raise FeatureValidationError(
                        "missing required modifier group %r" % modifier.name
                    )
            elif choice not in modifier.choices:
                raise FeatureValidationError(
                    "unknown %s modifier choice %r; expected one of %s"
                    % (
                        modifier.name,
                        choice,
                        ", ".join(repr(item) for item in modifier.choices),
                    )
                )
            if choice is not None:
                vector |= 1 << (
                    offset + modifier.choices.index(choice)
                )

        for flag, offset in zip(
            self.sentinels, self._sentinel_offsets
        ):
            if operands[flag.operand] == flag.value:
                vector |= 1 << offset

        active_inputs = [
            feature
            for feature in self._interaction_inputs
            if (vector >> feature) & 1
        ]
        for left_index, left in enumerate(active_inputs):
            for right in active_inputs[left_index + 1 :]:
                for output in self._interaction_lookup.get(
                    (left, right), ()
                ):
                    vector |= 1 << output

        return vector

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operands": [field.to_dict() for field in self.operands],
            "modifiers": [field.to_dict() for field in self.modifiers],
            "sentinels": [flag.to_dict() for flag in self.sentinels],
            "interactions": [item.to_dict() for item in self.interactions],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureSchema":
        _require_keys(
            data,
            {"operands", "modifiers", "sentinels", "interactions"},
            "schema",
        )
        for key in ("operands", "modifiers", "sentinels", "interactions"):
            if not isinstance(data[key], list):
                raise FeatureValidationError("schema %s must be a list" % key)
        return cls(
            operands=tuple(OperandField.from_dict(item) for item in data["operands"]),
            modifiers=tuple(
                ModifierEnum.from_dict(item) for item in data["modifiers"]
            ),
            sentinels=tuple(
                SentinelFlag.from_dict(item) for item in data["sentinels"]
            ),
            interactions=tuple(
                Interaction.from_dict(item) for item in data["interactions"]
            ),
        )


@dataclass(frozen=True)
class TrainingSample:
    operands: Mapping[str, int]
    word: int
    modifiers: Optional[Mapping[str, Optional[str]]] = None


class BitLinearModel:
    """A trained 128-bit instruction encoder over GF(2)."""

    def __init__(
        self,
        schema: FeatureSchema,
        basis: Mapping[int, Tuple[int, int]],
        sample_count: int,
    ) -> None:
        self.schema = schema
        self._basis = dict(basis)
        self.sample_count = sample_count
        self._validate_basis()
        # These derived values are deliberately absent from the wire format.
        # Encoding is a hot path, so do not repeatedly sort the immutable
        # echelon pivots or rescan every output row.
        self._sorted_pivots = tuple(sorted(self._basis, reverse=True))
        output_mask = 0
        for _, word in self._basis.values():
            output_mask |= word
        self._basis_output_mask = output_mask

    @classmethod
    def train(
        cls, schema: FeatureSchema, samples: Iterable[TrainingSample]
    ) -> "BitLinearModel":
        """Fit ``Y = Phi W`` exactly using bitset Gaussian elimination."""

        if not isinstance(schema, FeatureSchema):
            raise FeatureValidationError("schema must be a FeatureSchema")
        basis: Dict[int, Tuple[int, int]] = {}
        sample_count = 0
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, TrainingSample):
                raise FeatureValidationError(
                    "training sample %d is not a TrainingSample" % sample_index
                )
            word = _validate_word(sample.word)
            features = schema.vector(sample.operands, sample.modifiers)
            reduced_features, reduced_word = _reduce(
                features, word, basis
            )
            if reduced_features == 0:
                if reduced_word:
                    implied = word ^ reduced_word
                    raise ConflictingSamplesError(
                        sample_index, word, implied, reduced_word
                    )
            else:
                pivot = reduced_features.bit_length() - 1
                # Convert the maintained echelon basis into reduced row-echelon
                # form.  Pivot columns then form an identity matrix and each
                # pivot row's output is a canonical coefficient in W.
                for old_pivot, (old_features, old_word) in list(basis.items()):
                    if (old_features >> pivot) & 1:
                        basis[old_pivot] = (
                            old_features ^ reduced_features,
                            old_word ^ reduced_word,
                        )
                basis[pivot] = (reduced_features, reduced_word)
            sample_count += 1
        return cls(schema, basis, sample_count)

    def extend(
        self, samples: Iterable[TrainingSample]
    ) -> Tuple["BitLinearModel", BitLinearExtensionReport]:
        """Return a copy extended by consistent GF(2) equations.

        Existing state is never mutated. A dependent equation with a nonzero
        output residual raises ``ConflictingSamplesError`` before any candidate
        model is returned. Dependent consistent equations are retained in the
        sample count but do not change the basis.
        """

        basis = dict(self._basis)
        submitted = 0
        independent = 0
        redundant = 0
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, TrainingSample):
                raise FeatureValidationError(
                    "extension sample %d is not a TrainingSample"
                    % sample_index
                )
            word = _validate_word(sample.word)
            features = self.schema.vector(
                sample.operands, sample.modifiers
            )
            reduced_features, reduced_word = _reduce(
                features, word, basis
            )
            if reduced_features == 0:
                if reduced_word:
                    implied = word ^ reduced_word
                    raise ConflictingSamplesError(
                        sample_index, word, implied, reduced_word
                    )
                redundant += 1
            else:
                pivot = reduced_features.bit_length() - 1
                for old_pivot, (
                    old_features,
                    old_word,
                ) in list(basis.items()):
                    if (old_features >> pivot) & 1:
                        basis[old_pivot] = (
                            old_features ^ reduced_features,
                            old_word ^ reduced_word,
                        )
                basis[pivot] = (reduced_features, reduced_word)
                independent += 1
            submitted += 1

        candidate = type(self)(
            self.schema,
            basis,
            self.sample_count + submitted,
        )
        return candidate, BitLinearExtensionReport(
            submitted_samples=submitted,
            independent_rows_added=independent,
            redundant_samples=redundant,
            rank_before=self.rank,
            rank_after=candidate.rank,
        )

    @classmethod
    def merge(
        cls, models: Iterable["BitLinearModel"]
    ) -> "BitLinearModel":
        """Merge compatible trained row spaces into one exact model.

        A model basis contains equations in the same feature/output space as
        its original samples. Gaussian-eliminating the union of those basis
        rows is therefore equivalent to retraining on the union of the source
        observations, while preserving compact evidence repositories.
        """

        rows = tuple(models)
        if not rows:
            raise FeatureValidationError(
                "cannot merge an empty model collection"
            )
        if any(not isinstance(model, BitLinearModel) for model in rows):
            raise FeatureValidationError(
                "merged models must be BitLinearModel values"
            )
        schema = rows[0].schema
        if any(model.schema != schema for model in rows[1:]):
            raise FeatureValidationError(
                "cannot merge models with different schemas"
            )

        basis: Dict[int, Tuple[int, int]] = {}
        equation_index = 0
        for model in rows:
            for pivot in model._sorted_pivots:
                features, word = model._basis[pivot]
                reduced_features, reduced_word = _reduce(
                    features, word, basis
                )
                if reduced_features == 0:
                    if reduced_word:
                        implied = word ^ reduced_word
                        raise ConflictingSamplesError(
                            equation_index,
                            word,
                            implied,
                            reduced_word,
                        )
                else:
                    new_pivot = reduced_features.bit_length() - 1
                    for old_pivot, (
                        old_features,
                        old_word,
                    ) in list(basis.items()):
                        if (old_features >> new_pivot) & 1:
                            basis[old_pivot] = (
                                old_features ^ reduced_features,
                                old_word ^ reduced_word,
                            )
                    basis[new_pivot] = (
                        reduced_features,
                        reduced_word,
                    )
                equation_index += 1
        return cls(
            schema,
            basis,
            sum(model.sample_count for model in rows),
        )

    @property
    def rank(self) -> int:
        return len(self._basis)

    @property
    def nullity(self) -> int:
        return self.schema.size - self.rank

    @property
    def basis_output_mask(self) -> int:
        """Return every output bit that any basis row can produce."""

        return self._basis_output_mask

    @property
    def nullspace_constraints(self) -> Tuple[int, ...]:
        """Return a basis for ``ker(Phi)`` as feature-sized bitsets.

        A candidate feature vector is supported exactly when its GF(2) dot
        product with every returned constraint is zero.
        """

        pivots = set(self._basis)
        constraints: List[int] = []
        for free_feature in range(self.schema.size):
            if free_feature in pivots:
                continue
            constraint = 1 << free_feature
            for pivot, (row_features, _) in self._basis.items():
                if (row_features >> free_feature) & 1:
                    constraint |= 1 << pivot
            constraints.append(constraint)
        return tuple(constraints)

    @property
    def weights(self) -> Tuple[int, ...]:
        """Return one canonical 128-bit coefficient per feature.

        Non-pivot coefficients are zero.  Because the stored basis is in
        reduced row-echelon form, this is one exact solution of ``Phi W = Y``.
        """

        weights = [0] * self.schema.size
        for pivot, (_, word) in self._basis.items():
            weights[pivot] = word
        return tuple(weights)

    def supports(
        self,
        operands: Mapping[str, int],
        modifiers: Optional[Mapping[str, Optional[str]]] = None,
    ) -> bool:
        features = self.schema.vector(operands, modifiers)
        residual, _ = _reduce(
            features, 0, self._basis, self._sorted_pivots
        )
        return residual == 0

    def encode(
        self,
        operands: Mapping[str, int],
        modifiers: Optional[Mapping[str, Optional[str]]] = None,
    ) -> int:
        features = self.schema.vector(operands, modifiers)
        residual, word = _reduce(
            features, 0, self._basis, self._sorted_pivots
        )
        if residual:
            names = self.schema.feature_names
            unresolved = [
                names[bit]
                for bit in range(len(names))
                if (residual >> bit) & 1
            ]
            raise OutOfSpanError(unresolved)
        return word

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "hopperasm.bitlinear",
            "version": FORMAT_VERSION,
            "word_bits": WORD_BITS,
            "schema": self.schema.to_dict(),
            "sample_count": self.sample_count,
            "basis": [
                {
                    "pivot": pivot,
                    "features": _format_hex(features, self.schema.size),
                    "word": "0x%032x" % word,
                }
                for pivot in self._sorted_pivots
                for features, word in (self._basis[pivot],)
            ],
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BitLinearModel":
        _require_keys(
            data,
            {
                "format",
                "version",
                "word_bits",
                "schema",
                "sample_count",
                "basis",
            },
            "model",
        )
        if data["format"] != "hopperasm.bitlinear":
            raise FeatureValidationError("unknown bit-linear model format")
        if data["version"] != FORMAT_VERSION:
            raise FeatureValidationError(
                "unsupported bit-linear model version %r" % data["version"]
            )
        if data["word_bits"] != WORD_BITS:
            raise FeatureValidationError(
                "model word_bits must be %d" % WORD_BITS
            )
        _preflight_serialized_model(data)
        schema_data = data["schema"]
        if not isinstance(schema_data, Mapping):
            raise FeatureValidationError("model schema must be an object")
        schema = FeatureSchema.from_dict(schema_data)
        sample_count = _require_int(data["sample_count"], "sample_count")
        if sample_count < 0:
            raise FeatureValidationError("sample_count must be nonnegative")
        rows = data["basis"]
        if not isinstance(rows, list):
            raise FeatureValidationError("model basis must be a list")
        basis: Dict[int, Tuple[int, int]] = {}
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise FeatureValidationError(
                    "basis row %d must be an object" % row_index
                )
            _require_keys(row, {"pivot", "features", "word"}, "basis row")
            pivot = _require_int(row["pivot"], "basis pivot")
            if pivot in basis:
                raise FeatureValidationError(
                    "duplicate basis pivot %d" % pivot
                )
            features = _parse_hex(row["features"], "basis features")
            word = _parse_hex(row["word"], "basis word")
            basis[pivot] = (features, word)
        return cls(schema, basis, sample_count)

    @classmethod
    def from_json(cls, text: str) -> "BitLinearModel":
        if not isinstance(text, str):
            raise FeatureValidationError(
                "bit-linear model JSON must be a string"
            )
        if len(text) > MAX_MODEL_JSON_CHARS:
            raise FeatureValidationError(
                "bit-linear model JSON exceeds the supported size"
            )
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise FeatureValidationError(
                "invalid bit-linear model JSON: %s" % exc
            ) from exc
        if not isinstance(data, Mapping):
            raise FeatureValidationError("bit-linear model JSON must be an object")
        return cls.from_dict(data)

    def _validate_basis(self) -> None:
        if not isinstance(self.schema, FeatureSchema):
            raise FeatureValidationError("schema must be a FeatureSchema")
        if not isinstance(self.sample_count, int) or isinstance(
            self.sample_count, bool
        ):
            raise FeatureValidationError("sample_count must be an integer")
        if self.sample_count < 0:
            raise FeatureValidationError("sample_count must be nonnegative")
        if len(self._basis) > self.schema.size:
            raise FeatureValidationError("basis rank exceeds feature count")
        if self.sample_count < len(self._basis):
            raise FeatureValidationError("sample_count is smaller than basis rank")
        pivots = set(self._basis)
        for pivot, row in self._basis.items():
            if not isinstance(pivot, int) or isinstance(pivot, bool):
                raise FeatureValidationError("basis pivot must be an integer")
            if pivot < 0 or pivot >= self.schema.size:
                raise FeatureValidationError("basis pivot is out of range")
            if not isinstance(row, tuple) or len(row) != 2:
                raise FeatureValidationError("basis row must be a pair")
            features, word = row
            if not isinstance(features, int) or isinstance(features, bool):
                raise FeatureValidationError("basis features must be an integer")
            if features <= 0 or features.bit_length() > self.schema.size:
                raise FeatureValidationError("basis feature vector is out of range")
            if features.bit_length() - 1 != pivot:
                raise FeatureValidationError(
                    "basis pivot does not match its feature vector"
                )
            _validate_word(word)
            for other_pivot in pivots:
                if other_pivot != pivot and ((features >> other_pivot) & 1):
                    raise FeatureValidationError(
                        "basis is not in reduced row-echelon form"
                    )


def _reduce(
    features: int,
    word: int,
    basis: Mapping[int, Tuple[int, int]],
    pivots: Optional[Sequence[int]] = None,
) -> Tuple[int, int]:
    if pivots is None:
        pivots = sorted(basis, reverse=True)
    for pivot in pivots:
        if (features >> pivot) & 1:
            row_features, row_word = basis[pivot]
            features ^= row_features
            word ^= row_word
    return features, word


def _validate_operand_value(field: OperandField, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FeatureValidationError(
            "operand %r must be an integer" % field.name
        )
    if field.signed:
        minimum = -(1 << (field.width - 1))
        maximum = (1 << (field.width - 1)) - 1
    else:
        minimum = 0
        maximum = (1 << field.width) - 1
    if value < minimum or value > maximum:
        kind = "signed" if field.signed else "unsigned"
        raise FeatureValidationError(
            "operand %r value %d does not fit %s %d-bit range [%d, %d]"
            % (field.name, value, kind, field.width, minimum, maximum)
        )
    return value & ((1 << field.width) - 1)


def _validate_word(word: Any) -> int:
    if not isinstance(word, int) or isinstance(word, bool):
        raise FeatureValidationError("instruction word must be an integer")
    if word < 0 or word > WORD_MASK:
        raise FeatureValidationError(
            "instruction word must fit unsigned %d-bit range" % WORD_BITS
        )
    return word


def _preflight_serialized_model(data: Mapping[str, Any]) -> None:
    """Bound a standalone model before allocating schema-derived state."""

    schema = data.get("schema")
    if not isinstance(schema, Mapping):
        raise FeatureValidationError("model schema must be an object")
    _require_keys(
        schema,
        {"operands", "modifiers", "sentinels", "interactions"},
        "schema",
    )

    collections: Dict[str, List[Any]] = {}
    for name in ("operands", "modifiers", "sentinels", "interactions"):
        values = schema[name]
        if not isinstance(values, list):
            raise FeatureValidationError("schema %s must be a list" % name)
        if len(values) > _MAX_SERIALIZED_SCHEMA_ITEMS:
            raise FeatureValidationError(
                "schema %s exceeds the supported bound" % name
            )
        collections[name] = values

    feature_count = 1
    for index, operand in enumerate(collections["operands"]):
        if not isinstance(operand, Mapping):
            raise FeatureValidationError(
                "schema operand %d must be an object" % index
            )
        width = operand.get("width")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            or width > _MAX_SERIALIZED_FEATURES
        ):
            raise FeatureValidationError(
                "schema operand width exceeds the supported bound"
            )
        _preflight_serialized_name(
            operand.get("name"), "schema operand name"
        )
        feature_count += width
        if feature_count > _MAX_SERIALIZED_FEATURES:
            raise FeatureValidationError(
                "schema exceeds the supported feature bound"
            )

    for index, modifier in enumerate(collections["modifiers"]):
        if not isinstance(modifier, Mapping):
            raise FeatureValidationError(
                "schema modifier %d must be an object" % index
            )
        _preflight_serialized_name(
            modifier.get("name"), "schema modifier name"
        )
        choices = modifier.get("choices")
        if not isinstance(choices, list):
            raise FeatureValidationError(
                "schema modifier choices must be a list"
            )
        if len(choices) > _MAX_SERIALIZED_FEATURES:
            raise FeatureValidationError(
                "schema modifier choices exceed the supported bound"
            )
        for choice in choices:
            _preflight_serialized_name(
                choice, "schema modifier choice"
            )
        feature_count += len(choices)
        if feature_count > _MAX_SERIALIZED_FEATURES:
            raise FeatureValidationError(
                "schema exceeds the supported feature bound"
            )

    for index, sentinel in enumerate(collections["sentinels"]):
        if not isinstance(sentinel, Mapping):
            raise FeatureValidationError(
                "schema sentinel %d must be an object" % index
            )
        _preflight_serialized_name(
            sentinel.get("name"), "schema sentinel name"
        )
        _preflight_serialized_name(
            sentinel.get("operand"), "schema sentinel operand"
        )
    feature_count += len(collections["sentinels"])

    for index, interaction in enumerate(collections["interactions"]):
        if not isinstance(interaction, Mapping):
            raise FeatureValidationError(
                "schema interaction %d must be an object" % index
            )
        for field in ("name", "left", "right"):
            _preflight_serialized_name(
                interaction.get(field),
                "schema interaction %s" % field,
            )
    feature_count += len(collections["interactions"])
    if feature_count > _MAX_SERIALIZED_FEATURES:
        raise FeatureValidationError(
            "schema exceeds the supported feature bound"
        )

    sample_count = data.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 0
        or sample_count > _MAX_SERIALIZED_SAMPLE_COUNT
    ):
        raise FeatureValidationError(
            "model sample_count is outside the supported bound"
        )

    basis = data.get("basis")
    if not isinstance(basis, list):
        raise FeatureValidationError("model basis must be a list")
    if len(basis) > feature_count:
        raise FeatureValidationError(
            "model basis exceeds the supported bound"
        )
    maximum_feature_hex_length = 2 + max(
        1, (feature_count + 3) // 4
    )
    for index, row in enumerate(basis):
        if not isinstance(row, Mapping):
            raise FeatureValidationError(
                "basis row %d must be an object" % index
            )
        features = row.get("features")
        word = row.get("word")
        if (
            not isinstance(features, str)
            or len(features) > maximum_feature_hex_length
        ):
            raise FeatureValidationError(
                "basis features exceed the schema width"
            )
        if not isinstance(word, str) or len(word) > 34:
            raise FeatureValidationError(
                "basis word must fit unsigned 128-bit range"
            )


def _preflight_serialized_name(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_SERIALIZED_NAME_CHARS
    ):
        raise FeatureValidationError(
            "%s exceeds the supported bound" % label
        )


def _format_hex(value: int, bits: int) -> str:
    digits = max(1, (bits + 3) // 4)
    return "0x%0*x" % (digits, value)


def _parse_hex(value: Any, label: str) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise FeatureValidationError("%s must be a 0x-prefixed string" % label)
    try:
        parsed = int(value, 16)
    except ValueError as exc:
        raise FeatureValidationError("%s is not valid hexadecimal" % label) from exc
    if parsed < 0:
        raise FeatureValidationError("%s must be nonnegative" % label)
    return parsed


def _validate_name(name: Any, label: str) -> None:
    if not isinstance(name, str) or not name:
        raise FeatureValidationError("%s name must be a nonempty string" % label)
    if ":" in name:
        raise FeatureValidationError("%s name must not contain ':'" % label)


def _reject_duplicates(names: Sequence[str], label: str) -> None:
    if len(set(names)) != len(names):
        raise FeatureValidationError("duplicate %s names" % label)


def _require_keys(
    data: Mapping[str, Any], expected: set, label: str
) -> None:
    if not isinstance(data, Mapping):
        raise FeatureValidationError("%s must be an object" % label)
    actual = set(data)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise FeatureValidationError("%s fields: %s" % (label, "; ".join(details)))


def _require_type(value: Any, expected: type, label: str) -> Any:
    if not isinstance(value, expected):
        raise FeatureValidationError("%s must be %s" % (label, expected.__name__))
    return value


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FeatureValidationError("%s must be an integer" % label)
    return value


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise FeatureValidationError("%s must be boolean" % label)
    return value


def _require_instances(values: Sequence[Any], expected: type, label: str) -> None:
    if any(not isinstance(value, expected) for value in values):
        raise FeatureValidationError(
            "%s must contain only %s values" % (label, expected.__name__)
        )


__all__ = [
    "BitLinearError",
    "BitLinearExtensionReport",
    "BitLinearModel",
    "ConflictingSamplesError",
    "FeatureSchema",
    "FeatureValidationError",
    "Interaction",
    "ModifierEnum",
    "OperandField",
    "OutOfSpanError",
    "SentinelFlag",
    "TrainingSample",
    "modifier_choice",
    "operand_bit",
    "sentinel_flag",
]
