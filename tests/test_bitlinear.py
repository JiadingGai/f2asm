from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f2asm.bitlinear import (
    BitLinearModel,
    ConflictingSamplesError,
    FeatureSchema,
    FeatureValidationError,
    Interaction,
    ModifierEnum,
    OperandField,
    OutOfSpanError,
    SentinelFlag,
    TrainingSample,
    modifier_choice,
    sentinel_flag,
)


def _linear_word(base, fields, operands):
    word = base
    for name, output_bits in fields.items():
        value = operands[name] & ((1 << len(output_bits)) - 1)
        for source_bit, output_bit in enumerate(output_bits):
            if (value >> source_bit) & 1:
                word ^= 1 << output_bit
    return word


def _apply_weights(features, weights):
    word = 0
    for feature, weight in enumerate(weights):
        if (features >> feature) & 1:
            word ^= weight
    return word


class BitLinearModelTests(unittest.TestCase):
    def test_fixed_and_split_fields_encode_to_128_bits(self):
        schema = FeatureSchema(
            operands=(OperandField("dst", 3), OperandField("src", 3))
        )
        base = (1 << 0) | (1 << 79) | (1 << 127)
        fields = {
            "dst": (8, 40, 95),
            "src": (16, 63, 111),
        }
        samples = [TrainingSample({"dst": 0, "src": 0}, base)]
        for name in fields:
            for bit in range(3):
                operands = {"dst": 0, "src": 0}
                operands[name] = 1 << bit
                samples.append(
                    TrainingSample(
                        operands, _linear_word(base, fields, operands)
                    )
                )

        model = BitLinearModel.train(schema, samples)
        expected = _linear_word(base, fields, {"dst": 5, "src": 6})
        self.assertEqual(model.encode({"dst": 5, "src": 6}), expected)
        self.assertEqual(model.rank, schema.size)
        self.assertEqual(model.nullity, 0)
        self.assertEqual(model.nullspace_constraints, ())
        self.assertEqual(model.encode({"dst": 5, "src": 6}).bit_length(), 128)
        self.assertEqual(
            _apply_weights(
                schema.vector({"dst": 5, "src": 6}), model.weights
            ),
            expected,
        )

    def test_signed_finite_width_operand_uses_twos_complement_bits(self):
        schema = FeatureSchema(operands=(OperandField("imm", 4, signed=True),))
        base = 1 << 3
        output_bits = {"imm": (12, 31, 65, 100)}
        samples = [
            TrainingSample(
                {"imm": value},
                _linear_word(base, output_bits, {"imm": value}),
            )
            for value in (0, 1, 2, 4, -8)
        ]
        model = BitLinearModel.train(schema, samples)

        self.assertEqual(
            model.encode({"imm": -3}),
            _linear_word(base, output_bits, {"imm": -3}),
        )
        with self.assertRaisesRegex(FeatureValidationError, "signed 4-bit range"):
            model.encode({"imm": -9})
        with self.assertRaisesRegex(FeatureValidationError, "signed 4-bit range"):
            model.encode({"imm": 8})

    def test_modifier_enum_is_one_hot_and_serializes_round_trip(self):
        schema = FeatureSchema(
            operands=(OperandField("dst", 1),),
            modifiers=(ModifierEnum("round", ("rn", "rz", "rm")),),
        )
        base = 1 << 7
        modifier_words = {"rn": 1 << 18, "rz": 1 << 55, "rm": 1 << 109}
        samples = []
        for choice, modifier_word in modifier_words.items():
            samples.append(
                TrainingSample(
                    {"dst": 0}, base ^ modifier_word, {"round": choice}
                )
            )
        samples.append(
            TrainingSample(
                {"dst": 1},
                base ^ modifier_words["rn"] ^ (1 << 44),
                {"round": "rn"},
            )
        )
        model = BitLinearModel.train(schema, samples)
        restored = BitLinearModel.from_json(model.to_json())

        self.assertEqual(restored.schema, schema)
        self.assertEqual(restored.rank, model.rank)
        self.assertEqual(restored.weights, model.weights)
        self.assertEqual(
            restored.encode({"dst": 1}, {"round": "rm"}),
            base ^ modifier_words["rm"] ^ (1 << 44),
        )
        with self.assertRaisesRegex(FeatureValidationError, "unknown round"):
            restored.encode({"dst": 0}, {"round": "rp"})

    def test_selected_modifier_sentinel_interaction(self):
        interaction = Interaction(
            "special_zero",
            modifier_choice("mode", "special"),
            sentinel_flag("is_zero"),
        )
        schema = FeatureSchema(
            operands=(OperandField("src", 1),),
            modifiers=(ModifierEnum("mode", ("plain", "special")),),
            sentinels=(SentinelFlag("is_zero", "src", 0),),
            interactions=(interaction,),
        )

        def expected(src, mode):
            word = 1 << 2
            if src:
                word ^= 1 << 20
            if mode == "special":
                word ^= 1 << 50
            if mode == "special" and src == 0:
                word ^= 1 << 117
            return word

        samples = [
            TrainingSample({"src": src}, expected(src, mode), {"mode": mode})
            for src in (0, 1)
            for mode in ("plain", "special")
        ]
        model = BitLinearModel.train(schema, samples)

        for sample in samples:
            self.assertEqual(
                model.encode(sample.operands, sample.modifiers), sample.word
            )

        schema_without_interaction = FeatureSchema(
            operands=schema.operands,
            modifiers=schema.modifiers,
            sentinels=schema.sentinels,
        )
        with self.assertRaisesRegex(ConflictingSamplesError, "conflicts"):
            BitLinearModel.train(schema_without_interaction, samples)

    def test_conflicting_dependent_samples_are_rejected(self):
        schema = FeatureSchema(operands=(OperandField("r", 1),))
        samples = [
            TrainingSample({"r": 0}, 0x10),
            TrainingSample({"r": 1}, 0x20),
            TrainingSample({"r": 0}, 0x11),
        ]
        with self.assertRaises(ConflictingSamplesError) as caught:
            BitLinearModel.train(schema, samples)
        self.assertEqual(caught.exception.sample_index, 2)
        self.assertEqual(caught.exception.observed, 0x11)
        self.assertEqual(caught.exception.implied, 0x10)
        self.assertEqual(caught.exception.residual, 0x01)

    def test_merge_is_equivalent_to_training_on_union_of_row_spaces(self):
        schema = FeatureSchema(operands=(OperandField("r", 2),))

        def word(value):
            return 0x10 ^ (value << 8)

        lower = BitLinearModel.train(
            schema,
            [
                TrainingSample({"r": 0}, word(0)),
                TrainingSample({"r": 1}, word(1)),
            ],
        )
        upper = BitLinearModel.train(
            schema,
            [
                TrainingSample({"r": 2}, word(2)),
                TrainingSample({"r": 3}, word(3)),
            ],
        )

        merged = BitLinearModel.merge((lower, upper))
        direct = BitLinearModel.train(
            schema,
            [
                TrainingSample({"r": value}, word(value))
                for value in range(4)
            ],
        )
        self.assertEqual(merged.to_dict()["basis"], direct.to_dict()["basis"])
        self.assertEqual(merged.sample_count, 4)
        for value in range(4):
            self.assertEqual(merged.encode({"r": value}), word(value))

    def test_merge_rejects_conflicting_source_models(self):
        schema = FeatureSchema(operands=(OperandField("r", 1),))
        left = BitLinearModel.train(
            schema, [TrainingSample({"r": 0}, 0x10)]
        )
        right = BitLinearModel.train(
            schema, [TrainingSample({"r": 0}, 0x20)]
        )
        with self.assertRaises(ConflictingSamplesError):
            BitLinearModel.merge((left, right))

    def test_out_of_span_input_is_rejected_with_feature_names(self):
        schema = FeatureSchema(operands=(OperandField("r", 3),))
        model = BitLinearModel.train(
            schema,
            [
                TrainingSample({"r": 0}, 0x100),
                TrainingSample({"r": 1}, 0x200),
            ],
        )

        self.assertTrue(model.supports({"r": 1}))
        self.assertFalse(model.supports({"r": 2}))
        feature_vector = schema.vector({"r": 2})
        self.assertTrue(
            any(
                bin(feature_vector & constraint).count("1") % 2
                for constraint in model.nullspace_constraints
            )
        )
        with self.assertRaises(OutOfSpanError) as caught:
            model.encode({"r": 2})
        self.assertIn("operand:r:1", caught.exception.residual_features)
        self.assertIn("outside the training row span", str(caught.exception))

    def test_optional_modifier_can_be_absent(self):
        schema = FeatureSchema(
            modifiers=(ModifierEnum("sat", ("sat",), required=False),)
        )
        model = BitLinearModel.train(
            schema,
            [
                TrainingSample({}, 1 << 4),
                TrainingSample({}, (1 << 4) | (1 << 90), {"sat": "sat"}),
            ],
        )
        self.assertEqual(model.encode({}), 1 << 4)
        self.assertEqual(
            model.encode({}, {"sat": "sat"}), (1 << 4) | (1 << 90)
        )

    def test_corrupt_serialized_basis_is_rejected(self):
        schema = FeatureSchema(operands=(OperandField("r", 1),))
        model = BitLinearModel.train(
            schema,
            [TrainingSample({"r": 0}, 1), TrainingSample({"r": 1}, 2)],
        )
        data = model.to_dict()
        data["basis"][0]["word"] = "0x1" + ("0" * 32)
        with self.assertRaisesRegex(
            FeatureValidationError, "unsigned 128-bit"
        ):
            BitLinearModel.from_dict(data)

    def test_standalone_deserialization_preflights_resource_bounds(self):
        schema = FeatureSchema(operands=(OperandField("r", 1),))
        model = BitLinearModel.train(
            schema,
            [TrainingSample({"r": 0}, 1), TrainingSample({"r": 1}, 2)],
        )

        malformed = model.to_dict()
        malformed["schema"]["operands"][0]["width"] = 10**9
        with self.assertRaisesRegex(
            FeatureValidationError,
            "operand width exceeds the supported bound",
        ):
            BitLinearModel.from_dict(malformed)

        malformed = model.to_dict()
        malformed["basis"] = malformed["basis"] * 2
        with self.assertRaisesRegex(
            FeatureValidationError,
            "basis exceeds the supported bound",
        ):
            BitLinearModel.from_dict(malformed)

        malformed = model.to_dict()
        malformed["sample_count"] = 1 << 63
        with self.assertRaisesRegex(
            FeatureValidationError,
            "sample_count is outside the supported bound",
        ):
            BitLinearModel.from_dict(malformed)

        malformed = model.to_dict()
        malformed["basis"][0]["features"] = "0x" + ("0" * 1024)
        with self.assertRaisesRegex(
            FeatureValidationError,
            "basis features exceed the schema width",
        ):
            BitLinearModel.from_dict(malformed)

        malformed = model.to_dict()
        malformed["basis"][0]["word"] = "0x" + ("0" * 1024)
        with self.assertRaisesRegex(
            FeatureValidationError,
            "basis word must fit unsigned 128-bit range",
        ):
            BitLinearModel.from_dict(malformed)

    def test_standalone_json_size_is_bounded_before_parse(self):
        with mock.patch(
            "f2asm.bitlinear.MAX_MODEL_JSON_CHARS", 8
        ), self.assertRaisesRegex(
            FeatureValidationError,
            "JSON exceeds the supported size",
        ):
            BitLinearModel.from_json('{"padding":"too large"}')

    def test_cached_pivots_and_output_mask_are_serialization_neutral(self):
        schema = FeatureSchema(operands=(OperandField("r", 2),))
        model = BitLinearModel.train(
            schema,
            [
                TrainingSample({"r": 0}, 0x10),
                TrainingSample({"r": 1}, 0x12),
                TrainingSample({"r": 2}, 0x14),
            ],
        )
        before = model.to_json()

        self.assertEqual(
            model._sorted_pivots,
            tuple(sorted(model._basis, reverse=True)),
        )
        self.assertEqual(
            model.basis_output_mask,
            0x10 | 0x12 | 0x14,
        )
        self.assertTrue(model.supports({"r": 3}))
        self.assertEqual(model.encode({"r": 3}), 0x16)
        self.assertEqual(model.to_json(), before)

    def test_copy_on_write_extension_adds_only_independent_rows(self):
        schema = FeatureSchema(operands=(OperandField("r", 2),))
        base = BitLinearModel.train(
            schema,
            [
                TrainingSample({"r": 0}, 0x10),
                TrainingSample({"r": 1}, 0x12),
            ],
        )
        before = base.to_json()

        extended, report = base.extend(
            [
                TrainingSample({"r": 1}, 0x12),
                TrainingSample({"r": 2}, 0x14),
            ]
        )

        self.assertEqual(base.to_json(), before)
        self.assertFalse(base.supports({"r": 2}))
        self.assertEqual(extended.encode({"r": 2}), 0x14)
        self.assertEqual(report.submitted_samples, 2)
        self.assertEqual(report.independent_rows_added, 1)
        self.assertEqual(report.redundant_samples, 1)
        self.assertEqual(report.rank_before, 2)
        self.assertEqual(report.rank_after, 3)
        batch = BitLinearModel.train(
            schema,
            [
                TrainingSample({"r": 0}, 0x10),
                TrainingSample({"r": 1}, 0x12),
                TrainingSample({"r": 1}, 0x12),
                TrainingSample({"r": 2}, 0x14),
            ],
        )
        self.assertEqual(extended.to_json(), batch.to_json())

    def test_copy_on_write_extension_rejects_conflict_atomically(self):
        schema = FeatureSchema(operands=(OperandField("r", 1),))
        base = BitLinearModel.train(
            schema, [TrainingSample({"r": 0}, 0x10)]
        )
        before = base.to_json()

        with self.assertRaises(ConflictingSamplesError):
            base.extend([TrainingSample({"r": 0}, 0x11)])

        self.assertEqual(base.to_json(), before)


if __name__ == "__main__":
    unittest.main()
