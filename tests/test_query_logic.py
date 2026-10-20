import unittest

from rmot_tta.query.variants import generate_variants


class QueryLogicTest(unittest.TestCase):
    def test_registered_direction_counterfactual(self):
        spec, variants = generate_variants("white cars on the left")
        self.assertIn("appearance", spec.constraint_types)
        self.assertIn("spatial", spec.constraint_types)
        counterfactual = next(item for item in variants if item.query_type == "q_cf")
        self.assertIn("right", counterfactual.text)

    def test_unsupported_or_and_negation_are_not_rewritten(self):
        spec, variants = generate_variants("cars that are white or black")
        self.assertIsNotNone(spec.unsupported_reason)
        self.assertEqual([item.query_type for item in variants], ["q0"])

    def test_category_hypernym_is_not_counterfactual(self):
        _spec, variants = generate_variants("vehicles")
        self.assertFalse(any(item.query_type == "q_cf" for item in variants))


if __name__ == "__main__":
    unittest.main()
