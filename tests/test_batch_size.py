import unittest

from dnqsm.batch_size import largest_fitting_batch


class BatchSizeTests(unittest.TestCase):
    def test_selects_largest_fit_and_checks_next_integer(self):
        for capacity in (1, 3, 6, 9, 31):
            with self.subTest(capacity=capacity):
                measured = []

                def probe(batch):
                    measured.append(batch)
                    return batch <= capacity

                self.assertEqual(largest_fitting_batch(probe), capacity)
                self.assertIn(capacity, measured)
                self.assertIn(capacity + 1, measured)
                self.assertEqual(len(measured), len(set(measured)))

    def test_explicit_maximum_and_no_fit(self):
        measured = []
        self.assertEqual(
            largest_fitting_batch(lambda batch: measured.append(batch) or True, 7),
            7,
        )
        self.assertEqual(max(measured), 7)
        with self.assertRaisesRegex(RuntimeError, "batch size 1"):
            largest_fitting_batch(lambda _: False)
        with self.assertRaises(ValueError):
            largest_fitting_batch(lambda _: True, 0)
