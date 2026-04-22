from __future__ import annotations

import math
import unittest

from qmarg.prolog_bridge import (
    PrologUnavailable,
    parse_ladder_result,
    query_ladder_matrix_element,
    swipl_executable,
)


def direct_ladder_matrix_element(n: int, p: int, q: int, m: int) -> float:
    if q > m:
        return 0.0
    target = m - q + p
    if n != target:
        return 0.0
    denominator = m - q
    return math.sqrt(math.factorial(m) * math.factorial(target)) / math.factorial(
        denominator
    )


class PrologBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            swipl_executable()
        except PrologUnavailable as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_parse_ladder_zero(self) -> None:
        result = parse_ladder_result("zero\n")
        self.assertTrue(result.is_zero())
        self.assertEqual(result.evaluate(), 0.0)

    def test_parse_ladder_nonzero(self) -> None:
        result = parse_ladder_result("nonzero(3,coeff(2,3,1))")
        self.assertFalse(result.is_zero())
        self.assertEqual(result.target, 3)
        self.assertAlmostEqual(result.evaluate(), math.sqrt(12.0))

    def test_prolog_ladder_cases(self) -> None:
        cases = [
            (0, 0, 0, 0),
            (2, 1, 0, 1),
            (1, 0, 1, 2),
            (4, 3, 1, 2),
            (0, 0, 3, 2),
            (1, 2, 0, 0),
            (3, 1, 2, 5),
        ]

        for n, p, q, m in cases:
            with self.subTest(n=n, p=p, q=q, m=m):
                result = query_ladder_matrix_element(n, p, q, m)
                expected = direct_ladder_matrix_element(n, p, q, m)
                self.assertAlmostEqual(result.evaluate(), expected)
                if expected == 0.0:
                    self.assertTrue(result.is_zero())
                else:
                    self.assertEqual(result.target, n)

    def test_prolog_ladder_small_grid(self) -> None:
        for n in range(4):
            for p in range(4):
                for q in range(4):
                    for m in range(4):
                        with self.subTest(n=n, p=p, q=q, m=m):
                            result = query_ladder_matrix_element(n, p, q, m)
                            expected = direct_ladder_matrix_element(n, p, q, m)
                            self.assertAlmostEqual(result.evaluate(), expected)


if __name__ == "__main__":
    unittest.main()
