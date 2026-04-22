from __future__ import annotations

import math
import unittest

from qmarg.prolog_bridge import (
    DEFAULT_PROLOG_FILE,
    PrologUnavailable,
    PrologQueryError,
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


def direct_ladder_target(n: int, p: int, q: int, m: int) -> int | None:
    if q > m:
        return None
    target = m - q + p
    if n != target:
        return None
    return target


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
        result = parse_ladder_result(
            "nonzero(target(3),coeff(source(2),target(3),denominator(1)))"
        )
        self.assertFalse(result.is_zero())
        self.assertEqual(result.target, 3)
        self.assertIsNotNone(result.coefficient)
        self.assertEqual(result.coefficient.source, 2)
        self.assertEqual(result.coefficient.target, 3)
        self.assertEqual(result.coefficient.denominator, 1)
        self.assertAlmostEqual(result.evaluate(), math.sqrt(12.0))

    def test_query_error_includes_prolog_stderr(self) -> None:
        with self.assertRaises(PrologQueryError) as caught:
            query_ladder_matrix_element(0, 0, 0, 0, prolog_file=DEFAULT_PROLOG_FILE.with_name("missing.pl"))

        message = str(caught.exception)
        self.assertIn("SWI-Prolog query failed", message)
        self.assertIn("missing.pl", message)

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
                target = direct_ladder_target(n, p, q, m)
                self.assertAlmostEqual(result.evaluate(), expected)
                if target is None:
                    self.assertTrue(result.is_zero())
                else:
                    self.assertFalse(result.is_zero())
                    self.assertEqual(result.target, target)
                    self.assertEqual(result.target, n)
                    self.assertIsNotNone(result.coefficient)

    def test_prolog_ladder_small_grid(self) -> None:
        for n in range(4):
            for p in range(4):
                for q in range(4):
                    for m in range(4):
                        with self.subTest(n=n, p=p, q=q, m=m):
                            result = query_ladder_matrix_element(n, p, q, m)
                            expected = direct_ladder_matrix_element(n, p, q, m)
                            target = direct_ladder_target(n, p, q, m)
                            self.assertAlmostEqual(result.evaluate(), expected)
                            if target is None:
                                self.assertTrue(result.is_zero())
                            else:
                                self.assertEqual(result.target, target)

    def test_prolog_ladder_zero_structure_for_invalid_q(self) -> None:
        for n in range(4):
            for p in range(4):
                for m in range(4):
                    q = m + 1
                    with self.subTest(n=n, p=p, q=q, m=m):
                        self.assertTrue(query_ladder_matrix_element(n, p, q, m).is_zero())

    def test_prolog_ladder_zero_structure_for_wrong_bra(self) -> None:
        for p in range(4):
            for q in range(4):
                for m in range(q, q + 4):
                    target = m - q + p
                    n = target + 1
                    with self.subTest(n=n, p=p, q=q, m=m):
                        self.assertTrue(query_ladder_matrix_element(n, p, q, m).is_zero())


if __name__ == "__main__":
    unittest.main()
