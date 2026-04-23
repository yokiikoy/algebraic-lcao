from __future__ import annotations

import math
import unittest

from qmarg.fock import displacement_matrix_element
from qmarg.fock import origin_gaussian_matrix_element
from qmarg.prolog_bridge import (
    DEFAULT_PROLOG_FILE,
    PrologUnavailable,
    PrologQueryError,
    parse_displacement_finite_sum,
    parse_displacement_terms,
    parse_gaussian_term_structure,
    query_displacement_finite_sum,
    query_displacement_terms,
    query_gaussian_term_structure,
    query_ladder_matrix_element,
    parse_ladder_result,
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

    # -------------------------------------------------------------------
    # Ladder primitive tests
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Displacement finite-sum tests
    # -------------------------------------------------------------------

    def test_parse_displacement_sum(self) -> None:
        text = (
            "displacement_sum(prefactor(exp_minus_half_beta_sq),"
            "[term(p(0),q(1),ladder_coeff(source(3),target(2),denominator(2))),"
            "term(p(1),q(2),ladder_coeff(source(3),target(2),denominator(1)))])"
        )
        result = parse_displacement_finite_sum(text)
        self.assertEqual(len(result.terms), 2)
        self.assertEqual(result.terms[0].p, 0)
        self.assertEqual(result.terms[0].q, 1)
        self.assertEqual(result.terms[1].p, 1)
        self.assertEqual(result.terms[1].q, 2)

    def test_displacement_identity_at_zero_beta(self) -> None:
        """At beta=0, D(0)=I, so <n|D(0)|m> = delta_{n,m}."""
        for n in range(5):
            for m in range(5):
                with self.subTest(n=n, m=m):
                    result = query_displacement_finite_sum(n, m)
                    expected = 1.0 if n == m else 0.0
                    self.assertAlmostEqual(
                        result.evaluate(0.0), expected, places=14
                    )

    def test_displacement_matches_closed_form_small_grid(self) -> None:
        """Prolog-generated finite sum must agree with Python closed form."""
        for n in range(4):
            for m in range(4):
                result = query_displacement_finite_sum(n, m)
                for beta in (-1.2, -0.3, 0.0, 0.5, 1.5):
                    with self.subTest(n=n, m=m, beta=beta):
                        prolog_val = result.evaluate(beta)
                        direct_val = displacement_matrix_element(n, m, beta)
                        self.assertAlmostEqual(prolog_val, direct_val, places=12)

    def test_displacement_symmetry(self) -> None:
        """<n|D(beta)|m> = <m|D(-beta)|n>."""
        for n in range(5):
            for m in range(5):
                for beta in (-1.0, -0.2, 0.7):
                    with self.subTest(n=n, m=m, beta=beta):
                        left = query_displacement_finite_sum(n, m).evaluate(beta)
                        right = query_displacement_finite_sum(m, n).evaluate(-beta)
                        self.assertAlmostEqual(left, right, places=12)

    def test_displacement_finite_sum_explicit_cases(self) -> None:
        """Check specific known cases for structure."""
        # n=0, m=0: only p=0, q=0 term
        result = query_displacement_finite_sum(0, 0)
        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].p, 0)
        self.assertEqual(result.terms[0].q, 0)

        # n=1, m=0: only p=1, q=0 term
        result = query_displacement_finite_sum(1, 0)
        self.assertEqual(len(result.terms), 1)
        self.assertEqual(result.terms[0].p, 1)
        self.assertEqual(result.terms[0].q, 0)

        # n=2, m=3: p from max(0, 2-3)=0 to 2
        result = query_displacement_finite_sum(2, 3)
        self.assertEqual(len(result.terms), 3)
        ps = [t.p for t in result.terms]
        self.assertEqual(ps, [0, 1, 2])

    # -------------------------------------------------------------------
    # Displacement term-generator tests
    # -------------------------------------------------------------------

    def test_term_generator_reconstructs_finite_sum(self) -> None:
        """Term-by-term enumeration must match the bundled finite-sum path."""
        for n in range(5):
            for m in range(5):
                with self.subTest(n=n, m=m):
                    bundled = query_displacement_finite_sum(n, m)
                    terms = query_displacement_terms(n, m)
                    self.assertEqual(len(terms), len(bundled.terms))
                    for i, term in enumerate(terms):
                        bundled_term = bundled.terms[i]
                        self.assertEqual(term.p, bundled_term.p)
                        self.assertEqual(term.q, bundled_term.q)
                        self.assertEqual(
                            term.ladder_coefficient.source,
                            bundled_term.ladder_coefficient.source,
                        )
                        self.assertEqual(
                            term.ladder_coefficient.target,
                            bundled_term.ladder_coefficient.target,
                        )
                        self.assertEqual(
                            term.ladder_coefficient.denominator,
                            bundled_term.ladder_coefficient.denominator,
                        )

    def test_term_generator_evaluates_correctly(self) -> None:
        """Reconstructing the sum from individual terms must match closed form."""
        for n in range(4):
            for m in range(4):
                terms = query_displacement_terms(n, m)
                for beta in (-1.2, -0.3, 0.0, 0.5, 1.5):
                    with self.subTest(n=n, m=m, beta=beta):
                        prefactor = math.exp(-0.5 * beta ** 2)
                        total = sum(
                            term.evaluate_beta_factor(beta)
                            * term.ladder_coefficient.evaluate()
                            for term in terms
                        )
                        reconstructed = prefactor * total
                        direct = displacement_matrix_element(n, m, beta)
                        self.assertAlmostEqual(reconstructed, direct, places=12)

    def test_term_generator_structural_consistency(self) -> None:
        """Each term must satisfy the BCH delta constraint q = m - n + p."""
        for n in range(5):
            for m in range(5):
                with self.subTest(n=n, m=m):
                    terms = query_displacement_terms(n, m)
                    for term in terms:
                        self.assertEqual(term.q, m - n + term.p)

    def test_term_generator_identity_at_zero_beta(self) -> None:
        """At beta=0, D(0)=I, so the reconstructed sum must give delta_{n,m}."""
        for n in range(5):
            for m in range(5):
                with self.subTest(n=n, m=m):
                    terms = query_displacement_terms(n, m)
                    prefactor = math.exp(-0.5 * 0.0 ** 2)
                    total = sum(
                        term.evaluate_beta_factor(0.0)
                        * term.ladder_coefficient.evaluate()
                        for term in terms
                    )
                    expected = 1.0 if n == m else 0.0
                    self.assertAlmostEqual(prefactor * total, expected, places=14)

    def test_term_generator_matches_bundled_evaluation(self) -> None:
        """Both paths must give identical numerical results for all beta."""
        for n in range(4):
            for m in range(4):
                bundled = query_displacement_finite_sum(n, m)
                terms = query_displacement_terms(n, m)
                for beta in (-0.7, 0.0, 0.3):
                    with self.subTest(n=n, m=m, beta=beta):
                        bundled_val = bundled.evaluate(beta)
                        prefactor = math.exp(-0.5 * beta ** 2)
                        term_val = prefactor * sum(
                            t.evaluate_beta_factor(beta)
                            * t.ladder_coefficient.evaluate()
                            for t in terms
                        )
                        self.assertAlmostEqual(bundled_val, term_val, places=14)

    # -------------------------------------------------------------------
    # Gaussian operator term structure tests
    # -------------------------------------------------------------------

    def test_parse_gaussian_structure(self) -> None:
        result = parse_gaussian_term_structure(
            "gaussian_struct(parity(even),allowed(yes))"
        )
        self.assertEqual(result.parity, "even")
        self.assertTrue(result.allowed)
        self.assertFalse(result.is_zero())

        result = parse_gaussian_term_structure(
            "gaussian_struct(parity(odd),allowed(no))"
        )
        self.assertEqual(result.parity, "odd")
        self.assertFalse(result.allowed)
        self.assertTrue(result.is_zero())

    def test_gaussian_parity_rule_even(self) -> None:
        """Prolog marks even n+m as allowed."""
        for n in range(6):
            for m in range(6):
                if (n + m) % 2 == 0:
                    with self.subTest(n=n, m=m):
                        result = query_gaussian_term_structure(n, m)
                        self.assertEqual(result.parity, "even")
                        self.assertTrue(result.allowed)

    def test_gaussian_parity_rule_odd(self) -> None:
        """Prolog marks odd n+m as disallowed (structurally zero)."""
        for n in range(6):
            for m in range(6):
                if (n + m) % 2 == 1:
                    with self.subTest(n=n, m=m):
                        result = query_gaussian_term_structure(n, m)
                        self.assertEqual(result.parity, "odd")
                        self.assertFalse(result.allowed)

    def test_gaussian_structure_agrees_with_python_backend(self) -> None:
        """Prolog structural zeros must match Python backend zeros."""
        omega = 0.8
        alpha = 0.35
        for n in range(8):
            for m in range(8):
                with self.subTest(n=n, m=m):
                    prolog_result = query_gaussian_term_structure(n, m)
                    python_val = origin_gaussian_matrix_element(n, m, omega, alpha)
                    if prolog_result.is_zero():
                        self.assertAlmostEqual(python_val, 0.0, places=14)
                    else:
                        # Python may return 0.0 for accidental zeros,
                        # but parity rule is a necessary condition
                        self.assertNotEqual(prolog_result.parity, "odd")

    def test_gaussian_no_false_positives(self) -> None:
        """Prolog must never mark an odd-parity pair as allowed."""
        for n in range(10):
            for m in range(10):
                with self.subTest(n=n, m=m):
                    result = query_gaussian_term_structure(n, m)
                    if (n + m) % 2 == 1:
                        self.assertFalse(result.allowed)
                        self.assertTrue(result.is_zero())


if __name__ == "__main__":
    unittest.main()
