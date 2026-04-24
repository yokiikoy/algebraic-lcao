% Minimal operator-algebra rules for the algebraic-lcao prototype.
%
% This file intentionally does not do floating-point evaluation or matrix
% assembly. It emits canonical symbolic terms that Python evaluates.

nonnegative_integer(N) :-
    integer(N),
    N >= 0.

valid_ladder_args(N, P, Q, M) :-
    nonnegative_integer(N),
    nonnegative_integer(P),
    nonnegative_integer(Q),
    nonnegative_integer(M).

% ladder_action(+P, +Q, +M, -Result)
%
% Represents (adag)^P a^Q |M> independently of the bra index.
% The nonzero coefficient is sqrt(M! * Target!) / Den!, where
% Den = M - Q and Target = Den + P.
ladder_action(P, Q, M, zero) :-
    nonnegative_integer(P),
    nonnegative_integer(Q),
    nonnegative_integer(M),
    Q > M.
ladder_action(P, Q, M, nonzero(target(Target), coeff(source(M), target(Target), denominator(Den)))) :-
    nonnegative_integer(P),
    nonnegative_integer(Q),
    nonnegative_integer(M),
    Q =< M,
    Den is M - Q,
    Target is Den + P.

% me_ladder(+N, +P, +Q, +M, -Result)
%
% Matrix element <N| (adag)^P a^Q |M>.
me_ladder(N, P, Q, M, zero) :-
    valid_ladder_args(N, P, Q, M),
    Q > M.
me_ladder(N, P, Q, M, zero) :-
    valid_ladder_args(N, P, Q, M),
    Q =< M,
    Target is M - Q + P,
    N =\= Target.
me_ladder(N, P, Q, M, nonzero(target(Target), coeff(source(M), target(Target), denominator(Den)))) :-
    valid_ladder_args(N, P, Q, M),
    Q =< M,
    Den is M - Q,
    Target is Den + P,
    N =:= Target.

emit_me_ladder(N, P, Q, M) :-
    me_ladder(N, P, Q, M, Result),
    write_canonical(Result),
    nl.

emit_ladder_action(P, Q, M) :-
    ladder_action(P, Q, M, Result),
    write_canonical(Result),
    nl.

% ========================================================================
% Displacement operator finite-sum generator
% ========================================================================
%
% D(beta) = exp(-beta^2/2) * exp(beta adag) * exp(-beta a)
%
% <n | D(beta) | m> = exp(-beta^2/2)
%   * sum_{p=0}^{infinity} sum_{q=0}^{infinity}
%       beta^p / p! * (-beta)^q / q! * <n | (adag)^p a^q | m>
%
% The ladder primitive <n | (adag)^p a^q | m> is nonzero only when
% n = m - q + p, i.e. q = m - n + p.
% This delta constraint reduces the double sum to a single sum over p.
%
% For given n, m, the valid p range is:
%   p >= 0
%   q = m - n + p >= 0  (automatically satisfied when p >= max(0, n-m))
%   q <= m  =>  m - n + p <= m  =>  p <= n
%
% Therefore p runs from max(0, n-m) to n (inclusive).
% When p is in range, q = m - n + p and the ladder coefficient is:
%   sqrt(m! * n!) / (m - q)! = sqrt(m! * n!) / (n - p)!

valid_displacement_args(N, M) :-
    nonnegative_integer(N),
    nonnegative_integer(M).

% displacement_term(+N, +M, -P, -Q, -CoeffStruct)
%
% Enumerates individual BCH-reduced terms for <n | D(beta) | m>.
% Each term exposes:
%   P, Q  : the summation indices
%   CoeffStruct : the ladder coefficient skeleton
%
% On backtracking, returns one term at a time.
% The caller can collect terms into a list or process them individually.
displacement_term(N, M, P, Q, ladder_coeff(source(M), target(N), denominator(Den))) :-
    valid_displacement_args(N, M),
    P_min is max(0, N - M),
    P_max is N,
    between(P_min, P_max, P),
    Q is M - N + P,
    Den is M - Q.

emit_displacement_term(N, M) :-
    displacement_term(N, M, P, Q, Coeff),
    Term = term(p(P), q(Q), Coeff),
    write_canonical(Term),
    nl,
    fail.
emit_displacement_term(_, _).

% displacement_finite_sum(+N, +M, -Representation)
%
% Bundled finite-sum representation for backward compatibility.
% Internally collects all terms from displacement_term/5.
displacement_finite_sum(N, M, displacement_sum(prefactor(exp_minus_half_beta_sq), Terms)) :-
    valid_displacement_args(N, M),
    findall(
        term(p(P), q(Q), Coeff),
        displacement_term(N, M, P, Q, Coeff),
        Terms
    ).

emit_displacement_finite_sum(N, M) :-
    displacement_finite_sum(N, M, Result),
    write_canonical(Result),
    nl.

% ========================================================================
% Gaussian operator term structure
% ========================================================================
%
% For the centered Gaussian operator G_alpha = exp(-alpha x^2),
% the matrix element <n | G_alpha | m> has structural constraints.
% This module captures only parity and index structure, NOT coefficients.

% gaussian_term_structure(+N, +M, -Parity, -Allowed)
%
% Parity:   even   if N + M is even (term may be nonzero)
%           odd    if N + M is odd  (term is structurally zero)
% Allowed:  yes    if Parity = even and N, M >= 0
%           no     otherwise
gaussian_term_structure(N, M, even, yes) :-
    nonnegative_integer(N),
    nonnegative_integer(M),
    0 is (N + M) mod 2.
gaussian_term_structure(N, M, odd, no) :-
    nonnegative_integer(N),
    nonnegative_integer(M),
    1 is (N + M) mod 2.

emit_gaussian_term_structure(N, M) :-
    gaussian_term_structure(N, M, Parity, Allowed),
    Result = gaussian_struct(parity(Parity), allowed(Allowed)),
    write_canonical(Result),
    nl.

% ---------------------------------------------------------------------------
% Gaussian operator term skeleton (index structure only)
% ---------------------------------------------------------------------------
%
% The centered Gaussian operator G_alpha = exp(-alpha x^2) connects
% oscillator states via even-parity transitions only.  Its matrix element
% <n | G_alpha | m> can be written as a finite sum over an internal index K.
%
% Structural constraints on K (index-skeleton level, no coefficients):
%   * parity:  n + m must be even
%   * K ranges over values where both (n - K) and (m - K) are even
%   * K <= min(n, m)
%   * K >= 0
%
% Each valid K corresponds to one term in the finite-sum decomposition.
% The actual coefficient for each K is computed in Python, not here.

% gaussian_term_skeleton(+N, +M, -K)
%
% On backtracking, enumerates all valid summation indices K for the
% matrix element <n | G_alpha | m>.  K is an integer satisfying:
%   0 <= K <= min(N, M)
%   (N - K) mod 2 = 0
%   (M - K) mod 2 = 0
%
% These constraints come from the even-ladder structure of the Gaussian
% operator when expanded in the oscillator basis.
gaussian_term_skeleton(N, M, K) :-
    nonnegative_integer(N),
    nonnegative_integer(M),
    0 is (N + M) mod 2,
    K_min is 0,
    K_max is min(N, M),
    between(K_min, K_max, K),
    0 is (N - K) mod 2,
    0 is (M - K) mod 2.

emit_gaussian_term_skeletons(N, M) :-
    gaussian_term_skeleton(N, M, K),
    Term = gaussian_skeleton(k(K)),
    write_canonical(Term),
    nl,
    fail.
emit_gaussian_term_skeletons(_, _).

% ---------------------------------------------------------------------------
% Gaussian operator term with coefficient skeleton
% ---------------------------------------------------------------------------
%
% For each valid K, the centered Gaussian matrix element <n | G_alpha | m>
% contributes one term with the following structural coefficient skeleton:
%
%   i = (n - k) / 2
%   j = (m - k) / 2
%
%   normalization: sqrt(n! * m!) / sqrt(2^(n+m)) / sqrt(1 + g)
%   term factor:   (a^i / i!) * (a^j / j!) * (b^k / k!)
%
% where a = -g/(1+g), b = 2/(1+g), g = alpha/omega.
%
% Prolog generates only the index skeleton and factorial structure.
% Python evaluates the numeric coefficients using alpha and omega.

% gaussian_term(+N, +M, -K, -TermStruct)
%
% TermStruct = gaussian_term(
%   k(K),
%   i(I),
%   j(J),
%   factorial_skeleton(num([N, M]), den([I, J, K])),
%   power_of_two(N_plus_M)
% )
gaussian_term(N, M, K, gaussian_term(k(K), i(I), j(J), factorial_skeleton(num([N, M]), den([I, J, K])), power_of_two(N_plus_M))) :-
    gaussian_term_skeleton(N, M, K),
    I is (N - K) // 2,
    J is (M - K) // 2,
    N_plus_M is N + M.

emit_gaussian_terms(N, M) :-
    gaussian_term(N, M, _, Term),
    write_canonical(Term),
    nl,
    fail.
emit_gaussian_terms(_, _).
