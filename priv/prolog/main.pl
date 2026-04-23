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
