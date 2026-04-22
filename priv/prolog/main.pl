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
ladder_action(P, Q, M, nonzero(Target, coeff(M, Target, Den))) :-
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
me_ladder(N, P, Q, M, nonzero(Target, coeff(M, Target, Den))) :-
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
