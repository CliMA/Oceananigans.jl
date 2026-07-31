module Solvers

using Reactant
using Oceananigans: Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, Periodic, Flat
using Oceananigans.Solvers: ConjugateGradientSolver, initialize_solution!,
    initialize_search_direction!, iterate!

using ReactantCore: @trace

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform
import ..Architectures: AnyConcreteReactantArray

const AnyReactantArray = Union{AnyConcreteReactantArray, Reactant.AnyTracedRArray}
const ReactantAbstractFFTsExt = Base.get_extension(Reactant, :ReactantAbstractFFTsExt)

#####
##### Periodic topology (FFT) - uses ReactantAbstractFFTsExt plans
#####

function plan_forward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantFFTInPlacePlan{T}(dims)
end

function plan_backward_transform(A::AnyReactantArray, ::Periodic, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    T = eltype(A)
    return ReactantAbstractFFTsExt.ReactantIFFTInPlacePlan{T}(dims)
end

#####
##### Bounded topology (DCT) - not yet supported
#####

function plan_forward_transform(A::AnyReactantArray, ::Bounded, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic topology or ExplicitFreeSurface.")
end

function plan_backward_transform(A::AnyReactantArray, ::Bounded, dims, planner_flag=nothing)
    length(dims) == 0 && return nothing
    error("Bounded topology (DCT) not yet supported for Reactant. Use Periodic topology or ExplicitFreeSurface.")
end

#####
##### Flat topology - no transform needed
#####

plan_forward_transform(A::AnyReactantArray, ::Flat, args...) = nothing
plan_backward_transform(A::AnyReactantArray, ::Flat, args...) = nothing

#####
##### Conjugate gradient
#####
#####
##### The eager `solve!` drives `iterate!` from a plain `while` whose predicate reads a residual
##### norm. Under tracing that predicate is a `TracedRNumber{Bool}`, which cannot be used in
##### boolean context, so the loop needs `@trace`.
#####
##### Only the scaffolding is duplicated here. The iteration body is Oceananigans' own `iterate!`,
##### which takes `ρ` and returns the next one rather than storing it on the solver: the solver's
##### `iteration::Int` and `ρⁱ⁻¹::T` fields cannot hold traced values, and a field mutated inside a
##### traced loop would be frozen at its trace-time value anyway. Both therefore stay local here.
#####
##### `track_numbers=false` because the default would try to promote every plain `Number` reachable
##### from the captured `solver` into a traced number. `maxiter::Int` and `iteration::Int` are not
##### captured by any type parameter of `ConjugateGradientSolver`, so the promoted struct type does
##### not exist and tracing fails. The loop-carried scalars are made traced explicitly instead:
##### `ρ` already is one (it comes out of a `dot` over traced fields), and the counter is seeded
##### with `Ops.constant`.
#####

const ReactantCGSolver = ConjugateGradientSolver{<:ReactantState}

function Oceananigans.Solvers.solve!(x, solver::ReactantCGSolver, b, args...)
    q = solver.linear_operator_product

    initialize_solution!(q, x, b, solver, args...)

    # Squared, so the loop predicate compares against ρ = ⟨z, r⟩ directly and needs no `sqrt`.
    tolerance = max(solver.reltol * solver.residual_norm(solver.residual), solver.abstol)
    tolerance² = tolerance^2

    ρ = initialize_search_direction!(solver, args...)
    iteration = Reactant.Ops.constant(0)

    # `&` rather than `&&`: short-circuiting would hide the second predicate from tracing.
    @trace track_numbers = false while (iteration < solver.maxiter) & (ρ > tolerance²)
        ρ = iterate!(x, solver, ρ, args...)
        iteration += 1
    end

    return x
end

end # module
