module Solvers

using Reactant
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: Bounded, Periodic, Flat
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver,
                            BatchedTridiagonalSolver
using Oceananigans.Models.NonhydrostaticModels: _compute_source_term!, _fourier_tridiagonal_source_term!
using Oceananigans.Fields: interior, indices
using Oceananigans.Utils: launch!

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform, copy_real_component!, solve!
import Oceananigans.Models.NonhydrostaticModels: compute_source_term!
import ..Architectures: AnyConcreteReactantArray
import ..Grids: ReactantGrid

# Type aliases with eltype parameter for dispatch on Complex arrays
const AnyReactantArray{T} = Union{
    Reactant.AnyConcretePJRTArray{T},
    Reactant.AnyConcreteIFRTArray{T},
    Reactant.AnyTracedRArray{T}
}
const ComplexReactantArray = AnyReactantArray{<:Complex}
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
##### B.6.7 workaround: avoid Float64→ComplexF64 type mismatch in KA kernels.
##### Reuse the existing source-code kernels (_compute_source_term!, _fourier_tridiagonal_source_term!)
##### but target a Float64 scratch array, then broadcast into the complex rhs.
#####

function compute_source_term!(solver::FFTBasedPoissonSolver{<:ReactantGrid}, ::Nothing, Ũ, Δt)
    rhs = solver.storage
    grid = solver.grid
    scratch = similar(rhs, real(eltype(rhs)))
    launch!(architecture(solver), grid, :xyz, _compute_source_term!, scratch, grid, Ũ)
    rhs .= scratch
    return nothing
end

function compute_source_term!(solver::FourierTridiagonalPoissonSolver{<:ReactantGrid}, ::Nothing, Ũ, Δt)
    rhs  = solver.source_term
    grid = solver.grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    scratch = similar(rhs, real(eltype(rhs)))
    launch!(architecture(solver), grid, :xyz, _fourier_tridiagonal_source_term!, scratch, tdir, grid, Ũ)
    rhs .= scratch
    return nothing
end

#####
##### B.6.7 workaround: split complex tridiagonal solve into two real solves.
##### The tridiagonal coefficients (a, b, c) are real, so real and imaginary parts decouple.
##### Dispatches on ComplexReactantArray for ϕ so the original FourierTridiagonalPoissonSolver
##### solve! is untouched — only the inner BatchedTridiagonalSolver call is intercepted.
#####

function solve!(ϕ::ComplexReactantArray, solver::BatchedTridiagonalSolver, rhs, args...)
    T = real(eltype(ϕ))
    real_ϕ = similar(ϕ, T)
    imag_ϕ = similar(ϕ, T)

    # Coefficients a, b, c and scratch t are already real — only ϕ and rhs are complex.
    solve!(real_ϕ, solver, real.(rhs), args...)
    solve!(imag_ϕ, solver, imag.(rhs), args...)

    ϕ .= Complex.(real_ϕ, imag_ϕ)
    return nothing
end

#####
##### B.6.7 workaround: avoid ComplexF64 reads in KA kernels.
##### Use broadcast to extract real component and copy via interior view (no KA kernel needed).
#####

function copy_real_component!(grid::ReactantGrid, ϕ, ϕc, index_ranges)
    interior(ϕ) .= real.(ϕc)
    return nothing
end

end # module
