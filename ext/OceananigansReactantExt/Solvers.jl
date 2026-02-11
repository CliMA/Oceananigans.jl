module Solvers

using Reactant
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: Bounded, Periodic, Flat, inactive_cell,
                          XDirection, YDirection, ZDirection
using Oceananigans.Operators: divᶜᶜᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ
using Oceananigans.Solvers: FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver
using Oceananigans.Fields: interior
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: plan_forward_transform, plan_backward_transform, copy_real_component!
import Oceananigans.Models.NonhydrostaticModels: compute_source_term!
import ..Architectures: AnyConcreteReactantArray
import ..Grids: ReactantGrid

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
##### B.6.7 workaround: avoid Float64→ComplexF64 type mismatch in KA kernels.
##### KA kernel writes Float64 into a traced real-valued scratch,
##### then broadcast promotes into the complex rhs.
#####

@kernel function _compute_source_term_real!(scratch, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds scratch[i, j, k] = active * δ
end

function compute_source_term!(solver::FFTBasedPoissonSolver{<:ReactantGrid}, ::Nothing, Ũ, Δt)
    rhs = solver.storage
    grid = solver.grid
    # Derive a traced Float64 scratch from the traced ComplexF64 rhs.
    # (CenterField(grid) won't work here: allocations inside @compile produce
    #  untraced ConcretePJRTArrays that KA kernels can't write to.)
    scratch = similar(rhs, real(eltype(rhs)))
    launch!(architecture(solver), grid, :xyz, _compute_source_term_real!, scratch, grid, Ũ)
    rhs .= scratch
    return nothing
end

# --- FourierTridiagonalPoissonSolver: same pattern, but kernel also multiplies by grid spacing ---

@kernel function _fourier_tridiagonal_source_term_real!(scratch, ::XDirection, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds scratch[i, j, k] = active * Δxᶜᶜᶜ(i, j, k, grid) * δ
end

@kernel function _fourier_tridiagonal_source_term_real!(scratch, ::YDirection, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds scratch[i, j, k] = active * Δyᶜᶜᶜ(i, j, k, grid) * δ
end

@kernel function _fourier_tridiagonal_source_term_real!(scratch, ::ZDirection, grid, Ũ)
    i, j, k = @index(Global, NTuple)
    active = !inactive_cell(i, j, k, grid)
    u, v, w = Ũ
    δ = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    @inbounds scratch[i, j, k] = active * Δzᶜᶜᶜ(i, j, k, grid) * δ
end

function compute_source_term!(solver::FourierTridiagonalPoissonSolver{<:ReactantGrid}, ::Nothing, Ũ, Δt)
    rhs  = solver.source_term
    grid = solver.grid
    tdir = solver.batched_tridiagonal_solver.tridiagonal_direction
    scratch = similar(rhs, real(eltype(rhs)))
    launch!(architecture(solver), grid, :xyz, _fourier_tridiagonal_source_term_real!, scratch, tdir, grid, Ũ)
    rhs .= scratch
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
