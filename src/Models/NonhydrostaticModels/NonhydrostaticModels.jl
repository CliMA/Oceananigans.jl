module NonhydrostaticModels

export NonhydrostaticModel

using DocStringExtensions

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Solvers
using Oceananigans.Operators
using Oceananigans.Distributed: MultiArch, DistributedFFTBasedPoissonSolver, reconstruct_global_grid   
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

import Oceananigans: fields, prognostic_fields

struct MatrixPoissonSolver{S, R}
    solver::S
    storage::R
end

function MatrixPoissonSolver(arch; grid)
    poisson_coeffs = compute_poisson_weights(grid)
    template       = arch_array(arch, zeros(prod(size(grid))))
    solver         = HeptadiagonalIterativeSolver(poisson_coeffs; template, grid)
    return MatrixPoissonSolver(solver, template)
end

function compute_poisson_weights(grid)
    N = size(grid)
    Ax = zeros(N...)
    Ay = zeros(N...)
    Az = zeros(N...)
    C  = zeros(grid, N...)
    D  = zeros(grid, N...)
    for k = 1:grid.Nz, j = 1:grid.Ny, i = 1:grid.Nx
        Ax[i, j, k] = Δzᶜᶜᶜ(i, j, k, grid) * Δyᶠᶜᶜ(i, j, k, grid) / Δxᶠᶜᶜ(i, j, k, grid)
        Ay[i, j, k] = Δzᶜᶜᶜ(i, j, k, grid) * Δxᶜᶠᶜ(i, j, k, grid) / Δyᶜᶠᶜ(i, j, k, grid)
        Az[i, j, k] = Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid) / Δzᶜᶜᶠ(i, j, k, grid)
    end

    return (Ax, Ay, Az, C, D)
end

function PressureSolver(arch::MultiArch, local_grid::RegRectilinearGrid;)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFFTBasedPoissonSolver(global_grid, local_grid)
end

PressureSolver(arch, grid::RegRectilinearGrid)  = FFTBasedPoissonSolver(grid)
PressureSolver(arch, grid::HRegRectilinearGrid) = FourierTridiagonalPoissonSolver(grid)
PressureSolver(arch, ibg::ImmersedBoundaryGrid) = MatrixPoissonSolver(arch, grid = ibg)

#####
##### NonhydrostaticModel definition
#####

include("nonhydrostatic_model.jl")
include("pressure_field.jl")
include("show_nonhydrostatic_model.jl")
include("set_nonhydrostatic_model.jl")

#####
##### Time-stepping NonhydrostaticModels
#####

"""
    fields(model::NonhydrostaticModel)

Return a flattened `NamedTuple` of the fields in `model.velocities`, `model.tracers`, and any
auxiliary fields for a `NonhydrostaticModel` model.
"""
fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers, model.auxiliary_fields)

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `NonhydrostaticModel`.
"""
prognostic_fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers)

include("solve_for_pressure.jl")
include("update_hydrostatic_pressure.jl")
include("update_nonhydrostatic_model_state.jl")
include("pressure_correction.jl")
include("nonhydrostatic_tendency_kernel_functions.jl")
include("calculate_nonhydrostatic_tendencies.jl")
include("correct_nonhydrostatic_immersed_tendencies.jl")

end # module
