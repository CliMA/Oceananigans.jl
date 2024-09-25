module NonhydrostaticModels

export NonhydrostaticModel

using DocStringExtensions

using KernelAbstractions: @index, @kernel

using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Solvers

using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: reconstruct_global_grid, Distributed
using Oceananigans.DistributedComputations: DistributedFFTBasedPoissonSolver, DistributedFourierTridiagonalPoissonSolver
using Oceananigans.Grids: XYRegularRG, XZRegularRG, YZRegularRG, XYZRegularRG
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: SumOfArrays

import Oceananigans: fields, prognostic_fields
import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.TimeSteppers: step_lagrangian_particles!

function nonhydrostatic_pressure_solver(::Distributed, local_grid::XYZRegularRG)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFFTBasedPoissonSolver(global_grid, local_grid)
end

function nonhydrostatic_pressure_solver(::Distributed, local_grid::XYRegularRG)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)
end

function nonhydrostatic_pressure_solver(::Distributed, local_grid::YZRegularRG)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)
end

function nonhydrostatic_pressure_solver(::Distributed, local_grid::XZRegularRG)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)
end

nonhydrostatic_pressure_solver(arch, grid::XYZRegularRG) = FFTBasedPoissonSolver(grid)
nonhydrostatic_pressure_solver(arch, grid::XYRegularRG)  = FourierTridiagonalPoissonSolver(grid)
nonhydrostatic_pressure_solver(arch, grid::XZRegularRG)  = FourierTridiagonalPoissonSolver(grid)
nonhydrostatic_pressure_solver(arch, grid::YZRegularRG)  = FourierTridiagonalPoissonSolver(grid)

#=
function nonhydrostatic_pressure_solver(arch, ibg::ImmersedBoundaryGrid)
    msg = string("The nonhydrostatic_pressure_solver for `NonhydrostaticModel`s on `ImmersedBoundaryGrid`", '\n',
                 "is approximate. In particular, the pressure correction step does not produce a velocity", '\n',
                 "field that both satisfies impenetrability at solid walls and is also divergence-free.", '\n',
                 "As a result, boundary-adjacent velocity fields may be divergent.", '\n',
                 "Please report issues to https://github.com/CliMA/Oceananigans.jl/issues.")
    @warn msg

    return nonhydrostatic_pressure_solver(arch, ibg.underlying_grid)
end
=#

# fall back
nonhydrostatic_pressure_solver(arch, grid) = error("None of the implemented pressure solvers for NonhydrostaticModel \
                                    currently support more than one stretched direction.")

nonhydrostatic_pressure_solver(grid) = nonhydrostatic_pressure_solver(architecture(grid), grid)

#####
##### NonhydrostaticModel definition
#####

include("nonhydrostatic_model.jl")
include("pressure_field.jl")
include("show_nonhydrostatic_model.jl")
include("set_nonhydrostatic_model.jl")

#####
##### AbstractModel interface
#####

function cell_advection_timescale(model::NonhydrostaticModel)
    grid = model.grid
    velocities = total_velocities(model)
    return cell_advection_timescale(grid, velocities)
end

"""
    fields(model::NonhydrostaticModel)

Return a flattened `NamedTuple` of the fields in `model.velocities`, `model.tracers`, and any
auxiliary fields for a `NonhydrostaticModel` model.
"""
fields(model::NonhydrostaticModel) = merge(model.velocities,
                                           model.tracers,
                                           model.auxiliary_fields,
                                           biogeochemical_auxiliary_fields(model.biogeochemistry))

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `NonhydrostaticModel`.
"""
prognostic_fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers)

# Unpack model.particles to update particle properties. See Models/LagrangianParticleTracking/LagrangianParticleTracking.jl
step_lagrangian_particles!(model::NonhydrostaticModel, Δt) = step_lagrangian_particles!(model.particles, model, Δt)

include("solve_for_pressure.jl")
include("update_hydrostatic_pressure.jl")
include("update_nonhydrostatic_model_state.jl")
include("pressure_correction.jl")
include("immersed_poisson_solver.jl")
include("nonhydrostatic_tendency_kernel_functions.jl")
include("compute_nonhydrostatic_tendencies.jl")
include("compute_nonhydrostatic_boundary_tendencies.jl")

end # module
