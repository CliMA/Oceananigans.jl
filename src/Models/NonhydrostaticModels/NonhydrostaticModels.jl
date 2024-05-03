module NonhydrostaticModels

export NonhydrostaticModel

using DocStringExtensions

using KernelAbstractions: @index, @kernel

using Oceananigans.Utils
using Oceananigans.Grids
using Oceananigans.Solvers

using Oceananigans.DistributedComputations: Distributed, DistributedFFTBasedPoissonSolver, reconstruct_global_grid   
using Oceananigans.Grids: XYRegularRG, XZRegularRG, YZRegularRG, XYZRegularRG
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: SumOfArrays

import Oceananigans: fields, prognostic_fields
import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.TimeSteppers: step_lagrangian_particles!

function PressureSolver(arch::Distributed, local_grid::XYZRegularRG)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFFTBasedPoissonSolver(global_grid, local_grid)
end

PressureSolver(arch, grid::XYZRegularRG) = FFTBasedPoissonSolver(grid)
PressureSolver(arch, grid::XYRegularRG)  = FourierTridiagonalPoissonSolver(grid)
PressureSolver(arch, grid::XZRegularRG)  = FourierTridiagonalPoissonSolver(grid)
PressureSolver(arch, grid::YZRegularRG)  = FourierTridiagonalPoissonSolver(grid)

# *Evil grin*
PressureSolver(arch, ibg::ImmersedBoundaryGrid) = PressureSolver(arch, ibg.underlying_grid)

# fall back
PressureSolver(arch, grid) = error("None of the implemented pressure solvers for NonhydrostaticModel \
                                   currently support more than one stretched direction.")

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
include("nonhydrostatic_tendency_kernel_functions.jl")
include("compute_nonhydrostatic_tendencies.jl")
include("compute_nonhydrostatic_boundary_tendencies.jl")

end # module
