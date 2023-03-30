module NonhydrostaticModels

export NonhydrostaticModel

using DocStringExtensions

using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.Utils: launch!
using Oceananigans.Grids
using Oceananigans.Solvers
using Oceananigans.Distributed: DistributedArch, DistributedFFTBasedPoissonSolver, reconstruct_global_grid   
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Utils: SumOfArrays

import Oceananigans: fields, prognostic_fields, total_velocities
import Oceananigans.Advection: cell_advection_timescale

function PressureSolver(arch::DistributedArch, local_grid::RegRectilinearGrid)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFFTBasedPoissonSolver(global_grid, local_grid)
end

PressureSolver(arch, grid::RegRectilinearGrid)  = FFTBasedPoissonSolver(grid)
PressureSolver(arch, grid::HRegRectilinearGrid) = FourierTridiagonalPoissonSolver(grid)

# *Evil grin*
PressureSolver(arch, ibg::ImmersedBoundaryGrid) = PressureSolver(arch, ibg.underlying_grid)

# fall back
PressureSolver(arch, grid) = error("None of the implemented pressure solvers for NonhydrostaticModel support horizontally-stretched grids.")

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

cell_advection_timescale(model::NonhydrostaticModel) = cell_advection_timescale(model.grid, model.velocities)

"""
    fields(model::NonhydrostaticModel)

Return a flattened `NamedTuple` of the fields in `model.velocities`, `model.tracers`, and any
auxiliary fields for a `NonhydrostaticModel` model.
"""
fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers, model.auxiliary_fields, biogeochemical_auxiliary_fields(model.biogeochemistry))

"""
    prognostic_fields(model::HydrostaticFreeSurfaceModel)

Return a flattened `NamedTuple` of the prognostic fields associated with `NonhydrostaticModel`.
"""
prognostic_fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers)

"""
    total_velocities(model::NonhydrostaticModel)

Return the total velocity fields (velocity + background velocity) for `NonhydrostaticModel`.
"""
@inline total_velocities(model::NonhydrostaticModel) = (u = SumOfArrays{2}(model.velocities.u, model.background_fields.velocities.u),
                                                        v = SumOfArrays{2}(model.velocities.v, model.background_fields.velocities.v),
                                                        w = SumOfArrays{2}(model.velocities.w, model.background_fields.velocities.w))

include("solve_for_pressure.jl")
include("update_hydrostatic_pressure.jl")
include("update_nonhydrostatic_model_state.jl")
include("pressure_correction.jl")
include("nonhydrostatic_tendency_kernel_functions.jl")
include("calculate_nonhydrostatic_tendencies.jl")
include("correct_nonhydrostatic_immersed_tendencies.jl")

end # module
