module NonhydrostaticModels

export NonhydrostaticModel

using Oceananigans.Grids
using Oceananigans.Solvers

using Oceananigans.Architectures: CPU, GPU
using Oceananigans.Utils: launch!
using Oceananigans.Distributed: DistributedArch, DistributedFFTBasedPoissonSolver, reconstruct_global_grid   
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

using DocStringExtensions
using KernelAbstractions: @index, @kernel, Event, MultiEvent
using KernelAbstractions.Extras.LoopInfo: @unroll

import Oceananigans: fields, prognostic_fields

PressureSolver(::CPU,                   grid::RegRectilinearGrid)  = FFTBasedPoissonSolver(grid)
PressureSolver(::GPU,                   grid::RegRectilinearGrid)  = FFTBasedPoissonSolver(grid)
PressureSolver(::CPU,                   grid::HRegRectilinearGrid) = FourierTridiagonalPoissonSolver(grid)
PressureSolver(::GPU,                   grid::HRegRectilinearGrid) = FourierTridiagonalPoissonSolver(grid)
PressureSolver(::DistributedArch, local_grid::RegRectilinearGrid)  = DistributedFFTBasedPoissonSolver(local_grid)
PressureSolver(::DistributedArch, local_grid::HRegRectilinearGrid) = DistributedFFTBasedPoissonSolver(local_grid)

# *Evil grin*
PressureSolver(arch, ibg::ImmersedBoundaryGrid) = PressureSolver(arch, ibg.underlying_grid)

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
