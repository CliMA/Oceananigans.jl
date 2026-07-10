module NonhydrostaticModels

export NonhydrostaticModel, BackgroundField, BackgroundFields

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @index, @kernel

using Oceananigans: Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: DistributedComputations,
                                            reconstruct_global_grid, Distributed,
                                            DistributedFFTBasedPoissonSolver,
                                            DistributedFourierTridiagonalPoissonSolver
using Oceananigans.Grids
using Oceananigans.Grids: XYZRegularRG
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using Oceananigans.Solvers
using Oceananigans.Solvers: GridWithFFTSolver, GridWithFourierTridiagonalSolver,
                            ConjugateGradientPoissonSolver, FreeSurfaceLaplacian,
                            fourier_tridiagonal_free_surface_solver, no_gauge_enforcement!
using Oceananigans.Utils
using Oceananigans.Utils: sum_of_velocities

using ..Models: initialize_boundary_transport

import Oceananigans: fields, prognostic_fields
import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.Simulations: timestepper
import Oceananigans.TimeSteppers: step_lagrangian_particles!, update_state!

function nonhydrostatic_pressure_solver(::Distributed, local_grid::XYZRegularRG, ::Nothing)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFFTBasedPoissonSolver(global_grid, local_grid)
end

function nonhydrostatic_pressure_solver(::Distributed, local_grid::GridWithFourierTridiagonalSolver, ::Nothing)
    global_grid = reconstruct_global_grid(local_grid)
    return DistributedFourierTridiagonalPoissonSolver(global_grid, local_grid)
end

# XYZRegularRG <: GridWithFourierTridiagonalSolver, so on fully-regular grids the FFT
# methods shadow the Fourier-tridiagonal ones by specificity.
nonhydrostatic_pressure_solver(arch, grid::XYZRegularRG, ::Nothing) = FFTBasedPoissonSolver(grid)
nonhydrostatic_pressure_solver(arch, grid::GridWithFourierTridiagonalSolver, ::Nothing) = FourierTridiagonalPoissonSolver(grid)

# Free surface: the Robin boundary condition on pressure is solved directly with a
# Fourier-tridiagonal solver — z-tridiagonal InhomogeneousFormulation on grids with
# uniform x and y, RobinEigenbasisFormulation on x- or y-stretched grids. The per-grid
# choice lives in `fourier_tridiagonal_free_surface_solver`; this single union method is
# ambiguity-free against the `::Nothing` tiers above because it is strictly less specific
# than each of them.
nonhydrostatic_pressure_solver(arch, grid::GridWithFFTSolver, free_surface) = fourier_tridiagonal_free_surface_solver(grid)

# fallback
nonhydrostatic_pressure_solver(arch, grid, ::Nothing) = ConjugateGradientPoissonSolver(grid)

const IBGWithFFT = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:GridWithFFTSolver}
nonhydrostatic_pressure_solver(arch, ibg::IBGWithFFT, ::Nothing) = naive_solver_with_warning(arch, ibg, nothing)

# IBGWithFFT + free_surface: the free-surface FT solver on the underlying grid handles the
# Robin boundary condition exactly, so CG only corrects for the immersed boundary.
function nonhydrostatic_pressure_solver(arch, ibg::IBGWithFFT, free_surface)
    preconditioner = fourier_tridiagonal_free_surface_solver(ibg.underlying_grid)
    return ConjugateGradientPoissonSolver(ibg;
                                         linear_operation = FreeSurfaceLaplacian(),
                                         preconditioner,
                                         enforce_gauge_condition! = no_gauge_enforcement!)
end

function naive_solver_with_warning(arch, ibg, free_surface)
    msg = """The FFT-based pressure_solver for NonhydrostaticModels on ImmersedBoundaryGrid
          is approximate and will probably produce velocity fields that are divergent
          adjacent to the immersed boundary. An experimental but improved pressure_solver
          is available which may be used by writing

              using Oceananigans.Solvers: ConjugateGradientPoissonSolver
              pressure_solver = ConjugateGradientPoissonSolver(grid)

          Please report issues to https://github.com/CliMA/Oceananigans.jl/issues.
          """
    @warn msg

    return nonhydrostatic_pressure_solver(arch, ibg.underlying_grid, free_surface)
end


nonhydrostatic_pressure_solver(grid, free_surface) = nonhydrostatic_pressure_solver(architecture(grid), grid, free_surface)

#####
##### NonhydrostaticModel definition
#####

include("background_fields.jl")
include("enforce_net_zero_transport.jl")
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
$(TYPEDSIGNATURES)

Return a flattened `NamedTuple` of the fields in `model.velocities`, `model.tracers`, and any
auxiliary fields for a `NonhydrostaticModel` model.
"""
fields(model::NonhydrostaticModel) = merge(model.velocities,
                                           model.tracers,
                                           model.auxiliary_fields,
                                           biogeochemical_auxiliary_fields(model.biogeochemistry))

"""
$(TYPEDSIGNATURES)

Return a flattened `NamedTuple` of the prognostic fields associated with `NonhydrostaticModel`.
"""
prognostic_fields(model::NonhydrostaticModel) = merge(model.velocities, model.tracers)

# Unpack model.particles to update particle properties. See Models/LagrangianParticleTracking/LagrangianParticleTracking.jl
step_lagrangian_particles!(model::NonhydrostaticModel, Δt) = step_lagrangian_particles!(model.particles, model, Δt)

include("cache_nonhydrostatic_tendencies.jl")
include("nonhydrostatic_ab2_step.jl")
include("nonhydrostatic_rk3_substep.jl")
include("solve_for_pressure.jl")
include("update_hydrostatic_pressure.jl")
include("update_nonhydrostatic_model_state.jl")
include("pressure_correction.jl")
include("nonhydrostatic_tendency_kernel_functions.jl")
include("compute_nonhydrostatic_tendencies.jl")
include("compute_nonhydrostatic_buffer_tendencies.jl")

end # module
