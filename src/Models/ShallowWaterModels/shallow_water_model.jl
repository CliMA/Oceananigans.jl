using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: Field, tracernames, TracerFields
using Oceananigans.Grids: with_halo
using Oceananigans.TimeSteppers: Clock, TimeStepper
using Oceananigans.Utils: inflate_halo_size, tupleit

function ShallowWaterSolutionFields(arch, grid, bcs)
    uh_bcs = :uh ∈ keys(bcs) ? bcs.uh : UVelocityBoundaryConditions(grid)
    vh_bcs = :vh ∈ keys(bcs) ? bcs.v : VVelocityBoundaryConditions(grid)
    h_bcs = :h ∈ keys(bcs) ? bcs.w : TracerBoundaryConditions(grid)

    uh = XFaceField(arch, grid, u_bcs)
    vh = YFaceField(arch, grid, v_bcs)
    h = CellField(arch, grid, w_bcs)

    return (uh=uh, vh=vh, h=h)
end

mutable struct ShallowWaterModel{G, A<:AbstractArchitecture, T, V, R, Q, C, TS} <: AbstractModel{TS}
    
                 grid :: G         # Grid of physical points on which `Model` is solved
         architecture :: A         # Computer `Architecture` on which `Model` is run
                clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
            advection :: V         # Advection scheme for velocities _and_ tracers
             coriolis :: R         # Set of parameters for the background rotation rate of `Model`
             solution :: Q         # Container for transports `uh`, `vh`, and height `h`
              tracers :: C         # Container for tracer fields
          timestepper :: TS        # Object containing timestepper fields and parameters
end

function ShallowWaterModel(;
                           grid,
  architecture::AbstractArchitecture = CPU(),
                          float_type = Float64,
                               clock = Clock{float_type}(0, 0, 1),
                           advection = CenteredSecondOrder(),
                            coriolis = nothing,
                            solution = nothing,
                             tracers = NamedTuple(),
                 boundary_conditions = NamedTuple(),
                         timestepper = :RungeKutta3
    )

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    Hx, Hy, Hz = inflate_halo_size(grid.Hx, grid.Hy, grid.Hz, advection)
    grid = with_halo((Hx, Hy, Hz), grid)
    
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, nothing)
    
    solution = ShallowWaterSolutionFields(architecture, grid, boundary_conditions)
    tracers  = TracerFields(tracers, architecture, grid, boundary_conditions)
    
    timestepper = nothing #TimeStepper(timestepper, architecture, grid, tracernames(tracers))
    
    return ShallowWaterModel(grid,
                             architecture,
                             clock,
                             advection,
                             coriolis,
                             solution,
                             tracers,
                             timestepper)
end
