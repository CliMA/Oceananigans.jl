using CUDA: has_cuda
using OrderedCollections: OrderedDict

using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.Fields: Field, VelocityFields, PressureFields
using Oceananigans.Grids: with_halo
using Oceananigans.TimeSteppers: Clock, TimeStepper
using Oceananigans.Utils: inflate_halo_size, tupleit


mutable struct ShallowWaterModel{G, A<:AbstractArchitecture, T, V, R, U, C, H, TS} <: AbstractModel{TS}
    
                 grid :: G         # Grid of physical points on which `Model` is solved
         architecture :: A         # Computer `Architecture` on which `Model` is run
                clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
            advection :: V         # Advection scheme for velocities _and_ tracers
             coriolis :: R         # Set of parameters for the background rotation rate of `Model`
           velocities :: U         # Container for velocity fields `u`, and `v`
              tracers :: C         # Container for tracer fields
          layer_depth :: H         # depth of the layer
          timestepper :: TS        # Object containing timestepper fields and parameters
end

function ShallowWaterModel(;
                           grid,
  architecture::AbstractArchitecture = CPU(),
                          float_type = Float64,
                               clock = Clock{float_type}(0, 0, 1),
                           advection = CenteredSecondOrder(),
                            coriolis = nothing,
                           velocities = nothing,
                             tracers = nothing,
                         layer_depth = nothing,
                 boundary_conditions = NamedTuple(),
                         timestepper = :QuasiAdamsBashforth2
    )

    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, nothing)
    
    velocities    = VelocityFields(velocities,   architecture, grid, boundary_conditions)
    layer_depth   = PressureFields(layer_depth,  architecture, grid, boundary_conditions)
    
    #timestepper = TimeStepper(timestepper, architecture, grid)
    
    return ShallowWaterModel(grid, architecture, clock, advection, coriolis, velocities, tracers, layer_depth, timestepper)

end

