using Oceananigans: AbstractModel, AbstractOutputWriter, AbstractDiagnostic

using Oceananigans.Architectures: AbstractArchitecture
using Oceananigans.Advection: CenteredSecondOrder
using Oceananigans.BoundaryConditions: regularize_field_boundary_conditions

using Oceananigans.BoundaryConditions: UVelocityBoundaryConditions,
                                       VVelocityBoundaryConditions,
                                       TracerBoundaryConditions

using Oceananigans.Fields: XFaceField, YFaceField, CellField

using Oceananigans.Fields: Field, tracernames, TracerFields
using Oceananigans.Grids: with_halo
using Oceananigans.TimeSteppers: Clock, TimeStepper, RungeKutta3TimeStepper
using Oceananigans.Utils: inflate_halo_size, tupleit

function ShallowWaterTendencyFields(arch, grid, tracer_names)

    uh = XFaceField(arch, grid, UVelocityBoundaryConditions(grid))
    vh = YFaceField(arch, grid, VVelocityBoundaryConditions(grid))
    h  = CellField(arch,  grid, TracerBoundaryConditions(grid))
    tracers = TracerFields(tracer_names, arch, grid)
    
    return merge((uh=uh, vh=vh, h=h), tracers)
end

function ShallowWaterSolutionFields(arch, grid, bcs)
    
    uh_bcs = :uh ∈ keys(bcs) ? bcs.uh : UVelocityBoundaryConditions(grid)
    vh_bcs = :vh ∈ keys(bcs) ? bcs.vh : VVelocityBoundaryConditions(grid)
    h_bcs  = :h  ∈ keys(bcs) ? bcs.h  : TracerBoundaryConditions(grid)

    uh = XFaceField(arch, grid, uh_bcs)
    vh = YFaceField(arch, grid, vh_bcs)
    h = CellField(arch, grid, h_bcs)

    return (uh=uh, vh=vh, h=h)
end

struct ShallowWaterModel{G, A<:AbstractArchitecture, T, V, R, E, Q, C, TS} <: AbstractModel{TS}
    
                 grid :: G         # Grid of physical points on which `Model` is solved
         architecture :: A         # Computer `Architecture` on which `Model` is run
                clock :: Clock{T}  # Tracks iteration number and simulation time of `Model`
            advection :: V         # Advection scheme for velocities _and_ tracers
             coriolis :: R         # Set of parameters for the background rotation rate of `Model`
              closure :: E         # Diffusive 'turbulence closure' for all model fields
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
                             closure = nothing,
                            solution = nothing,
                             tracers = NamedTuple(),
                 boundary_conditions = NamedTuple(),
                         timestepper = nothing
#                         timestepper = RungeKutta3
    )

    grid.Nz == 1 || throw(ArgumentError("ShallowWaterModel must be constructed with Nz=1!"))

    tracers = tupleit(tracers) # supports tracers=:c keyword argument (for example)

    Hx, Hy, Hz = inflate_halo_size(grid.Hx, grid.Hy, grid.Hz, advection)
    grid = with_halo((Hx, Hy, Hz), grid)
    
    boundary_conditions = regularize_field_boundary_conditions(boundary_conditions, grid, nothing)
    
    solution = ShallowWaterSolutionFields(architecture, grid, boundary_conditions)
    tracers  = TracerFields(tracers, architecture, grid, boundary_conditions)

    timestepper = RungeKutta3TimeStepper(architecture, grid, tracernames(tracers);
                                         Gⁿ = ShallowWaterTendencyFields(architecture, grid, tracernames(tracers)),
                                         G⁻ = ShallowWaterTendencyFields(architecture, grid, tracernames(tracers)))

    return ShallowWaterModel(grid,
                             architecture,
                             clock,
                             advection,
                             coriolis,
                             closure,
                             solution,
                             tracers,
                             timestepper)
end

using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: work_layout
using Oceananigans.Architectures: device

import Oceananigans.TimeSteppers: rk3_substep!

function rk3_substep!(model::ShallowWaterModel, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    substep_solution_kernel! = rk3_substep_solution!(device(model.architecture), workgroup, worksize)
    substep_tracer_kernel! = rk3_substep_tracer!(device(model.architecture), workgroup, worksize)


    solution_event = substep_solution_kernel!(model.solution,
                                              Δt, γⁿ, ζⁿ,
                                              model.timestepper.Gⁿ,
                                              model.timestepper.G⁻;
                                              dependencies=barrier)

    events = [solution_event]

    for i in 1:length(model.tracers)
        @inbounds c = model.tracers[i]
        @inbounds Gcⁿ = model.timestepper.Gⁿ[i+3]
        @inbounds Gc⁻ = model.timestepper.G⁻[i+3]

        tracer_event = substep_tracer_kernel!(c, Δt, γⁿ, ζⁿ, Gcⁿ, Gc⁻, dependencies=barrier)

        push!(events, tracer_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

"""
Time step tracers via the 3rd-order Runge-Kutta method
    `c^{m+1} = c^m + Δt (γⁿ Gc^{m} + ζⁿ Gc^{m-1})`,
where `m` denotes the substage. 
"""
@kernel function rk3_substep_tracer!(c, Δt, γⁿ, ζⁿ, Gcⁿ, Gc⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds c[i, j, k] += Δt * (γⁿ * Gcⁿ[i, j, k] + ζⁿ * Gc⁻[i, j, k])
end

"""
Time step tracers from the first to the second stage via
the 3rd-order Runge-Kutta method
    `c^{2} = c^1 + Δt γ¹ Gc^{1}`.
"""
@kernel function rk3_substep_tracer!(c, Δt, γ¹, ::Nothing, Gc¹, Gc⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds c[i, j, k] += Δt * γ¹ * Gc¹[i, j, k]
end


"""
Time step solution fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_solution!(U, Δt, γⁿ, ζⁿ, Gⁿ, G⁻)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.uh[i, j, k] += Δt * (γⁿ * Gⁿ.uh[i, j, k] + ζⁿ * G⁻.uh[i, j, k])
        U.vh[i, j, k] += Δt * (γⁿ * Gⁿ.vh[i, j, k] + ζⁿ * G⁻.vh[i, j, k])
        U.h[i, j, k]  += Δt * (γⁿ * Gⁿ.h[i, j, k]  + ζⁿ * G⁻.h[i, j, k])
    end
end

"""
Time step solution fields with a 3rd-order Runge-Kutta method.
"""
@kernel function rk3_substep_solution!(U, Δt, γ¹, ::Nothing, G¹, G⁰)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        U.uh[i, j, k] += Δt * γ¹ * G¹.uh[i, j, k]
        U.vh[i, j, k] += Δt * γ¹ * G¹.vh[i, j, k]
        U.h[i, j, k]  += Δt * γ¹ * G¹.h[i, j, k]
    end
end


import Oceananigans.TimeSteppers: store_tendencies!

""" Store source terms for `uh`, `vh`, and `h`. """
@kernel function store_solution_tendencies!(G⁻, grid::AbstractGrid{FT}, G⁰) where FT
    i, j, k = @index(Global, NTuple)
    @inbounds G⁻.uh[i, j, k] = G⁰.uh[i, j, k]
    @inbounds G⁻.vh[i, j, k] = G⁰.vh[i, j, k]
    @inbounds G⁻.h[i, j, k]  = G⁰.h[i, j, k]
end

""" Store previous source terms for a tracer before updating them. """
@kernel function store_tracer_tendency!(Gc⁻, grid::AbstractGrid{FT}, Gc⁰) where FT
    i, j, k = @index(Global, NTuple)
    @inbounds Gc⁻[i, j, k] = Gc⁰[i, j, k]
end


""" Store previous source terms before updating them. """
function store_tendencies!(model::ShallowWaterModel)

    barrier = Event(device(model.architecture))

    workgroup, worksize = work_layout(model.grid, :xyz)

    store_solution_tendencies_kernel! = store_solution_tendencies!(device(model.architecture), workgroup, worksize)
    store_tracer_tendency_kernel! = store_tracer_tendency!(device(model.architecture), workgroup, worksize)

    solution_event = store_solution_tendencies_kernel!(model.timestepper.G⁻,
                                                       model.grid,
                                                       model.timestepper.Gⁿ,
                                                       dependencies=barrier)

    events = [solution_event]

    # Tracer fields
    for i in 4:length(model.timestepper.G⁻)
        @inbounds Gc⁻ = model.timestepper.G⁻[i]
        @inbounds Gc⁰ = model.timestepper.Gⁿ[i]
        tracer_event = store_tracer_tendency_kernel!(Gc⁻, model.grid, Gc⁰, dependencies=barrier)
        push!(events, tracer_event)
    end

    wait(device(model.architecture), MultiEvent(Tuple(events)))

    return nothing
end

