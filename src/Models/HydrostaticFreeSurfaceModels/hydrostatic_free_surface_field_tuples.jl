using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField, TracerFields
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKutta3TimeStepper

function hydrostatic_velocity_fields(::Nothing, grid, clock, bcs=NamedTuple())
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid)
    return (u=u, v=v, w=w)
end

function hydrostatic_tendency_fields(velocities, free_surface, grid, tracer_names)
    u = XFaceField(grid)
    v = YFaceField(grid)
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v), tracers)
end

function hydrostatic_tendency_fields(velocities, free_surface::ExplicitFreeSurface, grid, tracer_names)
    u = XFaceField(grid)
    v = YFaceField(grid)
    η = free_surface_displacement_field(velocities, free_surface, grid)
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v, η=η), tracers)
end

function hydrostatic_tendency_fields(velocities, free_surface::SplitExplicitFreeSurface, grid, tracer_names)
    u = XFaceField(grid)
    v = YFaceField(grid)
    U = similar(free_surface.barotropic_velocities.U)
    V = similar(free_surface.barotropic_velocities.V)
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v, U=U, V=V), tracers)
end

previous_hydrostatic_tendency_fields(::Val{:QuasiAdamsBashforth2}, args...) = hydrostatic_tendency_fields(args...)
previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta3}, args...) = nothing

function previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta3}, velocities, free_surface::SplitExplicitFreeSurface, args...)
    U = similar(free_surface.barotropic_velocities.U)
    V = similar(free_surface.barotropic_velocities.V)
    η = similar(free_surface.η)
    return (; U=U, V=V, η=η)
end
