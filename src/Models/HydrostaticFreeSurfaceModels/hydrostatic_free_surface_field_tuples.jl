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
    η = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v, η=η), tracers)
end

function hydrostatic_tendency_fields(velocities, ::SplitExplicitFreeSurface, grid, tracer_names)
    u = XFaceField(grid)
    v = YFaceField(grid)
    U = Field{Face, Center, Nothing}(grid)
    V = Field{Center, Face, Nothing}(grid)
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v, U=U, V=V), tracers)
end

previous_hydrostatic_tendency_fields(::Val{:QuasiAdamsBashforth2}, velocities, free_surface, grid, tracernames) = hydrostatic_tendency_fields(velocities, free_surface, grid, tracernames)
previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta3}, velocities, free_surface, grid, tracernames) = nothing

function previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta3}, velocities, free_surface::SplitExplicitFreeSurface, grid, tracernames)
    U = Field{Face, Center, Nothing}(grid)
    V = Field{Center, Face, Nothing}(grid)
    η = ZFaceField(grid, indices = (:, :, size(grid, 3)+1))
    return (; U=U, V=V, η=η)
end
