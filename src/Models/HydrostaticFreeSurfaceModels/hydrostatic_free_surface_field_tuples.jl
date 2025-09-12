function hydrostatic_velocity_fields(::Nothing, grid, clock, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid)
    return (u=u, v=v, w=w)
end

function hydrostatic_tendency_fields(velocities, free_surface, grid, tracer_names, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    tracers = TracerFields(tracer_names, grid, bcs)
    return merge((u=u, v=v), tracers)
end

function hydrostatic_tendency_fields(velocities, free_surface::ExplicitFreeSurface, grid, tracer_names, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    η = free_surface_displacement_field(velocities, free_surface, grid)
    tracers = TracerFields(tracer_names, grid, bcs)
    return merge((u=u, v=v, η=η), tracers)
end

previous_hydrostatic_tendency_fields(::Val{:QuasiAdamsBashforth2}, args...) = hydrostatic_tendency_fields(args...)
previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta3}, args...)     = nothing
