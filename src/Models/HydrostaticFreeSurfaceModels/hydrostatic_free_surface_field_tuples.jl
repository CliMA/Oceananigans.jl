function hydrostatic_velocity_fields(::Nothing, grid, clock, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid)
    return (u=u, v=v, w=w)
end

function hydrostatic_tendency_fields(velocities, free_surface, grid, tracers, bcs)
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    tracer_tendencies = tracer_tendency_fields(tracers, grid, bcs)
    return merge((u=u, v=v), tracer_tendencies)
end

previous_hydrostatic_tendency_fields(timestepper, args...)         = nothing
previous_hydrostatic_tendency_fields(timestepper::Symbol, args...) = previous_hydrostatic_tendency_fields(Val(timestepper), args...)
previous_hydrostatic_tendency_fields(::QuasiAdamsBashforth2TimeStepper, args...) = hydrostatic_tendency_fields(args...)

previous_hydrostatic_tendency_fields(::Val{:QuasiAdamsBashforth2}, args...) = hydrostatic_tendency_fields(args...)
previous_hydrostatic_tendency_fields(::Val{:SplitRungeKutta}, args...)      = nothing
