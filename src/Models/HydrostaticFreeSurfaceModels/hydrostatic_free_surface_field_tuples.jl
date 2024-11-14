using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField, TracerFields
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKutta3TimeStepper

function HydrostaticFreeSurfaceVelocityFields(::Nothing, grid, clock, bcs=NamedTuple())
    u = XFaceField(grid, boundary_conditions=bcs.u)
    v = YFaceField(grid, boundary_conditions=bcs.v)
    w = ZFaceField(grid)
    return (u=u, v=v, w=w)
end

function HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, grid, tracer_names)
    u = XFaceField(grid)
    v = YFaceField(grid)
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v), tracers)
end

function HydrostaticFreeSurfaceTendencyFields(velocities, free_surface::ExplicitFreeSurface, grid, tracer_names)
    u = XFaceField(grid)
    v = YFaceField(grid)
    η = free_surface_displacement_field(velocities, free_surface, grid)
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v, η=η), tracers)
end

function HydrostaticFreeSurfaceTendencyFields(velocities, free_surface::SplitExplicitFreeSurface, grid, tracer_names)
    u = XFaceField(grid)
    v = YFaceField(grid)
    U = deepcopy(free_surface.barotropic_velocities.U)
    V = deepcopy(free_surface.barotropic_velocities.V)
    tracers = TracerFields(tracer_names, grid)
    return merge((u=u, v=v, U=U, V=V), tracers)
end

PreviousHydrostaticTendencyFields(::Val{:QuasiAdamsBashforth2}, args...) = HydrostaticFreeSurfaceTendencyFields(args...)
PreviousHydrostaticTendencyFields(::Val{:SplitRungeKutta3}, args...) = nothing

function PreviousHydrostaticTendencyFields(::Val{:SplitRungeKutta3}, velocities, free_surface::SplitExplicitFreeSurface, args...)
    U = deepcopy(free_surface.barotropic_velocities.U)
    V = deepcopy(free_surface.barotropic_velocities.V)
    return (; U=U, V=V)
end