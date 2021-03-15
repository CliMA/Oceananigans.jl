using Oceananigans.BoundaryConditions: UVelocityBoundaryConditions, VVelocityBoundaryConditions, TracerBoundaryConditions
using Oceananigans.Fields: XFaceField, YFaceField, CenterField, TracerFields

function HorizontalVelocityFields(velocities, arch, grid)
    u = XFaceField(arch, grid, UVelocityBoundaryConditions(grid))
    v = YFaceField(arch, grid, VVelocityBoundaryConditions(grid))
    return u, v
end

FreeSurfaceDisplacementField(velocities, arch, grid) = CenterField(arch, grid, TracerBoundaryConditions(grid))

function HydrostaticFreeSurfaceTendencyFields(velocities, free_surface, arch, grid, tracer_names)

    u, v = HorizontalVelocityFields(velocities, arch, grid)
    η = FreeSurfaceDisplacementField(velocities, arch, grid)
    tracers = TracerFields(tracer_names, arch, grid)

    return merge((u=u, v=v, η=η), tracers)
end


