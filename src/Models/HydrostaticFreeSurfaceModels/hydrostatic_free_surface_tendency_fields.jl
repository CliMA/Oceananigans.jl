using Oceananigans.BoundaryConditions: UVelocityBoundaryConditions, VVelocityBoundaryConditions, TracerBoundaryConditions
using Oceananigans.Fields: XFaceField, YFaceField, CenterField, TracerFields

function HydrostaticFreeSurfaceTendencyFields(arch, grid, tracer_names)

    u = XFaceField(arch, grid, UVelocityBoundaryConditions(grid))
    v = YFaceField(arch, grid, VVelocityBoundaryConditions(grid))
    η = CenterField(arch, grid, TracerBoundaryConditions(grid))
    tracers = TracerFields(tracer_names, arch, grid)

    return merge((u=u, v=v, η=η), tracers)
end


