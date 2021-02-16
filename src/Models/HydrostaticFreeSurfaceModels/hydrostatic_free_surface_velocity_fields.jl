using Oceananigans.Grids: Center, Face
using Oceananigans.Fields: XFaceField, YFaceField, ZFaceField
using Oceananigans.BoundaryConditions: UVelocityBoundaryConditions, VVelocityBoundaryConditions, FieldBoundaryConditions

function HydrostaticFreeSurfaceVelocityFields(arch, grid, bcs=NamedTuple())
    u_bcs = :u ∈ keys(bcs) ? bcs.u : UVelocityBoundaryConditions(grid)
    v_bcs = :v ∈ keys(bcs) ? bcs.v : VVelocityBoundaryConditions(grid)
    w_bcs = :w ∈ keys(bcs) ? bcs.w : FreeSurfaceWVelocityBoundaryConditions(grid)

    u = XFaceField(arch, grid, u_bcs)
    v = YFaceField(arch, grid, v_bcs)
    w = ZFaceField(arch, grid, w_bcs)

    return (u=u, v=v, w=w)
end

HydrostaticFreeSurfaceVelocityFields(::Nothing, arch, grid, bcs) =
    HydrostaticFreeSurfaceVelocityFields(arch, grid, bcs)

FreeSurfaceWVelocityBoundaryConditions(grid; user_defined_bcs...) =
    FieldBoundaryConditions(grid, (Center, Center, Face); top=nothing, user_defined_bcs...)
