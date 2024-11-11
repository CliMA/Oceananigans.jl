import Oceananigans.Grids: xspacings, yspacings, zspacings

const c = Center()
const f = Face()

@inline Base.zero(ibg::IBG) = zero(ibg.underlying_grid)

@inline xnode(i, ibg::IBG, ℓx) = xnode(i, ibg.underlying_grid, ℓx)
@inline ynode(j, ibg::IBG, ℓy) = ynode(j, ibg.underlying_grid, ℓy)
@inline znode(k, ibg::IBG, ℓz) = znode(k, ibg.underlying_grid, ℓz)

@inline λnode(i, ibg::IBG, ℓx) = λnode(i, ibg.underlying_grid, ℓx)
@inline φnode(j, ibg::IBG, ℓy) = φnode(i, ibg.underlying_grid, ℓy)

@inline xnode(i, j, ibg::IBG, ℓx, ℓy) = xnode(i, j, ibg.underlying_grid, ℓx, ℓy)

@inline xnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = xnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline ynode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ynode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline znode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = znode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline λnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = λnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline φnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = φnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline ξnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ξnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline ηnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ηnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline rnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = rnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline node(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = node(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

nodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = nodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
nodes(ibg::IBG, (ℓx, ℓy, ℓz); kwargs...) = nodes(ibg, ℓx, ℓy, ℓz; kwargs...)

xnodes(ibg::IBG, loc; kwargs...) = xnodes(ibg.underlying_grid, loc; kwargs...)
ynodes(ibg::IBG, loc; kwargs...) = ynodes(ibg.underlying_grid, loc; kwargs...)
znodes(ibg::IBG, loc; kwargs...) = znodes(ibg.underlying_grid, loc; kwargs...)

λnodes(ibg::IBG, loc; kwargs...) = λnodes(ibg.underlying_grid, loc; kwargs...)
φnodes(ibg::IBG, loc; kwargs...) = φnodes(ibg.underlying_grid, loc; kwargs...)

ξnodes(ibg::IBG, loc; kwargs...) = ξnodes(ibg.underlying_grid, loc; kwargs...)
ηnodes(ibg::IBG, loc; kwargs...) = ηnodes(ibg.underlying_grid, loc; kwargs...)
rnodes(ibg::IBG, loc; kwargs...) = rnodes(ibg.underlying_grid, loc; kwargs...)

xnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = xnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
ynodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ynodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
znodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = znodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

λnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = λnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
φnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = φnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

ξnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ξnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
ηnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ηnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
rnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = rnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

@inline cpu_face_constructor_x(ibg::IBG) = cpu_face_constructor_x(ibg.underlying_grid)
@inline cpu_face_constructor_y(ibg::IBG) = cpu_face_constructor_y(ibg.underlying_grid)
@inline cpu_face_constructor_z(ibg::IBG) = cpu_face_constructor_z(ibg.underlying_grid)

node_names(ibg::IBG, ℓx, ℓy, ℓz) = node_names(ibg.underlying_grid, ℓx, ℓy, ℓz)
ξname(ibg::IBG) = ξname(ibg.underlying_grid)
ηname(ibg::IBG) = ηname(ibg.underlying_grid)
rname(ibg::IBG) = rname(ibg.underlying_grid)

function on_architecture(arch, ibg::IBG)
    underlying_grid   = on_architecture(arch, ibg.underlying_grid)
    immersed_boundary = on_architecture(arch, ibg.immersed_boundary)
    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
end

isrectilinear(ibg::IBG) = isrectilinear(ibg.underlying_grid)

@inline fractional_x_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_x_index(x, locs, grid.underlying_grid)
@inline fractional_y_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_y_index(x, locs, grid.underlying_grid)
@inline fractional_z_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_z_index(x, locs, grid.underlying_grid)

xspacings(ibg::ImmersedBoundaryGrid, ℓ...) = xspacings(ibg.underlying_grid, ℓ...)
yspacings(ibg::ImmersedBoundaryGrid, ℓ...) = yspacings(ibg.underlying_grid, ℓ...)
zspacings(ibg::ImmersedBoundaryGrid, ℓ...) = zspacings(ibg.underlying_grid, ℓ...)
