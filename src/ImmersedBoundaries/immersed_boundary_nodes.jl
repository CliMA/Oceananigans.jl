using Oceananigans.Grids: Grids, LatitudeLongitudeGrid, OrthogonalSphericalShellGrid, RectilinearGrid,
                          cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z, xspacings, yspacings,
                          zspacings

const c = Center()
const f = Face()

@inline Grids.xnode(i, ibg::IBG, ℓx) = xnode(i, ibg.underlying_grid, ℓx)
@inline Grids.ynode(j, ibg::IBG, ℓy) = ynode(j, ibg.underlying_grid, ℓy)
@inline Grids.znode(k, ibg::IBG, ℓz) = znode(k, ibg.underlying_grid, ℓz)

@inline Grids.λnode(i, ibg::IBG, ℓx) = λnode(i, ibg.underlying_grid, ℓx)
@inline Grids.φnode(j, ibg::IBG, ℓy) = φnode(j, ibg.underlying_grid, ℓy)

@inline Grids.xnode(i, j, ibg::IBG, ℓx, ℓy) = xnode(i, j, ibg.underlying_grid, ℓx, ℓy)

@inline Grids.xnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = xnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline Grids.ynode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ynode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline Grids.znode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = znode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline Grids.λnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = λnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline Grids.φnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = φnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline Grids.ξnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ξnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline Grids.ηnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = ηnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline Grids.rnode(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = rnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)
@inline Grids.rnode(i::AbstractArray, j::AbstractArray, k, ibg::IBG, ℓx, ℓy, ℓz) = rnode(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

@inline Grids.node(i, j, k, ibg::IBG, ℓx, ℓy, ℓz) = node(i, j, k, ibg.underlying_grid, ℓx, ℓy, ℓz)

Grids.nodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = nodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
Grids.nodes(ibg::IBG, (ℓx, ℓy, ℓz); kwargs...) = nodes(ibg, ℓx, ℓy, ℓz; kwargs...)

Grids.xnodes(ibg::IBG, loc; kwargs...) = xnodes(ibg.underlying_grid, loc; kwargs...)
Grids.ynodes(ibg::IBG, loc; kwargs...) = ynodes(ibg.underlying_grid, loc; kwargs...)
Grids.xnodes(ibg::IBG, ::Nothing; kwargs...) = 1:1
Grids.ynodes(ibg::IBG, ::Nothing; kwargs...) = 1:1
Grids.znodes(ibg::IBG, loc; kwargs...) = znodes(ibg.underlying_grid, loc; kwargs...)

Grids.λnodes(ibg::IBG, loc; kwargs...) = λnodes(ibg.underlying_grid, loc; kwargs...)
Grids.φnodes(ibg::IBG, loc; kwargs...) = φnodes(ibg.underlying_grid, loc; kwargs...)

Grids.ξnodes(ibg::IBG, loc; kwargs...) = ξnodes(ibg.underlying_grid, loc; kwargs...)
Grids.ηnodes(ibg::IBG, loc; kwargs...) = ηnodes(ibg.underlying_grid, loc; kwargs...)
Grids.rnodes(ibg::IBG, loc; kwargs...) = rnodes(ibg.underlying_grid, loc; kwargs...)

Grids.xnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = xnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
Grids.ynodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ynodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
Grids.znodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = znodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

Grids.λnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = λnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
Grids.φnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = φnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

Grids.ξnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ξnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
Grids.ηnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = ηnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)
Grids.rnodes(ibg::IBG, ℓx, ℓy, ℓz; kwargs...) = rnodes(ibg.underlying_grid, ℓx, ℓy, ℓz; kwargs...)

@inline Grids.cpu_face_constructor_x(ibg::IBG) = cpu_face_constructor_x(ibg.underlying_grid)
@inline Grids.cpu_face_constructor_y(ibg::IBG) = cpu_face_constructor_y(ibg.underlying_grid)
@inline Grids.cpu_face_constructor_z(ibg::IBG) = cpu_face_constructor_z(ibg.underlying_grid)

Grids.node_names(ibg::IBG, ℓx, ℓy, ℓz) = node_names(ibg.underlying_grid, ℓx, ℓy, ℓz)

Grids.ξname(ibg::IBG) = ξname(ibg.underlying_grid)
Grids.ηname(ibg::IBG) = ηname(ibg.underlying_grid)
Grids.rname(ibg::IBG) = rname(ibg.underlying_grid)

@inline Fields.fractional_x_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_x_index(x, locs, grid.underlying_grid)
@inline Fields.fractional_y_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_y_index(x, locs, grid.underlying_grid)
@inline Fields.fractional_z_index(x, locs, grid::ImmersedBoundaryGrid) = fractional_z_index(x, locs, grid.underlying_grid)

#####
##### Grid-specific grid spacings
#####

const RGIBG{F, X, Y, Z} = ImmersedBoundaryGrid{F, X, Y, Z, <:RectilinearGrid}
const LLIBG{F, X, Y, Z} = ImmersedBoundaryGrid{F, X, Y, Z, <:LatitudeLongitudeGrid}
const OSIBG{F, X, Y, Z} = ImmersedBoundaryGrid{F, X, Y, Z, <:OrthogonalSphericalShellGrid}

@inline Grids.xspacings(grid::RGIBG, ℓx) = xspacings(grid, ℓx, nothing, nothing)
@inline Grids.yspacings(grid::RGIBG, ℓy) = yspacings(grid, nothing, ℓy, nothing)
@inline Grids.zspacings(grid::RGIBG, ℓz) = zspacings(grid, nothing, nothing, ℓz)

@inline Grids.xspacings(grid::LLIBG, ℓx, ℓy) = xspacings(grid, ℓx, ℓy, nothing)
@inline Grids.yspacings(grid::LLIBG, ℓx, ℓy) = yspacings(grid, ℓx, ℓy, nothing)
@inline Grids.zspacings(grid::LLIBG, ℓz)     = zspacings(grid, nothing, nothing, ℓz)

@inline Grids.xspacings(grid::OSIBG, ℓx, ℓy) = xspacings(grid, ℓx, ℓy, nothing)
@inline Grids.yspacings(grid::OSIBG, ℓx, ℓy) = yspacings(grid, ℓx, ℓy, nothing)
@inline Grids.zspacings(grid::OSIBG, ℓz)     = zspacings(grid, nothing, nothing, ℓz)

@inline λspacings(grid::LLIBG, ℓx) = λspacings(grid, ℓx, nothing, nothing)
@inline φspacings(grid::LLIBG, ℓy) = φspacings(grid, nothing, ℓy, nothing)

@inline λspacings(grid::OSIBG, ℓx, ℓy) = λspacings(grid, ℓx, ℓy, nothing)
@inline φspacings(grid::OSIBG, ℓx, ℓy) = φspacings(grid, ℓx, ℓy, nothing)
