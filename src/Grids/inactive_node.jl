#####
##### "Inactive node" logic for Bounded topology and ImmersedBoundaryGrid
#####
##### * inactive_cell
##### * inactive_node
##### * peripheral_node
##### * x_boundary
##### * y_boundary
##### * z_boundary
#####

const c = Center()
const f = Face()

function build_condition(Topo, side, dim) 
    if Topo == :Bounded 
        return :(($side < 1) | ($side > grid.$dim))
    elseif Topo == :LeftConnected
        return :(($side > grid.$dim))
    else 
        return :(($side < 1))
    end
end

Topos = [:Bounded, :LeftConnected, :RightConnected] 

#####
##### Exterior node and peripheral node
#####

"""
    inactive_cell(i, j, k, grid)

Return `true` when the tracer cell at `i, j, k` is lies outside the "active domain" of
the grid in `Bounded` directions. Otherwise, return `false`.
"""
@inline inactive_cell(i, j, k, grid) = false

for Topo in Topos

    xcondition = build_condition(Topo, :i, :Nx)
    ycondition = build_condition(Topo, :j, :Ny)
    zcondition = build_condition(Topo, :k, :Nz)

    @eval begin
        XBoundedGrid = AbstractGrid{<:Any, <:$Topo}
        YBoundedGrid = AbstractGrid{<:Any, <:Any, <:$Topo}
        ZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Any, <:$Topo}

        @inline inactive_cell(i, j, k, grid::XBoundedGrid) = ifelse($xcondition, true, false)
        @inline inactive_cell(i, j, k, grid::YBoundedGrid) = ifelse($ycondition, true, false)
        @inline inactive_cell(i, j, k, grid::ZBoundedGrid) = ifelse($zcondition, true, false)
    end
    for OtherTopo in Topos
        xycondition = :( $xcondition | $(build_condition(OtherTopo, :j, :Ny)))
        xzcondition = :( $xcondition | $(build_condition(OtherTopo, :k, :Nz)))
        yzcondition = :( $ycondition | $(build_condition(OtherTopo, :k, :Nz)))

        @eval begin
            XYBoundedGrid = AbstractGrid{<:Any, <:$Topo, <:$OtherTopo}
            XZBoundedGrid = AbstractGrid{<:Any, <:$Topo, <:Any, <:$OtherTopo}
            YZBoundedGrid = AbstractGrid{<:Any, <:Any, <:$Topo, <:$OtherTopo}

            @inline inactive_cell(i, j, k, grid::XYBoundedGrid) = ifelse($xycondition, true, false)
            @inline inactive_cell(i, j, k, grid::XZBoundedGrid) = ifelse($xzcondition, true, false)
            @inline inactive_cell(i, j, k, grid::YZBoundedGrid) = ifelse($yzcondition, true, false)
        end
        for LastTopo in Topos
            xyzcondition = :( $xycondition | $(build_condition(LastTopo, :k, :Nz)))

            @eval begin
                XYZBoundedGrid = AbstractGrid{<:Any, <:$Topo, <:$OtherTopo, <:$LastTopo}
               
                @inline inactive_cell(i, j, k, grid::XYZBoundedGrid) = ifelse($xyzcondition, true, false)
            end
        end
    end
end

"""
    inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

Return `true` when the location `(ℓx, ℓy, ℓz)` is "inactive" and thus not directly
associated with an "active" cell.

For `Face` locations, this means the node is surrounded by `inactive_cell`s:
the interfaces of "active" cells are _not_ `inactive_node`.

For `Center` locations, this means the direction is `Bounded` and that the
cell or interface centered on the location is completely outside the active
region of the grid.
"""
@inline inactive_node(i, j, k, grid, ℓx, ℓy, ℓz) = inactive_cell(i, j, k, grid)

@inline inactive_node(i, j, k, grid, ::Face, ℓy, ℓz) = inactive_cell(i, j, k, grid) & inactive_cell(i-1, j, k, grid)
@inline inactive_node(i, j, k, grid, ℓx, ::Face, ℓz) = inactive_cell(i, j, k, grid) & inactive_cell(i, j-1, k, grid)
@inline inactive_node(i, j, k, grid, ℓx, ℓy, ::Face) = inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)

@inline inactive_node(i, j, k, grid, ::Face, ::Face, ℓz) = inactive_node(i, j, k, grid, c, f, c) & inactive_node(i-1, j, k, grid, c, f, c)
@inline inactive_node(i, j, k, grid, ::Face, ℓy, ::Face) = inactive_node(i, j, k, grid, c, c, f) & inactive_node(i-1, j, k, grid, c, c, f)
@inline inactive_node(i, j, k, grid, ℓx, ::Face, ::Face) = inactive_node(i, j, k, grid, c, f, c) & inactive_node(i, j, k-1, grid, c, f, c)

@inline inactive_node(i, j, k, grid, ::Face, ::Face, ::Face) = inactive_node(i, j, k, grid, c, f, f) & inactive_node(i-1, j, k, grid, c, f, f)

#####
##### Peripheral node
#####

"""
    peripheral_node(ℓx, ℓy, ℓz, i, j, k, grid)

Return `true` when the location `(ℓx, ℓy, ℓz)`, is _either_ inactive or
lies on the boundary between inactive and active cells in a `Bounded` direction.
"""
@inline peripheral_node(ℓx, ℓy, ℓz, i, j, k, grid) = inactive_cell(i, j, k, grid)

@inline peripheral_node(i, j, k, grid, ::Face, ℓy, ℓz) = inactive_cell(i, j, k, grid) | inactive_cell(i-1, j, k, grid)
@inline peripheral_node(i, j, k, grid, ℓx, ::Face, ℓz) = inactive_cell(i, j, k, grid) | inactive_cell(i, j-1, k, grid)
@inline peripheral_node(i, j, k, grid, ℓx, ℓy, ::Face) = inactive_cell(i, j, k, grid) | inactive_cell(i, j, k-1, grid)

@inline peripheral_node(i, j, k, grid, ::Face, ::Face, ℓz) = peripheral_node(i, j, k, grid, c, f, c) | peripheral_node(i-1, j, k, grid, c, f, c)
@inline peripheral_node(i, j, k, grid, ::Face, ℓy, ::Face) = peripheral_node(i, j, k, grid, c, c, f) | peripheral_node(i-1, j, k, grid, c, c, f)
@inline peripheral_node(i, j, k, grid, ℓx, ::Face, ::Face) = peripheral_node(i, j, k, grid, c, f, c) | peripheral_node(i, j, k-1, grid, c, f, c)

@inline peripheral_node(i, j, k, grid, ::Face, ::Face, ::Face) = peripheral_node(i, j, k, grid, c, f, f) | peripheral_node(i-1, j, k, grid, c, f, f)

#####
##### x_, y_, z_boundary node
#####
##### |xxxxx|     |
##### |xxxxx|  ∘  | 
##### |xxxxx|     |
#####
##### `x_`, `y_`, and `z_boundary` identify whether a node on either side of the specified node,
##### in the `x`-, `y`-, or `z`-direction, are inactive. This function is used to control the behavior
##### of differencing and interpolation:
#####
##### 1. Differences across `x_`, `y_`, `z_boundary` are zero
##### 2. Interpolation across `x_`, `y_`, `z_boundary` returns values from _active_ node only.
#####
##### With these rules, cells that lie outside `Bounded` are never "touched".
#####

# Indexing conventions
@inline idxᴿ(i, ::Face)   = i
@inline idxᴸ(i, ::Face)   = i-1
@inline idxᴿ(i, ::Center) = i+1
@inline idxᴸ(i, ::Center) = i

@inline flip(::Face)   = c
@inline flip(::Center) = f

"""
    x_boundary(i, j, k, grid, ℓx, ℓy, ℓz)

Return true if one or both of the nodes to the east or west of the location `(ℓx, ℓy, ℓz)`
at `i, j, k` in `x` has `inactive_node`.
"""
@inline x_boundary(i, j, k, grid, ℓx, ℓy, ℓz) = inactive_node(idxᴸ(i, ℓx), j, k, grid, flip(ℓx), ℓy, ℓz) | inactive_node(idxᴿ(i, ℓx), j, k, grid, flip(ℓx), ℓy, ℓz)
@inline y_boundary(i, j, k, grid, ℓx, ℓy, ℓz) = inactive_node(i, idxᴸ(j, ℓy), k, grid, ℓx, flip(ℓy), ℓz) | inactive_node(i, idxᴿ(j, ℓy), k, grid, ℓx, flip(ℓy), ℓz)
@inline z_boundary(i, j, k, grid, ℓx, ℓy, ℓz) = inactive_node(i, j, idxᴸ(k, ℓz), grid, ℓx, ℓy, flip(ℓz)) | inactive_node(i, j, idxᴿ(k, ℓz), grid, ℓx, ℓy, flip(ℓz))

