
const c = Center()
const f = Face()

function build_condition(Topo, side, dim) 
    if Topo == :Bounded 
        return :(($side < 1) | ($side > grid.$dim))
    elseif Topo == :LeftConnected
        return :(($side > grid.$dim))
    else # RightConnected
        return :(($side < 1))
    end
end

#####
##### Exterior node and peripheral node
#####

"""
    inactive_cell(i, j, k, grid)

Return `true` when the tracer cell at `i, j, k` is "external" to the domain boundary.

`inactive_cell`s include halo cells in `Bounded` directions, right halo cells in
`LeftConnected` directions, left halo cells in `RightConnected` directions, and cells
within an immersed boundary. Cells that are staggered with respect to tracer cells and
which lie _on_ the boundary are considered active.
"""
@inline inactive_cell(i, j, k, grid) = false

# We use metaprogramming to handle all the permutations between
# Bounded, LeftConnected, and RightConnected topologies.
# Note that LeftConnected is equivalent to "RightBounded" and
# RightConnected is equivalent to "LeftBounded".
# So LeftConnected and RightConnected are "half Bounded" topologies.
Topos = (:Bounded, :LeftConnected, :RightConnected)

for PrimaryTopo in Topos

    xcondition = build_condition(PrimaryTopo, :i, :Nx)
    ycondition = build_condition(PrimaryTopo, :j, :Ny)
    zcondition = build_condition(PrimaryTopo, :k, :Nz)

    @eval begin
        XBoundedGrid = AbstractGrid{<:Any, <:$PrimaryTopo}
        YBoundedGrid = AbstractGrid{<:Any, <:Any, <:$PrimaryTopo}
        ZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Any, <:$PrimaryTopo}

        @inline inactive_cell(i, j, k, grid::XBoundedGrid) = $xcondition
        @inline inactive_cell(i, j, k, grid::YBoundedGrid) = $ycondition
        @inline inactive_cell(i, j, k, grid::ZBoundedGrid) = $zcondition
    end

    for SecondaryTopo in Topos

        xycondition = :( $xcondition | $(build_condition(SecondaryTopo, :j, :Ny)))
        xzcondition = :( $xcondition | $(build_condition(SecondaryTopo, :k, :Nz)))
        yzcondition = :( $ycondition | $(build_condition(SecondaryTopo, :k, :Nz)))

        @eval begin
            XYBoundedGrid = AbstractGrid{<:Any, <:$PrimaryTopo, <:$SecondaryTopo}
            XZBoundedGrid = AbstractGrid{<:Any, <:$PrimaryTopo, <:Any, <:$SecondaryTopo}
            YZBoundedGrid = AbstractGrid{<:Any, <:Any, <:$PrimaryTopo, <:$SecondaryTopo}

            @inline inactive_cell(i, j, k, grid::XYBoundedGrid) = $xycondition
            @inline inactive_cell(i, j, k, grid::XZBoundedGrid) = $xzcondition
            @inline inactive_cell(i, j, k, grid::YZBoundedGrid) = $yzcondition
        end

        for TeritiaryTopo in Topos
            xyzcondition = :( $xycondition | $(build_condition(TeritiaryTopo, :k, :Nz)))

            @eval begin
                XYZBoundedGrid = AbstractGrid{<:Any, <:$PrimaryTopo, <:$SecondaryTopo, <:$TertiaryTopo}

                @inline inactive_cell(i, j, k, grid::XYZBoundedGrid) = $xyzcondition
            end
        end
    end
end

"""
    inactive_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)` is "inactive" and thus not directly
associated with an "active" cell.

For `Face` locations, this means the node is surrounded by `inactive_cell`s:
the interfaces of "active" cells are _not_ `inactive_node`.

For `Center` locations, this means the direction is `Bounded` and that the
cell or interface centered on the location is completely outside the active
region of the grid.
"""
@inline inactive_node(LX, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)

@inline inactive_node(::Face, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid) & inactive_cell(i-1, j, k, grid)
@inline inactive_node(LX, ::Face, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid) & inactive_cell(i, j-1, k, grid)
@inline inactive_node(LX, LY, ::Face, i, j, k, grid) = inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)

@inline inactive_node(::Face, ::Face, LZ, i, j, k, grid) = inactive_node(c, f, c, i, j, k, grid) & inactive_node(c, f, c, i-1, j, k, grid)
@inline inactive_node(::Face, LY, ::Face, i, j, k, grid) = inactive_node(c, c, f, i, j, k, grid) & inactive_node(c, c, f, i-1, j, k, grid)
@inline inactive_node(LX, ::Face, ::Face, i, j, k, grid) = inactive_node(c, f, c, i, j, k, grid) & inactive_node(c, f, c, i, j, k-1, grid)

@inline inactive_node(::Face, ::Face, ::Face, i, j, k, grid) = inactive_node(c, f, f, i, j, k, grid) & inactive_node(c, f, f, i-1, j, k, grid)

"""
    peripheral_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)`, is _either_ inactive or
lies on the boundary between inactive and active cells in a `Bounded` direction.
"""
@inline peripheral_node(LX, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)

@inline peripheral_node(::Face, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid) | inactive_cell(i-1, j, k, grid)
@inline peripheral_node(LX, ::Face, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid) | inactive_cell(i, j-1, k, grid)
@inline peripheral_node(LX, LY, ::Face, i, j, k, grid) = inactive_cell(i, j, k, grid) | inactive_cell(i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, LZ, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i-1, j, k, grid)
@inline peripheral_node(::Face, LY, ::Face, i, j, k, grid) = peripheral_node(c, c, f, i, j, k, grid) | peripheral_node(c, c, f, i-1, j, k, grid)
@inline peripheral_node(LX, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, f, i, j, k, grid) | peripheral_node(c, f, f, i-1, j, k, grid)

"""
    boundary_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)` lies on a boundary.
"""
@inline boundary_node(LX, LY, LZ, i, j, k, grid) = peripheral_node(LX, LY, LZ, i, j, k, grid) & !inactive_node(LX, LY, LZ, i, j, k, grid)

