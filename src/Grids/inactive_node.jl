
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

@inline inactive_node(::Face, ::Center, ::Center, i, j, k, grid) = inactive_cell(i, j, k, grid) & inactive_cell(i-1, j, k, grid)
@inline inactive_node(::Center, ::Face, ::Center, i, j, k, grid) = inactive_cell(i, j, k, grid) & inactive_cell(i, j-1, k, grid)
@inline inactive_node(::Center, ::Center, ::Face, i, j, k, grid) = inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)

@inline inactive_node(::Face, ::Face, ::Center, i, j, k, grid) = inactive_node(c, f, c, i, j, k, grid) & inactive_node(c, f, c, i-1, j, k, grid)
@inline inactive_node(::Face, ::Center, ::Face, i, j, k, grid) = inactive_node(c, c, f, i, j, k, grid) & inactive_node(c, c, f, i-1, j, k, grid)
@inline inactive_node(::Center, ::Face, ::Face, i, j, k, grid) = inactive_node(c, f, c, i, j, k, grid) & inactive_node(c, f, c, i, j, k-1, grid)

@inline inactive_node(::Face, ::Face, ::Face, i, j, k, grid) = inactive_node(c, f, f, i, j, k, grid) & inactive_node(c, f, f, i-1, j, k, grid)

@inline inactive_node(::Flat, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)
@inline inactive_node(LX, ::Flat, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)
@inline inactive_node(LX, LY, ::Flat, i, j, k, grid) = inactive_cell(i, j, k, grid)

"""
    peripheral_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)`, is _either_ inactive or
lies on the boundary between inactive and active cells in a `Bounded` direction.
"""
@inline peripheral_node(LX, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)

@inline peripheral_node(::Face, ::Center, ::Center, i, j, k, grid) = inactive_cell(i, j, k, grid) | inactive_cell(i-1, j, k, grid)
@inline peripheral_node(::Center, ::Face, ::Center, i, j, k, grid) = inactive_cell(i, j, k, grid) | inactive_cell(i, j-1, k, grid)
@inline peripheral_node(::Center, ::Center, ::Face, i, j, k, grid) = inactive_cell(i, j, k, grid) | inactive_cell(i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, ::Center, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i-1, j, k, grid)
@inline peripheral_node(::Face, ::Center, ::Face, i, j, k, grid) = peripheral_node(c, c, f, i, j, k, grid) | peripheral_node(c, c, f, i-1, j, k, grid)
@inline peripheral_node(::Center, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, f, i, j, k, grid) | peripheral_node(c, f, f, i-1, j, k, grid)

@inline peripheral_node(::Flat, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)
@inline peripheral_node(LX, ::Flat, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)
@inline peripheral_node(LX, LY, ::Flat, i, j, k, grid) = inactive_cell(i, j, k, grid)

"""
    boundary_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)` lies on a boundary.
"""
@inline boundary_node(LX, LY, LZ, i, j, k, grid) = peripheral_node(LX, LY, LZ, i, j, k, grid) & !inactive_node(LX, LY, LZ, i, j, k, grid)

