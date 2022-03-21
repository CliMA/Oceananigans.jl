
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

for Topo in Topos

    xcondition = build_condition(Topo, :i, :Nx)
    ycondition = build_condition(Topo, :j, :Ny)
    zcondition = build_condition(Topo, :k, :Nz)

    @eval begin
        XBoundedGrid = AbstractGrid{<:Any, <:$Topo}
        YBoundedGrid = AbstractGrid{<:Any, <:Any, <:$Topo}
        ZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Any, <:$Topo}

        @inline solid_node(i, j, k, grid::XBoundedGrid) = ifelse($xcondition, true, false)
        @inline solid_node(i, j, k, grid::YBoundedGrid) = ifelse($ycondition, true, false)
        @inline solid_node(i, j, k, grid::ZBoundedGrid) = ifelse($zcondition, true, false)
    end
    for OtherTopo in Topos
        xycondition = :( $xcondition | $(build_condition(Topo, :j, :Ny)))
        xzcondition = :( $xcondition | $(build_condition(Topo, :k, :Nz)))
        yzcondition = :( $ycondition | $(build_condition(Topo, :k, :Nz)))

        @eval begin
            XYBoundedGrid = AbstractGrid{<:Any, <:$Topo, <:$OtherTopo}
            XZBoundedGrid = AbstractGrid{<:Any, <:Any, <:$Topo, <:Any, <:$OtherTopo}
            YZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Any, <:$Topo, <:$OtherTopo}

            @inline solid_node(i, j, k, grid::XYBoundedGrid) = ifelse($xycondition, true, false)
            @inline solid_node(i, j, k, grid::XZBoundedGrid) = ifelse($xzcondition, true, false)
            @inline solid_node(i, j, k, grid::YZBoundedGrid) = ifelse($yzcondition, true, false)
        end
        for LastTopo in Topos
            xyzcondition = :( $xycondition | $(build_condition(Topo, :k, :Nz)))

            @eval begin
                XYZBoundedGrid = AbstractGrid{<:Any, <:$Topo, <:$OtherTopo, <:$LastTopo}
               
                @inline solid_node(i, j, k, grid::XYZBoundedGrid) = ifelse($xyzcondition, true, false)
            end
        end
    end
end

# Fallback for general grids
@inline solid_node(i, j, k, grid) = false

@inline solid_node(LX, LY, LZ, i, j, k, grid)      = solid_node(i, j, k, grid)
@inline solid_interface(LX, LY, LZ, i, j, k, grid) = solid_node(i, j, k, grid)

@inline solid_node(::Face, LY, LZ, i, j, k, grid) = solid_node(i, j, k, grid) & solid_node(i-1, j, k, grid)
@inline solid_node(LX, ::Face, LZ, i, j, k, grid) = solid_node(i, j, k, grid) & solid_node(i, j-1, k, grid)
@inline solid_node(LX, LY, ::Face, i, j, k, grid) = solid_node(i, j, k, grid) & solid_node(i, j, k-1, grid)

@inline solid_node(::Face, ::Face, LZ, i, j, k, grid) = solid_node(c, f, c, i, j, k, grid) & solid_node(c, f, c, i-1, j, k, grid)
@inline solid_node(::Face, LY, ::Face, i, j, k, grid) = solid_node(c, c, f, i, j, k, grid) & solid_node(c, c, f, i-1, j, k, grid)
@inline solid_node(LX, ::Face, ::Face, i, j, k, grid) = solid_node(c, f, c, i, j, k, grid) & solid_node(c, f, c, i, j, k-1, grid)

@inline solid_node(::Face, ::Face, ::Face, i, j, k, grid) = solid_node(c, f, f, i, j, k, grid) & solid_node(c, f, f, i-1, j, k, grid)

@inline solid_interface(::Face, LY, LZ, i, j, k, grid) = solid_node(i, j, k, grid) | solid_node(i-1, j, k, grid)
@inline solid_interface(LX, ::Face, LZ, i, j, k, grid) = solid_node(i, j, k, grid) | solid_node(i, j-1, k, grid)
@inline solid_interface(LX, LY, ::Face, i, j, k, grid) = solid_node(i, j, k, grid) | solid_node(i, j, k-1, grid)

@inline solid_interface(::Face, ::Face, LZ, i, j, k, grid) = solid_interface(c, f, c, i, j, k, grid) | solid_interface(c, f, c, i-1, j, k, grid)
@inline solid_interface(::Face, LY, ::Face, i, j, k, grid) = solid_interface(c, c, f, i, j, k, grid) | solid_interface(c, c, f, i-1, j, k, grid)
@inline solid_interface(LX, ::Face, ::Face, i, j, k, grid) = solid_interface(c, f, c, i, j, k, grid) | solid_interface(c, f, c, i, j, k-1, grid)

@inline solid_interface(::Face, ::Face, ::Face, i, j, k, grid) = solid_interface(c, f, f, i, j, k, grid) | solid_interface(c, f, f, i-1, j, k, grid)
