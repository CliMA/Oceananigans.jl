"""
    abstract type AbstractImmersedBoundary

Abstract supertype for immersed boundary grids.
"""
abstract type AbstractImmersedBoundary end

struct ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    underlying_grid :: G
    immersed_boundary :: I
    interior_active_cells :: M
    active_z_columns :: S
end

# Internal interface
function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I, mi::M, ms::S) where {TX, TY, TZ, G <: AbstractUnderlyingGrid, I, M, S}
    FT = eltype(grid)
    arch = architecture(grid)
    Arch = typeof(arch)
    return ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, Arch}(arch, grid, ib, mi, ms)
end

"""
    ImmersedBoundaryGrid(grid, ib::AbstractImmersedBoundary; active_cells_map::Bool=true)

Return a grid with an `AbstractImmersedBoundary` immersed boundary (`ib`). If `active_cells_map` is `true`,
the grid will also populate an `interior_active_cells` and `active_z_columns` fields that are a list of active indices in the 
interior and on a reduced x-y plane, respectively.
"""
function ImmersedBoundaryGrid(underlying_grid::AbstractUnderlyingGrid, ib::AbstractImmersedBoundary; active_cells_map::Bool=true) 
    immersed_boundary = numerical_immersed_boundary(grid, ib)
    
    # Create the cells map on the CPU, then switch it to the GPU
    if active_cells_map 
        interior_active_cells = map_interior_active_cells(ibg)
        active_z_columns = map_active_z_columns(ibg)
    else
        interior_active_cells = nothing
        active_z_columns = nothing
    end
    
    TX, TY, TZ = topology(grid)
    
    return ImmersedBoundaryGrid{TX, TY, TZ}(underlying_grid, 
                                            immersed_boundary, 
                                            interior_active_cells,
                                            active_z_columns)
end


const IBG = ImmersedBoundaryGrid

@inline Base.getproperty(ibg::IBG, property::Symbol) = get_ibg_property(ibg, Val(property))
@inline get_ibg_property(ibg::IBG, ::Val{property}) where property = getfield(getfield(ibg, :underlying_grid), property)
@inline get_ibg_property(ibg::IBG, ::Val{:immersed_boundary})      = getfield(ibg, :immersed_boundary)
@inline get_ibg_property(ibg::IBG, ::Val{:underlying_grid})        = getfield(ibg, :underlying_grid)
@inline get_ibg_property(ibg::IBG, ::Val{:interior_active_cells})  = getfield(ibg, :interior_active_cells)
@inline get_ibg_property(ibg::IBG, ::Val{:active_z_columns})       = getfield(ibg, :active_z_columns)

@inline architecture(ibg::IBG) = architecture(ibg.underlying_grid)

@inline x_domain(ibg::IBG) = x_domain(ibg.underlying_grid)
@inline y_domain(ibg::IBG) = y_domain(ibg.underlying_grid)
@inline z_domain(ibg::IBG) = z_domain(ibg.underlying_grid)

Adapt.adapt_structure(to, ibg::IBG{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    ImmersedBoundaryGrid{TX, TY, TZ}(adapt(to, ibg.underlying_grid),
                                     adapt(to, ibg.immersed_boundary),
                                     nothing,
                                     nothing)

with_halo(halo, ibg::ImmersedBoundaryGrid) =
    ImmersedBoundaryGrid(with_halo(halo, ibg.underlying_grid), ibg.immersed_boundary)

# ImmersedBoundaryGrids require an extra halo point to check the "inactivity" of a `Face` node at N + H
# (which requires checking `Center` nodes at N + H and N + H + 1)
inflate_halo_size_one_dimension(req_H, old_H, _, ::IBG)            = max(req_H + 1, old_H)
inflate_halo_size_one_dimension(req_H, old_H, ::Type{Flat}, ::IBG) = 0

# Defining the bottom
@inline z_bottom(i, j, grid) = znode(i, j, 1, grid, c, c, f)
@inline z_bottom(i, j, ibg::IBG) = error("The function `bottom` has not been defined for $(summary(ibg))!")

function Base.summary(grid::ImmersedBoundaryGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)

    return string(size_summary(size(grid)),
                  " ImmersedBoundaryGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function show(io::IO, ibg::ImmersedBoundaryGrid)
    print(io, summary(ibg), ":", "\n",
             "├── immersed_boundary: ", summary(ibg.immersed_boundary), "\n",
             "├── underlying_grid: ", summary(ibg.underlying_grid), "\n")

    return show(io, ibg.underlying_grid, false)
end

@inline Base.zero(ibg::IBG) = zero(ibg.underlying_grid)

function on_architecture(arch, ibg::IBG)
    underlying_grid   = on_architecture(arch, ibg.underlying_grid)
    immersed_boundary = on_architecture(arch, ibg.immersed_boundary)
    return ImmersedBoundaryGrid(underlying_grid, immersed_boundary)
end

isrectilinear(ibg::IBG) = isrectilinear(ibg.underlying_grid)
