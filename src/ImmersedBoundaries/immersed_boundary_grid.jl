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
function ImmersedBoundaryGrid{TX, TY, TZ}(grid::G, ib::I, mi::M, ms::S) where {TX, TY, TZ, G<:AbstractUnderlyingGrid, I, M, S}
    FT = eltype(grid)
    arch = architecture(grid)
    Arch = typeof(arch)
    return ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, Arch}(arch, grid, ib, mi, ms)
end

const CellMaps = Union{AbstractArray, NamedTuple, Tuple}
const ActiveInteriorIBG   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:CellMaps}
const NoActiveInteriorIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}
const ActiveZColumnsIBG   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:CellMaps}
const NoActiveZColumnsIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}

has_active_cells_map(::ActiveInteriorIBG) = true
has_active_z_columns(::ActiveZColumnsIBG) = true
has_active_cells_map(::NoActiveInteriorIBG) = false
has_active_z_columns(::NoActiveZColumnsIBG) = false

"""
    ImmersedBoundaryGrid(grid, ib::AbstractImmersedBoundary;
                         active_cells_map=false, active_z_columns=active_cells_map)

Return a grid with an `AbstractImmersedBoundary` immersed boundary (`ib`). If `active_cells_map` or `active_z_columns` are `true`,
the grid will populate `interior_active_cells` and `active_z_columns` fields -- a list of active indices in the
interior and on a reduced x-y plane, respectively.
"""
function ImmersedBoundaryGrid(grid::AbstractUnderlyingGrid, ib::AbstractImmersedBoundary;
                              active_cells_map::Bool=false,
                              active_z_columns::Bool=active_cells_map)

    materialized_ib = materialize_immersed_boundary(grid, ib)

    # Create the cells map on the CPU, then switch it to the GPU
    interior_active_cells = if active_cells_map
        build_active_cells_map(grid, materialized_ib)
    else
        nothing
    end

    active_z_columns = if active_z_columns
        build_active_z_columns(grid, materialized_ib)
    else
        nothing
    end

    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid,
                                            materialized_ib,
                                            interior_active_cells,
                                            active_z_columns)
end

function with_halo(halo, ibg::ImmersedBoundaryGrid)
    active_cells_map = has_active_cells_map(ibg)
    active_z_columns = has_active_z_columns(ibg)
    underlying_grid = with_halo(halo, ibg.underlying_grid)
    return ImmersedBoundaryGrid(underlying_grid, ibg.immersed_boundary;
                                active_cells_map,
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
