"""
    ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ, R, A} <: AbstractRectilinearGrid{FT, TX, TY, TZ}
A rectilinear grid with with constant grid spacings `Δy` and `Δz`, and
non-uniform or stretched zonal grid spacing `Δx` between cell centers and cell faces,
topology `{TX, TY, TZ}`, and coordinate ranges of type `R` (where a range can be used) and
`A` (where an array is needed).
"""
struct ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ, A, S, R} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

    # Number of grid points in (x,y,z).
    Nx :: Int
    Ny :: Int
    Nz :: Int

    # Halo size in (x,y,z).
    Hx :: Int
    Hy :: Int
    Hz :: Int

    # Domain size [m].
    Lx :: FT
    Ly :: FT
    Lz :: FT

   # Range of coordinates at the centers of the cells.
    xᶜ :: A
    yᶜ :: S
    zᶜ :: R

    # Range of grid coordinates at the faces of the cells.
    xᶠ :: A
    yᶠ :: S
    zᶠ :: R

    # Grid spacing [m].
    Δxᶜ :: A
    Δxᶠ :: A
    Δyᶜ :: FT
    Δzᶜ :: FT 
end

generate_regular_grid(FT,   ::Type{Flat}, Lx, x, N, H) = 0, 0, 1, 1

function generate_regular_grid(FT, topo, Lx, x, N, H)

    # uniform spacing
    Δx = Lx / N

   # Total length of xᶠ and xᶜ
   Txᶠ = total_length(Face,   topo, N, H) 
   Txᶜ = total_length(Center, topo, N, H)

   # Halo locations on left and right
   xᶠ₋ = x[1] - H * Δx
   xᶠ₊ = xᶠ₋  + total_extent(topo, H, Δx, Lx)
   xᶜ₋ = xᶠ₋ + Δx / 2
   xᶜ₊ = xᶜ₋ + Lx + Δx * (2 * H - 1)

    # Construct locations of face and cell centers 
    xᶠ = OffsetArray(collect(range(xᶠ₋, xᶠ₊; length = Txᶠ)), -H)
    xᶜ = OffsetArray(collect(range(xᶜ₋, xᶜ₊; length = Txᶜ)), -H)

   return xᶜ, xᶠ, Δx, Δx
end

# FJP: How to deal with periodic domains? Depends on how halos are extended
generate_stretched_grid(FT, ::Type{Flat}, Lx, x, N, H) = 0, 0, 1, 1

function generate_stretched_grid(FT, topo, Lx, x, N, H)

    # Total length of xᶠ and xᶜ
    Txᶠ = total_length(Face,   topo, N, H) 
    Txᶜ = total_length(Center, topo, N, H)

    interior_xᶠ = x[1] .+ Lx*collect(range(0, 1, length=N+1)).^2        # assume quadratic spacing for now

    # Find withs near boundaries
    Δxᶠ₋ = interior_xᶠ[2]   - interior_xᶠ[1]
    Δxᶠ₊ = interior_xᶠ[end] - interior_xᶠ[end-1]

    # Halos have constant widths
    xᶠ₋ = [x[1]   - Δxᶠ₋ * sum(i) for i=1:H]
    xᶠ₊ = [x[end] + Δxᶠ₊ * sum(i) for i=1:H]

    # Build shifted grids
     xᶠ = OffsetArray(vcat(xᶠ₋, interior_xᶠ, xᶠ₊),                       -H)
     xᶜ = OffsetArray([(xᶠ[i + 1] + xᶠ[i]) / 2 for i = (1-H):(Txᶜ-H)],   -H)
    Δxᶜ = OffsetArray([xᶜ[i]     - xᶜ[i - 1]  for i = (2-H):(Txᶜ-H)],   -H)
    Δxᶠ = OffsetArray([xᶠ[i+1]   - xᶠ[i]      for i = (1-H):(Txᶠ-1-H)], -H)

   return xᶜ, xᶠ, Δxᶜ, Δxᶠ
end

function ZonallyStretchedRectilinearGrid(FT=Float64; 
                                            size, 
                                   architecture = CPU(),
                                              x = nothing, 
                                              y = nothing, 
                                              z = nothing,
                                         extent = nothing,
                                       topology = (Bounded, Periodic, Flat),
                                           halo = nothing
                                       )

             TX, TY, TZ = validate_topology(topology)
                   size = validate_size(TX, TY, TZ, size)
                   halo = validate_halo(TX, TY, TZ, halo)
    Lx, Ly, Lz, x, y, z = validate_zonally_stretched_grid(TX, TY, TZ, FT, extent, x, y, z)

             Nx, Ny, Nz = size
             Hx, Hy, Hz = halo
                      L = (Lx, Ly, Lz)

       xᶜ, xᶠ, Δxᶜ, Δxᶠ = generate_stretched_grid(FT, topology[1], Lx, x, Nx, Hx)
       yᶜ, yᶠ, Δyᶜ, Δyᶠ = generate_regular_grid(FT,   topology[2], Ly, y, Ny, Hy)
       zᶜ, zᶠ, Δzᶜ, Δzᶠ = generate_regular_grid(FT,   topology[3], Lz, z, Nz, Hz)

       return ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ, typeof(xᶜ), typeof(yᶜ), typeof(zᶜ)}(
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, xᶜ, yᶜ, zᶜ, xᶠ, yᶠ, zᶠ, Δxᶜ, Δxᶠ, Δyᶜ, Δzᶜ)
end

#####
##### Vertically stretched grid utilities
#####

function Base.show(io::IO, grid::ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "ZonallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}\n",
              "                    domain: (Lx=$(grid.Lx),     Ly=$(grid.Ly),   Lz=$(grid.Lz))\n",
              "                resolution: (Nx=$(grid.Nx),     Ny=$(grid.Ny),   Nz=$(grid.Nz))\n",
              "                 halo size: (Hx=$(grid.Hx),     Hy=$(grid.Hy),   Hz=$(grid.Hz))\n",
              "              grid spacing: (Δx=$(grid.Δxᶜ[1]),  Δy=$(grid.Δyᶜ), Δz=$(grid.Δzᶜ))\n")
end

short_show(grid::ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "ZonallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

#####
##### Should merge with grid_utils.jl at some point
#####

@inline xnode(::Type{Center}, i, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.xᶜ[i]
@inline xnode(::Type{Face},   i, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.xᶠ[i]
@inline ynode(::Type{Center}, j, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.yᶜ[j]
@inline ynode(::Type{Face},   j, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.yᶠ[j]
@inline znode(::Type{Center}, k, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.zᶜ[k]
@inline znode(::Type{Face},   k, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.zᶠ[k]
all_x_nodes(::Type{Center},      grid::ZonallyStretchedRectilinearGrid) = grid.xᶜ
all_x_nodes(::Type{Face},        grid::ZonallyStretchedRectilinearGrid) = grid.xᶠ
all_y_nodes(::Type{Center},      grid::ZonallyStretchedRectilinearGrid) = grid.yᶜ
all_y_nodes(::Type{Face},        grid::ZonallyStretchedRectilinearGrid) = grid.yᶠ
all_z_nodes(::Type{Center},      grid::ZonallyStretchedRectilinearGrid) = grid.zᶜ
all_z_nodes(::Type{Face},        grid::ZonallyStretchedRectilinearGrid) = grid.zᶠ
#
# Get minima of grid
#
#min_Δx(grid::ZonallyStretchedRectilinearGrid) = minimum(view(grid.Δxᶜ, 1:grid.Nx))
#min_Δy(grid::ZonallyStretchedRectilinearGrid) = grid.Δy
#min_Δz(grid::ZonallyStretchedRectilinearGrid) = grid.Δz
