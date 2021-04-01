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

    Δx = Lx / N

   TFx = total_length(Face,   topo, N, H) 
   TCx = total_length(Center, topo, N, H)

   xF₋ = x[1] - H * Δx
   xF₊ = xF₋  + total_extent(topo, H, Δx, Lx)

   xC₋ = xF₋ + Δx / 2
   xC₊ = xC₋ + Lx + Δx * (2 * H - 1)

    xᶠ = collect(range(xF₋, xF₊; length = TFx))  
    xᶜ = collect(range(xC₋, xC₊; length = TCx))

    xᶠ = OffsetArray(xᶠ, -H)
    xᶜ = OffsetArray(xᶜ, -H)

   return xᶜ, xᶠ, Δx, Δx
end

# FJP: should not use periodic in a stretched direction!!!!
generate_stretched_grid(FT, ::Type{Flat}, Lx, x, N, H) = 0, 0, 1, 1

function generate_stretched_grid(FT, topo, Lx, x, N, H)

    TFx = total_length(Face,   topo, N, H) 
    TCx = total_length(Center, topo, N, H)

    interior_xᶠ = x[1] .+ Lx*collect(0:1/N:1).^2        # assume quadratic spacing for now

    # Find withs near boundaries
    ΔxF₋ = interior_xᶠ[2]   - interior_xᶠ[1]
    ΔxF₊ = interior_xᶠ[end] - interior_xᶠ[end-1]

    # Build halos of constant width and cell faces
    xF₋ = [x[1]   - ΔxF₋*sum(i) for i=1:H]
    xF₊ = [x[end] + ΔxF₊*sum(i) for i=1:H]
     xᶠ = vcat(xF₋, interior_xᶠ, xF₊)
    Δxᶠ = [  xᶠ[i+1] - xᶠ[i]        for i = 1:TFx-1 ]

    # Build cell centers, cell spacings 
      xᶜ = [ (xᶠ[i + 1] + xᶠ[i]) / 2 for i = 1:TCx   ]
     Δxᶜ = [  xᶜ[i] - xᶜ[i - 1]      for i = 2:TCx   ]

      xᶠ = OffsetArray(xᶠ,  -H)
      xᶜ = OffsetArray(xᶜ,  -H)
     Δxᶜ = OffsetArray(Δxᶜ, -H)
     Δxᶠ = OffsetArray(Δxᶠ, -H)

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

       return ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ, typeof(xᶜ), typeof(yᶠ), typeof(zᶠ)}(
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, xᶜ, yᶜ, zᶜ, xᶠ, yᶠ, zᶠ, Δxᶜ, Δxᶠ, Δyᶜ, Δzᶜ)
end

#####
##### Vertically stretched grid utilities
#####

#=
short_show(grid::ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "ZonallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"
function show(io::IO, g::ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    Δx_min = minimum(view(g.Δxᶜ, 1:g.Nx))
    Δx_max = maximum(view(g.Δxᶜ, 1:g.Nx))
    print(io, "ZonallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz): , [min=", Δx_min, ", max=", Δx_max,"])", g.Δyᶜ, ", ", g.Δzᶜ,)
end
=#

#=
Adapt.adapt_structure(to, grid::ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ, typeof(Adapt.adapt(to, grid.xᶠᵃᵃ)), typeof(grid.zᵃᵃᶠ)}(
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        Adapt.adapt(to, grid.Δxᶜᵃᵃ),
        Adapt.adapt(to, grid.Δxᶠᵃᵃ),
        grid.Δy, grid.Δz,
        Adapt.adapt(to, grid.xᶜᵃᵃ),
        Adapt.adapt(to, grid.xᶠᵃᵃ),
        grid.yᵃᶜᵃ, grid.zᵃᵃᶜ,
        grid.yᵃᶠᵃ, grid.zᵃᵃᶠ)
=#

#####
##### Should merge with grid_utils.jl at some point
#####

#=
@inline xnode(::Type{Center}, i, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.xᶜᵃᵃ[i]
@inline xnode(::Type{Face},   i, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.xᶠᵃᵃ[i]
@inline ynode(::Type{Center}, j, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.yᵃᶜᵃ[j]
@inline ynode(::Type{Face},   j, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.yᵃᶠᵃ[j]
@inline znode(::Type{Center}, k, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(::Type{Face},   k, grid::ZonallyStretchedRectilinearGrid) = @inbounds grid.zᵃᵃᶠ[k]
all_x_nodes(::Type{Center}, grid::ZonallyStretchedRectilinearGrid) = grid.xᶜᵃᵃ
all_x_nodes(::Type{Face},   grid::ZonallyStretchedRectilinearGrid) = grid.xᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::ZonallyStretchedRectilinearGrid) = grid.yᵃᶜᵃ
all_y_nodes(::Type{Face},   grid::ZonallyStretchedRectilinearGrid) = grid.yᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::ZonallyStretchedRectilinearGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face},   grid::ZonallyStretchedRectilinearGrid) = grid.zᵃᵃᶠ
#
# Get minima of grid
#
min_Δx(grid::ZonallyStretchedRectilinearGrid) = minimum(view(grid.Δxᶜ, 1:grid.Nx))
min_Δy(grid::ZonallyStretchedRectilinearGrid) = grid.Δy
min_Δz(grid::ZonallyStretchedRectilinearGrid) = grid.Δz
=#
