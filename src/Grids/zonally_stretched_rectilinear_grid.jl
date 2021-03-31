"""
    ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ, R, A} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

A rectilinear grid with with constant grid spacings `Δy` and `Δz`, and
non-uniform or stretched zonal grid spacing `Δx` between cell centers and cell faces,
topology `{TX, TY, TZ}`, and coordinate ranges of type `R` (where a range can be used) and
`A` (where an array is needed).
"""
struct ZonallyStretchedRectilinearGrid{FT, TX, TY, TZ, R, A} <: AbstractRectilinearGrid{FT, TX, TY, TZ}

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

    # Grid spacing [m].
    Δxᶜᵃᵃ :: A
    Δxᶠᵃᵃ :: A
       Δy :: FT
       Δz :: FT

    # Range of coordinates at the centers of the cells.
    xᶜᵃᵃ :: A
    yᵃᶜᵃ :: R
    zᵃᵃᶜ :: R

    # Range of grid coordinates at the faces of the cells.
    # Note: there are Nx+1 faces in the x-dimension, Ny+1 in the y, and Nz+1 in the z.
    xᶠᵃᵃ :: A
    yᵃᶠᵃ :: R
    zᵃᵃᶠ :: R
end

# FJP: since this is for ShallowWater, should the default in z be Flat?
function ZonallyStretchedRectilinearGrid(FT=Float64; architecture = CPU(),
                                              size, xF, y, z,
                                              halo = (1, 1, 1),
                                          topology = (Bounded, Periodic, Periodic))

    TX, TY, TZ = validate_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, halo)
    Lx, Ly, x, y = validate_zonally_stretched_grid_yz(TY, TZ, FT, y, z)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    # Initialize zonally-stretched arrays on CPU
    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶜᵃᵃ, Δxᶠᵃᵃ = generate_stretched_zonal_grid(FT, topology[3], Nx, Hx, xF)

    # Construct uniform horizontal grid
    # FJP: don't want to force the grid to be uniform in the vertical slice!
    Lh, Nh, Hh, X₁ = (Lx, Ly), size[1:2], halo[1:2], (x[1], y[1])
    Δx, Δy = Δh = Lh ./ Nh

    # Face-node limits in x, y, z
    xF₋, yF₋ = XF₋ = @. X₁ - Hh * Δh
    xF₊, yF₊ = XF₊ = @. XF₋ + total_extent(topology[1:2], Hh, Δh, Lh)

    # Center-node limits in x, y, z
    xC₋, yC₋ = XC₋ = @. XF₋ + Δh / 2
    xC₊, yC₊ = XC₊ = @. XC₋ + Lh + Δh * (2Hh - 1)

    # Total length of Center and Face quantities
    TFx, TFy, TFz = total_length.(Face, topology, size, halo)
    TCx, TCy, TCz = total_length.(Center, topology, size, halo)

    # Include halo points in coordinate arrays
    xᶠᵃᵃ = range(xF₋, xF₊; length = TFx)
    yᵃᶠᵃ = range(yF₋, yF₊; length = TFy)

    xᶜᵃᵃ = range(xC₋, xC₊; length = TCx)
    yᵃᶜᵃ = range(yC₋, yC₊; length = TCy)

    xᶜᵃᵃ = OffsetArray(xᶜᵃᵃ,  -Hx)
    yᵃᶜᵃ = OffsetArray(yᵃᶜᵃ,  -Hy)
    zᵃᵃᶜ = OffsetArray(zᵃᵃᶜ,  -Hz)

    xᶠᵃᵃ = OffsetArray(xᶠᵃᵃ,  -Hx)
    yᵃᶠᵃ = OffsetArray(yᵃᶠᵃ,  -Hy)
    zᵃᵃᶠ = OffsetArray(zᵃᵃᶠ,  -Hz)

    Δzᵃᵃᶠ = OffsetArray(Δzᵃᵃᶠ, -Hz)
    Δzᵃᵃᶜ = OffsetArray(Δzᵃᵃᶜ, -Hz)

    # Needed for pressure solver solution to be divergence-free.
    # Will figure out why later...
    Δzᵃᵃᶠ[Nz] = Δzᵃᵃᶠ[Nz-1]

    # Seems needed to avoid out-of-bounds error in viscous dissipation
    # operators wanting to access Δzᵃᵃᶠ[Nz+2].
    Δzᵃᵃᶠ = OffsetArray(cat(Δzᵃᵃᶠ[0], Δzᵃᵃᶠ..., Δzᵃᵃᶠ[Nz], dims=1), -Hz-1)

    # Convert to appropriate array type for arch
    zᵃᵃᶠ  = OffsetArray(arch_array(architecture,  zᵃᵃᶠ.parent),  zᵃᵃᶠ.offsets...)
    zᵃᵃᶜ  = OffsetArray(arch_array(architecture,  zᵃᵃᶜ.parent),  zᵃᵃᶜ.offsets...)
    Δzᵃᵃᶜ = OffsetArray(arch_array(architecture, Δzᵃᵃᶜ.parent), Δzᵃᵃᶜ.offsets...)
    Δzᵃᵃᶠ = OffsetArray(arch_array(architecture, Δzᵃᵃᶠ.parent), Δzᵃᵃᶠ.offsets...)

    return VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ, typeof(xᶠᵃᵃ), typeof(zᵃᵃᶠ)}(
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δx, Δy, Δzᵃᵃᶜ, Δzᵃᵃᶠ, xᶜᵃᵃ, yᵃᶜᵃ, zᵃᵃᶜ, xᶠᵃᵃ, yᵃᶠᵃ, zᵃᵃᶠ)
end

#####
##### Vertically stretched grid utilities
#####

get_z_face(z::Function, k) = z(k)
get_z_face(z::AbstractVector, k) = z[k]

lower_exterior_Δzᵃᵃᶜ(z_topo,          zFi, Hz) = [zFi[end - Hz + k] - zFi[end - Hz + k - 1] for k = 1:Hz]
lower_exterior_Δzᵃᵃᶜ(::Type{Bounded}, zFi, Hz) = [zFi[2]  - zFi[1] for k = 1:Hz]

upper_exterior_Δzᵃᵃᶜ(z_topo,          zFi, Hz) = [zFi[k + 1] - zFi[k] for k = 1:Hz]
upper_exterior_Δzᵃᵃᶜ(::Type{Bounded}, zFi, Hz) = [zFi[end]   - zFi[end - 1] for k = 1:Hz]

function generate_stretched_zonal_grid(FT, x_topo, Nx, Hx, xF_generator)

    # Ensure correct type for xF and derived quantities
    interior_xF = zeros(FT, Nx+1)

    for i = 1:Nx+1
        interior_xF[i] = get_x_face(xF_generator, i)
    end

    Lx = interior_xF[Nx+1] - interior_xF[1]

    # Build halo regions
    ΔxF₋ = lower_exterior_Δxᶜᵃᵃ(x_topo, interior_xF, Hx)
    ΔxF₊ = lower_exterior_Δxᶜᵃᵃ(x_topo, interior_xF, Hx)

    x¹, xᴺ⁺¹ = interior_xF[1], interior_xF[Nx+1]

    xF₋ = [x¹   - sum(ΔxF₋[i:Hx]) for i = 1:Hx] # locations of faces in lower halo
    xF₊ = [xᴺ⁺¹ + ΔxF₊[i]         for i = 1:Hx] # locations of faces in width of top halo region

    xF = vcat(xF₋, interior_xF, xF₊)

    # Build cell centers, cell center spacings, and cell interface spacings
    TCx = total_length(Center, x_topo, Nx, Hx)
     xC = [ (xF[i + 1] + xF[i]) / 2 for i = 1:TCx ]
    ΔxC = [  xC[i] - xC[i - 1]      for i = 2:TCx ]

    # Trim face locations for periodic domains
    TFx = total_length(Face, x_topo, Nx, Hx)
    xF = xF[1:TFx]

    #FJP: what if Flat????
    ΔxF = [xF[i + 1] - xF[i] for i = 1:TFx-1]

    return Lx, xF, xC, ΔxF, ΔxC
end

# We cannot reconstruct a VerticallyStretchedRectilinearGrid without the zF_generator.
# So the best we can do is tell the user what they should have done.
function with_halo(new_halo, old_grid::VerticallyStretchedRectilinearGrid)
    new_halo != halo_size(old_grid) &&
        @error "You need to construct your VerticallyStretchedRectilinearGrid with the keyword argument halo=$new_halo"
    return old_grid
end

@inline x_domain(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.yᵃᶠᵃ)
@inline z_domain(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

short_show(grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "VerticallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"

function show(io::IO, g::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    Δz_min = minimum(view(g.Δzᵃᵃᶜ, 1:g.Nz))
    Δz_max = maximum(view(g.Δzᵃᵃᶜ, 1:g.Nz))
    print(io, "VerticallyStretchedRectilinearGrid{$FT, $TX, $TY, $TZ}\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "  resolution (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "   halo size (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δx, Δy, Δz): (", g.Δx, ", ", g.Δy, ", [min=", Δz_min, ", max=", Δz_max,"])",)
end

Adapt.adapt_structure(to, grid::VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    VerticallyStretchedRectilinearGrid{FT, TX, TY, TZ, typeof(grid.xᶠᵃᵃ), typeof(Adapt.adapt(to, grid.zᵃᵃᶠ))}(
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        grid.Δx, grid.Δy,
        Adapt.adapt(to, grid.Δzᵃᵃᶜ),
        Adapt.adapt(to, grid.Δzᵃᵃᶠ),
        grid.xᶜᵃᵃ, grid.yᵃᶜᵃ,
        Adapt.adapt(to, grid.zᵃᵃᶜ),
        grid.xᶠᵃᵃ, grid.yᵃᶠᵃ,
        Adapt.adapt(to, grid.zᵃᵃᶠ))

#####
##### Should merge with grid_utils.jl at some point
#####

@inline xnode(::Type{Center}, i, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.xᶜᵃᵃ[i]
@inline xnode(::Type{Face}, i, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.xᶠᵃᵃ[i]

@inline ynode(::Type{Center}, j, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.yᵃᶜᵃ[j]
@inline ynode(::Type{Face}, j, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.yᵃᶠᵃ[j]

@inline znode(::Type{Center}, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(::Type{Face}, k, grid::VerticallyStretchedRectilinearGrid) = @inbounds grid.zᵃᵃᶠ[k]


all_x_nodes(::Type{Center}, grid::VerticallyStretchedRectilinearGrid) = grid.xᶜᵃᵃ
all_x_nodes(::Type{Face}, grid::VerticallyStretchedRectilinearGrid) = grid.xᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::VerticallyStretchedRectilinearGrid) = grid.yᵃᶜᵃ
all_y_nodes(::Type{Face}, grid::VerticallyStretchedRectilinearGrid) = grid.yᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::VerticallyStretchedRectilinearGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face}, grid::VerticallyStretchedRectilinearGrid) = grid.zᵃᵃᶠ



#
# Get minima of grid
#

min_Δx(grid::VerticallyStretchedRectilinearGrid) = grid.Δx
min_Δy(grid::VerticallyStretchedRectilinearGrid) = grid.Δy
min_Δz(grid::VerticallyStretchedRectilinearGrid) = minimum(view(grid.Δzᵃᵃᶜ, 1:grid.Nz))

