struct RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch} <: AbstractRectilinearGrid{FT, TX, TY, TZ}
        architecture::Arch
        Nx :: Int
        Ny :: Int
        Nz :: Int
        Hx :: Int
        Hy :: Int
        Hz :: Int
        Lx :: FT
        Ly :: FT
        Lz :: FT
      # All directions can be either regular (FX, FY, FZ) <: Number or stretched (FX, FY, FZ)<: AbstractVector
      Δxᶠᵃᵃ :: FX
      Δxᶜᵃᵃ :: FX
      xᶠᵃᵃ  :: VX
      xᶜᵃᵃ  :: VX
      Δyᵃᶠᵃ :: FY
      Δyᵃᶜᵃ :: FY
      yᵃᶠᵃ  :: VY
      yᵃᶜᵃ  :: VY
      Δzᵃᵃᶠ :: FZ 
      Δzᵃᵃᶜ :: FZ
      zᵃᵃᶠ  :: VZ
      zᵃᵃᶜ  :: VZ
      # temporarly just to help refractoring with regular rectilinear grid (DELETE WHEN ALL THE Δx, Δy and Δz ARE REMOVED FROM THE CODE)
      Δx :: FT
      Δy :: FT
      Δz :: FT
end

function RectilinearGrid(FT = Float64;
         architecture = CPU(),
         size,
         x = nothing,
         y = nothing,
         z = nothing,
         halo = nothing,
         extent = nothing,
         topology = (Periodic, Periodic, Bounded))

    TX, TY, TZ = validate_topology(topology)
    size = validate_size(TX, TY, TZ, size)
    halo = validate_halo(TX, TY, TZ, halo)

    # Validate the rectilinear domain
    x, y, z = validate_rectilinear_domain(TX, TY, TZ, FT, extent, x, y, z)

    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    Lx, xᶠᵃᵃ, xᶜᵃᵃ, Δxᶠᵃᵃ, Δxᶜᵃᵃ = generate_coordinate(FT, topology[1], Nx, Hx, x, architecture)
    Ly, yᵃᶠᵃ, yᵃᶜᵃ, Δyᵃᶠᵃ, Δyᵃᶜᵃ = generate_coordinate(FT, topology[2], Ny, Hy, y, architecture)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topology[3], Nz, Hz, z, architecture)
 
    FX   = typeof(Δxᶠᵃᵃ)
    FY   = typeof(Δyᵃᶠᵃ)
    FZ   = typeof(Δzᵃᵃᶠ)
    VX   = typeof(xᶠᵃᵃ)
    VY   = typeof(yᵃᶠᵃ)
    VZ   = typeof(zᵃᵃᶠ)
    Arch = typeof(architecture) 

    FX<:AbstractVector ? Δx = Δxᶠᵃᵃ[1] : Δx = Δxᶠᵃᵃ
    FY<:AbstractVector ? Δy = Δyᵃᶠᵃ[1] : Δy = Δyᵃᶠᵃ
    FZ<:AbstractVector ? Δz = Δzᵃᵃᶠ[1] : Δz = Δzᵃᵃᶠ

    return RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
    Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ, Δx, Δy, Δz)
end

@inline x_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.yᵃᶠᵃ)
@inline z_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

short_show(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "RectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"
    

function show(io::IO, g::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RectilinearGrid{$FT, $TX, $TY, $TZ} \n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid in x: ", show_coordinate(g.Δxᶜᵃᵃ, TX), '\n',
              "grid in y: ", show_coordinate(g.Δyᵃᶜᵃ, TY), '\n',
              "grid in z: ", show_coordinate(g.Δzᵃᵃᶜ, TZ))
end


Adapt.adapt_structure(to, grid::RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ}) where {FT, TX, TY, TZ, FX, FY, FZ} =
    LatitudeLongitudeGrid{FT, TX, TY, TZ, FX, FY, FZ,
                            typeof(grid.xᶠᵃᵃ),
                            typeof(grid.yᵃᶠᵃ),
                            typeof(grid.zᵃᵃᶠ),
                            Nothing}(
        nothing,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        Adapt.adapt(to, grid.Δxᶜᵃᵃ),
        Adapt.adapt(to, grid.xᶠᵃᵃ),
        Adapt.adapt(to, grid.Δxᶠᵃᵃ),
        Adapt.adapt(to, grid.xᶜᵃᵃ),
        Adapt.adapt(to, grid.Δyᵃᶠᵃ),
        Adapt.adapt(to, grid.Δyᵃᶜᵃ),
        Adapt.adapt(to, grid.yᵃᶠᵃ),
        Adapt.adapt(to, grid.yᵃᶜᵃ),
        Adapt.adapt(to, grid.Δzᵃᵃᶠ),
        Adapt.adapt(to, grid.Δzᵃᵃᶜ),
        Adapt.adapt(to, grid.zᵃᵃᶠ),
        Adapt.adapt(to, grid.zᵃᵃᶜ),
        grid.Δx, grid.Δy, grid.Δz)

@inline xnode(::Center, i, grid::RectilinearGrid) = @inbounds grid.xᶜᵃᵃ[i]
@inline xnode(::Face, i, grid::RectilinearGrid) = @inbounds grid.xᶠᵃᵃ[i]

@inline ynode(::Center, j, grid::RectilinearGrid) = @inbounds grid.yᵃᶜᵃ[j]
@inline ynode(::Face, j, grid::RectilinearGrid) = @inbounds grid.yᵃᶠᵃ[j]

@inline znode(::Center, k, grid::RectilinearGrid) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(::Face, k, grid::RectilinearGrid) = @inbounds grid.zᵃᵃᶠ[k]

all_x_nodes(::Type{Center}, grid::RectilinearGrid) = grid.xᶜᵃᵃ
all_x_nodes(::Type{Face}, grid::RectilinearGrid) = grid.xᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::RectilinearGrid) = grid.yᵃᶜᵃ
all_y_nodes(::Type{Face}, grid::RectilinearGrid) = grid.yᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::RectilinearGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face}, grid::RectilinearGrid) = grid.zᵃᵃᶠ

# Get minima of grid
#

function min_Δx(grid::RectilinearGrid)
    topo = topology(grid)
    if topo[1] == Flat
        return Inf
    else
        return min_number_or_array(grid.Δxᶜᵃᵃ)
    end
end

function min_Δy(grid::RectilinearGrid)
    topo = topology(grid)
    if topo[2] == Flat
        return Inf
    else
        return min_number_or_array(grid.Δyᵃᶜᵃ)
    end
end

function min_Δz(grid::RectilinearGrid)
    topo = topology(grid)
    if topo[3] == Flat
        return Inf
    else
        return min_number_or_array(grid.Δzᵃᵃᶜ)
    end
end

@inline min_number_or_array(var) = var
@inline min_number_or_array(var::AbstractVector) = minimum(parent(var))