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
end

const XRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number}
const YRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const ZRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const HRegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number, <:Number}
const  RegRectilinearGrid = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Number, <:Number, <:Number}

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

    return RectilinearGrid{FT, TX, TY, TZ, FX, FY, FZ, VX, VY, VZ, Arch}(architecture,
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δxᶠᵃᵃ, Δxᶜᵃᵃ, xᶠᵃᵃ, xᶜᵃᵃ, Δyᵃᶜᵃ, Δyᵃᶠᵃ, yᵃᶠᵃ, yᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ)
end

@inline x_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.xᶠᵃᵃ)
@inline y_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.yᵃᶠᵃ)
@inline z_domain(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

short_show(grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
    "RectilinearGrid{$FT, $TX, $TY, $TZ}(Nx=$(grid.Nx), Ny=$(grid.Ny), Nz=$(grid.Nz))"
    

function domain_string(grid::RectilinearGrid)
    x₁, x₂ = domain(topology(grid, 1), grid.Nx, grid.xᶠᵃᵃ)
    y₁, y₂ = domain(topology(grid, 2), grid.Ny, grid.yᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)
    return "x ∈ [$x₁, $x₂], y ∈ [$y₁, $y₂], z ∈ [$z₁, $z₂]"
end

function show(io::IO, g::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    print(io, "RectilinearGrid{$FT, $TX, $TY, $TZ} on the $(g.architecture)\n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid in x: ", show_coordinate(g.Δxᶜᵃᵃ, TX), '\n',
              "grid in y: ", show_coordinate(g.Δyᵃᶜᵃ, TY), '\n',
              "grid in z: ", show_coordinate(g.Δzᵃᵃᶜ, TZ))
end


Adapt.adapt_structure(to, grid::RectilinearGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} =
             RectilinearGrid{FT, TX, TY, TZ, 
                          typeof(Adapt.adapt(to, grid.Δxᶜᵃᵃ)),
                          typeof(Adapt.adapt(to, grid.Δyᵃᶠᵃ)),
                          typeof(Adapt.adapt(to, grid.Δzᵃᵃᶠ)),
                          typeof(Adapt.adapt(to, grid.xᶠᵃᵃ)),
                          typeof(Adapt.adapt(to, grid.yᵃᶠᵃ)),
                          typeof(Adapt.adapt(to, grid.zᵃᵃᶠ)),
                          Nothing}(
        nothing,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        Adapt.adapt(to, grid.Δxᶠᵃᵃ),
        Adapt.adapt(to, grid.Δxᶜᵃᵃ),
        Adapt.adapt(to, grid.xᶠᵃᵃ),
        Adapt.adapt(to, grid.xᶜᵃᵃ),
        Adapt.adapt(to, grid.Δyᵃᶠᵃ),
        Adapt.adapt(to, grid.Δyᵃᶜᵃ),
        Adapt.adapt(to, grid.yᵃᶠᵃ),
        Adapt.adapt(to, grid.yᵃᶜᵃ),
        Adapt.adapt(to, grid.Δzᵃᵃᶠ),
        Adapt.adapt(to, grid.Δzᵃᵃᶜ),
        Adapt.adapt(to, grid.zᵃᵃᶠ),
        Adapt.adapt(to, grid.zᵃᵃᶜ))

@inline xnode(::Center, i, grid::RectilinearGrid) = @inbounds grid.xᶜᵃᵃ[i]
@inline xnode(::Face  , i, grid::RectilinearGrid) = @inbounds grid.xᶠᵃᵃ[i]

@inline ynode(::Center, j, grid::RectilinearGrid) = @inbounds grid.yᵃᶜᵃ[j]
@inline ynode(::Face  , j, grid::RectilinearGrid) = @inbounds grid.yᵃᶠᵃ[j]

@inline znode(::Center, k, grid::RectilinearGrid) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(::Face  , k, grid::RectilinearGrid) = @inbounds grid.zᵃᵃᶠ[k]

all_x_nodes(::Type{Center}, grid::RectilinearGrid) = grid.xᶜᵃᵃ
all_x_nodes(::Type{Face}  , grid::RectilinearGrid) = grid.xᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::RectilinearGrid) = grid.yᵃᶜᵃ
all_y_nodes(::Type{Face}  , grid::RectilinearGrid) = grid.yᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::RectilinearGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face}  , grid::RectilinearGrid) = grid.zᵃᵃᶠ

function with_halo(new_halo, old_grid::RectilinearGrid)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

   
    if old_grid.Δxᶠᵃᵃ isa Number
        x = x_domain(old_grid)
    else
        x = old_grid.xᶠᵃᵃ
    end

    if old_grid.Δyᵃᶠᵃ isa Number
        y = y_domain(old_grid)
    else
        y = old_grid.yᵃᶠᵃ
    end

    if old_grid.Δzᵃᵃᶠ isa Number
        z = z_domain(old_grid)
    else
        z = old_grid.zᵃᵃᶠ
    end

    # Remove elements of size and new_halo in Flat directions as expected by grid
    # constructor
    size     = pop_flat_elements(size, topo)
    new_halo = pop_flat_elements(new_halo, topo)

    new_grid = RectilinearGrid(eltype(old_grid);
               architecture = old_grid.architecture,
               size = size,
               x = x, y = y,z = z,
               topology = topo,
               halo = new_halo)

    return new_grid
end

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