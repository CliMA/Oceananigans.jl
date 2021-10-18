import Oceananigans.Architectures: architecture

const R_Earth = 6371.0e3    # Mean radius of the Earth [m] https://en.wikipedia.org/wiki/Earth

struct LatitudeLongitudeGrid{FT, TX, TY, TZ, FA, R, A, Arch} <: AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ}
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
        Δλ :: FT
        Δφ :: FT
      λᶠᵃᵃ :: R
      λᶜᵃᵃ :: R
      φᵃᶠᵃ :: R
      φᵃᶜᵃ :: R
      # z vertical direction can be stretched or regular
      Δzᵃᵃᶠ :: FA
      Δzᵃᵃᶜ :: FA
      zᵃᵃᶠ  :: A
      zᵃᵃᶜ  :: A
    radius  :: FT
end

# z can be a 2-tuple that specifies the end of the domain (see RegularRectilinearDomain) or an array or function that specifies the z-faces (see VerticallyStretchedRectilinearGrid)

function LatitudeLongitudeGrid(FT=Float64; 
                               architecture=CPU(),
                               size,
                               latitude,
                               longitude,
                               z,                      
                               radius=R_Earth,
                               halo=(1, 1, 1))

    # Horizontal directions 
    @assert length(latitude) == 2
    @assert length(longitude) == 2

    λ₁, λ₂ = longitude
    @assert λ₁ < λ₂ && λ₂ - λ₁ ≤ 360

    φ₁, φ₂ = latitude
    @assert -90 <= φ₁ < φ₂ <= 90

    (φ₁ == -90 || φ₂ == 90) &&
        @warn "Are you sure you want to use a latitude-longitude grid with a grid point at the pole?"

    Lλ = λ₂ - λ₁
    Lφ = φ₂ - φ₁

    TX = Lλ == 360 ? Periodic : Bounded
    TY = Bounded
    TZ = Bounded
    topo = (TX, TY, TZ)
    
    Nλ, Nφ, Nz = N = validate_size(TX, TY, TZ, size)
    Hλ, Hφ, Hz = H = validate_halo(TX, TY, TZ, halo)
    
    Nh = N[1:2]
    Hh = H[1:2]
    
        Λ₁ = (λ₁, φ₁)
        Lh = (Lλ, Lφ)
    Δλ, Δφ = Δh = @. Lh / Nh

    # Calculate end points for cell faces and centers (Horizontal directions)
    λF₋, φF₋ = ΛF₋ = @. Λ₁ - Hh * Δh
    λF₊, φF₊ = ΛF₊ = @. ΛF₋ + total_extent(topo[1:2], Hh, Δh, Lh)

    λC₋, φC₋ = ΛC₋ = @. ΛF₋ + Δh / 2
    λC₊, φC₊ = ΛC₊ = @. ΛC₋ + Lh + Δh * (2Hh - 1)

    TFλ, TFφ, TFz = total_length.(Face,   topo, N, H)
    TCλ, TCφ, TCz = total_length.(Center, topo, N, H)

    λᶠᵃᵃ = range(λF₋, λF₊, length = TFλ)
    φᵃᶠᵃ = range(φF₋, φF₊, length = TFφ)

    λᶜᵃᵃ = range(λC₋, λC₊, length = TCλ)
    φᵃᶜᵃ = range(φC₋, φC₊, length = TCφ)

    λᶠᵃᵃ = OffsetArray(λᶠᵃᵃ, -Hλ)
    φᵃᶠᵃ = OffsetArray(φᵃᶠᵃ, -Hφ)

    λᶜᵃᵃ = OffsetArray(λᶜᵃᵃ, -Hλ)
    φᵃᶜᵃ = OffsetArray(φᵃᶜᵃ, -Hφ)

    # Calculate vertical direction (which might be stretched)
    # It is regular if z is Tuple{<:Real, <:Real}, 
    # it is stretched if being passed is a function or vector (as for the VerticallyStretchedRectilinearGrid)
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, topo[3], Nz, Hz, z, architecture)

    R    = typeof(λᶠᵃᵃ)
    FA   = typeof(Δzᵃᵃᶠ)
    A    = typeof(zᵃᵃᶠ)
    Arch = typeof(architecture) 

    return LatitudeLongitudeGrid{FT, TX, TY, TZ, FA, R, A, Arch}(architecture,
            Nλ, Nφ, Nz, Hλ, Hφ, Hz, Lλ, Lφ, Lz, Δλ, Δφ, λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, zᵃᵃᶠ, zᵃᵃᶜ, radius)
end

function domain_string(grid::LatitudeLongitudeGrid)
    λ₁, λ₂ = domain(topology(grid, 1), grid.Nx, grid.λᶠᵃᵃ)
    φ₁, φ₂ = domain(topology(grid, 2), grid.Ny, grid.φᵃᶠᵃ)
    z₁, z₂ = domain(topology(grid, 3), grid.Nz, grid.zᵃᵃᶠ)
    return "longitude λ ∈ [$λ₁, $λ₂], latitude ∈ [$φ₁, $φ₂], z ∈ [$z₁, $z₂]"
end

function show(io::IO, g::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    Δz_min = minimum(view(parent(g.Δzᵃᵃᶜ), g.Hz+1:g.Nz+g.Hz))
    Δz_max = maximum(view(parent(g.Δzᵃᵃᶜ), g.Hz+1:g.Nz+g.Hz))
    print(io, "LatitudeLongitudeGrid{$FT, $TX, $TY, $TZ} Stretched in the vertical direction \n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δλ, Δφ, Δz): (", g.Δλ, ", ", g.Δφ, ", [min=", Δz_min, ", max=", Δz_max,"])",)
end

function show(io::IO, g::LatitudeLongitudeGrid{FT, TX, TY, TZ, FA}) where {FT, TX, TY, TZ, FA<:Number}
    print(io, "LatitudeLongitudeGrid{$FT, $TX, $TY, $TZ}  Regular in the vertical direction \n",
              "                   domain: $(domain_string(g))\n",
              "                 topology: ", (TX, TY, TZ), '\n',
              "        size (Nx, Ny, Nz): ", (g.Nx, g.Ny, g.Nz), '\n',
              "        halo (Hx, Hy, Hz): ", (g.Hx, g.Hy, g.Hz), '\n',
              "grid spacing (Δλ, Δφ, Δz): ", (g.Δλ, g.Δφ, g.Δzᵃᵃᶜ))
end

# Node by node
@inline xnode(::Center, i, grid::LatitudeLongitudeGrid) = @inbounds grid.λᶜᵃᵃ[i]
@inline xnode(::Face,   i, grid::LatitudeLongitudeGrid) = @inbounds grid.λᶠᵃᵃ[i]

@inline ynode(::Center, j, grid::LatitudeLongitudeGrid) = @inbounds grid.φᵃᶜᵃ[j]
@inline ynode(::Face,   j, grid::LatitudeLongitudeGrid) = @inbounds grid.φᵃᶠᵃ[j]

@inline znode(::Center, k, grid::LatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶠ[k]
@inline znode(::Face,   k, grid::LatitudeLongitudeGrid) = @inbounds grid.zᵃᵃᶜ[k]

all_x_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.λᶜᵃᵃ
all_x_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.λᶠᵃᵃ
all_y_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.φᵃᶜᵃ
all_y_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.φᵃᶠᵃ
all_z_nodes(::Type{Center}, grid::LatitudeLongitudeGrid) = grid.zᵃᵃᶜ
all_z_nodes(::Type{Face},   grid::LatitudeLongitudeGrid) = grid.zᵃᵃᶠ

architecture(::LatitudeLongitudeGrid) = nothing

# get_z_lat_lon(z::Function, k) = z(k)
# get_z_lat_lon(z::AbstractVector, k) = CUDA.@allowscalar z[k]

# lower_lat_lon_exterior_Δzᵃᵃᶜ(z_topo,          zFi, Hz) = [zFi[end - Hz + k] - zFi[end - Hz + k - 1] for k = 1:Hz]
# lower_lat_lon_exterior_Δzᵃᵃᶜ(::Type{Bounded}, zFi, Hz) = [zFi[2]  - zFi[1] for k = 1:Hz]

# upper_lat_lon_exterior_Δzᵃᵃᶜ(z_topo,          zFi, Hz) = [zFi[k + 1] - zFi[k] for k = 1:Hz]
# upper_lat_lon_exterior_Δzᵃᵃᶜ(::Type{Bounded}, zFi, Hz) = [zFi[end]   - zFi[end - 1] for k = 1:Hz]

# function generate_vertical_grid_lat_lon(FT, z_topo, Nz, Hz, z_faces, architecture)

#     # Ensure correct type for zF and derived quantities
#     interior_zF = zeros(FT, Nz+1)

#     for k = 1:Nz+1
#         interior_zF[k] = get_z_lat_lon(z_faces, k)
#     end

#     Lz = interior_zF[Nz+1] - interior_zF[1]

#     # Build halo regions
#     ΔzF₋ = lower_lat_lon_exterior_Δzᵃᵃᶜ(z_topo, interior_zF, Hz)
#     ΔzF₊ = upper_lat_lon_exterior_Δzᵃᵃᶜ(z_topo, interior_zF, Hz)

#     z¹, zᴺ⁺¹ = interior_zF[1], interior_zF[Nz+1]

#     zF₋ = [z¹   - sum(ΔzF₋[k:Hz]) for k = 1:Hz] # locations of faces in lower halo
#     zF₊ = reverse([zᴺ⁺¹ + sum(ΔzF₊[k:Hz]) for k = 1:Hz]) # locations of faces in width of top halo region

#     zF = vcat(zF₋, interior_zF, zF₊)

#     # Build cell centers, cell center spacings, and cell interface spacings
#     TCz = total_length(Center, z_topo, Nz, Hz)
#      zC = [ (zF[k + 1] + zF[k]) / 2 for k = 1:TCz ]
#     ΔzC = [  zC[k] - zC[k - 1]      for k = 2:TCz ]

#     # Trim face locations for periodic domains
#     TFz = total_length(Face, z_topo, Nz, Hz)
#     zF = zF[1:TFz]

#     ΔzF = [zF[k + 1] - zF[k] for k = 1:TFz-1]

#     ΔzF = OffsetArray(ΔzF, -Hz)
#     ΔzC = OffsetArray(ΔzC, -Hz)

#     # Seems needed to avoid out-of-bounds error in viscous dissipation
#     # operators wanting to access Δzᵃᵃᶠ[Nz+2].
#     ΔzF = OffsetArray(cat(ΔzF[0], ΔzF..., ΔzF[Nz], dims=1), -Hz-1)

#     ΔzF = OffsetArray(arch_array(architecture, ΔzF.parent), ΔzF.offsets...)
#     ΔzC = OffsetArray(arch_array(architecture, ΔzC.parent), ΔzC.offsets...)

#     zF = OffsetArray(zF, -Hz)
#     zC = OffsetArray(zC, -Hz)

#     # Convert to appropriate array type for arch
#     zF  = OffsetArray(arch_array(architecture,  zF.parent),  zF.offsets...)
#     zC  = OffsetArray(arch_array(architecture,  zC.parent),  zC.offsets...)

#     return Lz, zF, zC, ΔzF, ΔzC
# end

# #function generate_vertical_grid_lat_lon(FT, coord_topo, Ncoord, Hcoord, coord::Tuple{<:Number, <:Number}, architecture)
# function generate_vertical_grid_lat_lon(FT, coord_topo, Ncoord, Hcoord, coord::Tuple{<:Number, <:Number}, architecture)

#     @assert length(z_faces) == 2

#     z₁, z₂ = z_faces
#     @assert z₁ < z₂
#     Lz = z₂ - z₁

#     # Convert to get the correct type also when using single precision
#     ΔzF = ΔzC = Δz = convert(FT, Lz / Nz)

#     zF₋ = z₁ - Hz * Δz
#     zF₊ = zF₋ + total_extent(z_topo, Hz, Δz, Lz)

#     zC₋ = zF₋ + Δz / 2
#     zC₊ = zC₋ + Lz + Δz * (2Hz - 1)

#     TFz = total_length(Face,   z_topo, Nz, Hz)
#     TCz = total_length(Center, z_topo, Nz, Hz)

#     zF = range(zF₋, zF₊, length = TFz)
#     zC = range(zC₋, zC₊, length = TCz)

#     zF = OffsetArray(zF, -Hz)
#     zC = OffsetArray(zC, -Hz)
    
#     return Lz, zF, zC, ΔzF, ΔzC
# end

@inline x_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TX, grid.Nx, grid.λᶠᵃᵃ)
@inline y_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TY, grid.Ny, grid.φᵃᶠᵃ)
@inline z_domain(grid::LatitudeLongitudeGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)

Adapt.adapt_structure(to, grid::LatitudeLongitudeGrid{FT, TX, TY, TZ, FA}) where {FT, TX, TY, TZ, FA<:AbstractVector} =
    LatitudeLongitudeGrid{FT, TX, TY, TZ, FA,
                          typeof(Adapt.adapt(to, grid.λᶜᵃᵃ)),
                          typeof(Adapt.adapt(to, grid.zᵃᵃᶠ)),
                          Nothing}(
        nothing,
        grid.Nx, grid.Ny, grid.Nz,
        grid.Hx, grid.Hy, grid.Hz,
        grid.Lx, grid.Ly, grid.Lz,
        grid.Δλ, grid.Δφ,
        grid.λᶠᵃᵃ, grid.λᶜᵃᵃ,
        grid.φᵃᶠᵃ, grid.φᵃᶜᵃ,
        Adapt.adapt(to, grid.Δzᵃᵃᶠ),
        Adapt.adapt(to, grid.Δzᵃᵃᶜ),
        Adapt.adapt(to, grid.zᵃᵃᶠ),
        Adapt.adapt(to, grid.zᵃᵃᶜ))
