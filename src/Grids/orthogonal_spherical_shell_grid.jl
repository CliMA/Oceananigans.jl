using CubedSphere
using JLD2
using OffsetArrays
using Adapt
using Distances

using Adapt: adapt_structure

using Oceananigans
using Oceananigans.Grids: prettysummary, coordinate_summary, BoundedTopology, length
                  
abstract type AbstractOrthogonalMapping end

struct OrthogonalSphericalShellGrid{FT, C, TX, TY, TZ, FX, FY, FZ, X, Y, Z, Arch} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, Arch}
    architecture :: Arch
    mapping :: C
    Nx :: Int
    Ny :: Int
    Nz :: Int
    Hx :: Int
    Hy :: Int
    Hz :: Int
    Lx :: FT
    Ly :: FT
    Lz :: FT
    λᶜᶜᵃ :: X
    λᶠᶜᵃ :: X
    λᶜᶠᵃ :: X
    λᶠᶠᵃ :: X
    φᶜᶜᵃ :: Y
    φᶠᶜᵃ :: Y
    φᶜᶠᵃ :: Y
    φᶠᶠᵃ :: Y
    zᵃᵃᶜ :: Z
    zᵃᵃᶠ :: Z
    # Spacings
    Δzᵃᵃᶜ :: FZ
    Δzᵃᵃᶠ :: FZ
    Δxᶜᶜᵃ :: FX
    Δxᶠᶜᵃ :: FX
    Δxᶜᶠᵃ :: FX
    Δxᶠᶠᵃ :: FX
    Δyᶜᶜᵃ :: FY
    Δyᶜᶠᵃ :: FY
    Δyᶠᶜᵃ :: FY
    Δyᶠᶠᵃ :: FY
    Azᶜᶜᵃ :: FX
    Azᶠᶜᵃ :: FX
    Azᶜᶠᵃ :: FX
    Azᶠᶠᵃ :: FX
    radius :: FT

    """
        OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture::Arch,
                                                 mapping :: C,
                                                 Nx, Ny, Nz,
                                                 Hx, Hy, Hz,
                                                 Lx :: FT, Ly :: FT, Lz :: FT,
                                                  λᶜᶜᵃ :: X,   λᶠᶜᵃ :: X,   λᶜᶠᵃ :: X,   λᶠᶠᵃ :: X,
                                                  φᶜᶜᵃ :: Y,   φᶠᶜᵃ :: Y,   φᶜᶠᵃ :: Y,   φᶠᶠᵃ :: Y,  zᵃᵃᶜ :: Z,   zᵃᵃᶠ :: Z,
                                                 Δzᵃᵃᶜ :: FZ, Δzᵃᵃᶠ :: FZ,
                                                 Δxᶜᶜᵃ :: FX,  Δxᶠᶜᵃ :: FX, Δxᶜᶠᵃ :: FX, Δxᶠᶠᵃ :: FX,
                                                 Δyᶜᶜᵃ :: FY,  Δyᶜᶠᵃ :: FY, Δyᶠᶜᵃ :: FY, Δyᶠᶠᵃ :: FY, 
                                                 Azᶜᶜᵃ :: FX,  Azᶠᶜᵃ :: FX, Azᶜᶠᵃ :: FX, Azᶠᶠᵃ :: FX,
                                                 radius :: FT)
    
    An internal constructor for a generic `OrthogonalSphericalShellGrid`. 
    """
    OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture::Arch,
                                             mapping :: C,
                                             Nx, Ny, Nz,
                                             Hx, Hy, Hz,
                                             Lx :: FT, Ly :: FT, Lz :: FT,
                                              λᶜᶜᵃ :: X,   λᶠᶜᵃ :: X,   λᶜᶠᵃ :: X,   λᶠᶠᵃ :: X,
                                              φᶜᶜᵃ :: Y,   φᶠᶜᵃ :: Y,   φᶜᶠᵃ :: Y,   φᶠᶠᵃ :: Y,  zᵃᵃᶜ :: Z,   zᵃᵃᶠ :: Z,
                                             Δzᵃᵃᶜ :: FZ, Δzᵃᵃᶠ :: FZ,
                                             Δxᶜᶜᵃ :: FX,  Δxᶠᶜᵃ :: FX, Δxᶜᶠᵃ :: FX, Δxᶠᶠᵃ :: FX,
                                             Δyᶜᶜᵃ :: FY,  Δyᶜᶠᵃ :: FY, Δyᶠᶜᵃ :: FY, Δyᶠᶠᵃ :: FY, 
                                             Azᶜᶜᵃ :: FX,  Azᶠᶜᵃ :: FX, Azᶜᶠᵃ :: FX, Azᶠᶠᵃ :: FX,
                                             radius :: FT) where {TX, TY, TZ, FT, C, X, Y, Z, FX, FY, FZ, Arch} =
        new{FT, C, TX, TY, TZ, FX, FY, FZ, X, Y, Z, Arch}(architecture,
                                                          mapping,
                                                          Nx, Ny, Nz,
                                                          Hx, Hy, Hz,
                                                          Lx, Ly, Lz,
                                                          λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                                          φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                                          Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                                          Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                          Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, 
                                                          Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)
end

const OSSG = OrthogonalSphericalShellGrid
const ZRegOSSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Number}
const ZRegOrthogonalSphericalShellGrid = ZRegOSSG

OrthogonalSphericalShellGrid(architecture, Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz,
                             λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ, φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                             Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                             Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius) =
    OrthogonalSphericalShellGrid(architecture, nothing, Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz,
                                 λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ, φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, zᵃᵃᶜ, zᵃᵃᶠ,
                                 Δzᵃᵃᶜ, Δzᵃᵃᶠ,
                                 Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, 
                                 Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, radius)

function on_architecture(arch::AbstractArchitecture, grid::OrthogonalSphericalShellGrid)

    coordinates = (:λᶜᶜᵃ,
                   :λᶠᶜᵃ,
                   :λᶜᶠᵃ,
                   :λᶠᶠᵃ,
                   :φᶜᶜᵃ,
                   :φᶠᶜᵃ,
                   :φᶜᶠᵃ,
                   :φᶠᶠᵃ,
                   :zᵃᵃᶜ,
                   :zᵃᵃᶠ)

    grid_spacings = (:Δzᵃᵃᶜ,
                     :Δzᵃᵃᶠ,      
                     :Δxᶜᶜᵃ,
                     :Δxᶠᶜᵃ,
                     :Δxᶜᶠᵃ,
                     :Δxᶠᶠᵃ,
                     :Δyᶜᶜᵃ,
                     :Δyᶜᶠᵃ,
                     :Δyᶠᶜᵃ,
                     :Δyᶠᶠᵃ)

    horizontal_areas = (:Azᶜᶜᵃ,
                        :Azᶠᶜᵃ,
                        :Azᶜᶠᵃ,
                        :Azᶠᶠᵃ)

    grid_spacing_data    = Tuple(arch_array(arch, getproperty(grid, name)) for name in grid_spacings)
    coordinate_data      = Tuple(arch_array(arch, getproperty(grid, name)) for name in coordinates)
    horizontal_area_data = Tuple(arch_array(arch, getproperty(grid, name)) for name in horizontal_areas)

    TX, TY, TZ = topology(grid)

    new_grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch,
                                                        grid.mapping,
                                                        grid.Nx, grid.Ny, grid.Nz,
                                                        grid.Hx, grid.Hy, grid.Hz,
                                                        grid.Lx, grid.Ly, grid.Lz,
                                                        coordinate_data...,
                                                        grid_spacing_data...,
                                                        horizontal_area_data...,
                                                        grid.radius)

    return new_grid
end

function Adapt.adapt_structure(to, grid::OrthogonalSphericalShellGrid)
    TX, TY, TZ = topology(grid)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(nothing,
                                                    grid.mapping,
                                                    grid.Nx, grid.Ny, grid.Nz,
                                                    grid.Hx, grid.Hy, grid.Hz,
                                                    grid.Lx, grid.Ly, grid.Lz,
                                                    adapt(to, grid.λᶜᶜᵃ),
                                                    adapt(to, grid.λᶠᶜᵃ),
                                                    adapt(to, grid.λᶜᶠᵃ),
                                                    adapt(to, grid.λᶠᶠᵃ),
                                                    adapt(to, grid.φᶜᶜᵃ),
                                                    adapt(to, grid.φᶠᶜᵃ),
                                                    adapt(to, grid.φᶜᶠᵃ),
                                                    adapt(to, grid.φᶠᶠᵃ),
                                                    adapt(to, grid.zᵃᵃᶜ),
                                                    adapt(to, grid.zᵃᵃᶠ),
                                                    adapt(to, grid.Δzᵃᵃᶜ),
                                                    adapt(to, grid.Δzᵃᵃᶠ),
                                                    adapt(to, grid.Δxᶜᶜᵃ),
                                                    adapt(to, grid.Δxᶠᶜᵃ),
                                                    adapt(to, grid.Δxᶜᶠᵃ),
                                                    adapt(to, grid.Δxᶠᶠᵃ),
                                                    adapt(to, grid.Δyᶜᶜᵃ),
                                                    adapt(to, grid.Δyᶜᶠᵃ),
                                                    adapt(to, grid.Δyᶠᶜᵃ),
                                                    adapt(to, grid.Δyᶠᶠᵃ),
                                                    adapt(to, grid.Azᶜᶜᵃ),
                                                    adapt(to, grid.Azᶠᶜᵃ),
                                                    adapt(to, grid.Azᶜᶠᵃ),
                                                    adapt(to, grid.Azᶠᶠᵃ),
                                                    grid.radius)
end

function Base.summary(grid::OrthogonalSphericalShellGrid)
    FT = eltype(grid)
    TX, TY, TZ = topology(grid)
    metric_computation = isnothing(grid.Δxᶠᶜᵃ) ? "without precomputed metrics" : "with precomputed metrics"

    return string(size_summary(size(grid)),
                  " OrthogonalSphericalShellGrid{$FT, $TX, $TY, $TZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo",
                  " and ", metric_computation)
end

"""
    get_extents_of_shell(grid::OSSG)

Return the longitudinal and latitudinal extend of the shell `(extent_λ, extent_φ)`.
"""
function get_extents_of_shell(grid)
    Nx, Ny, _ = size(grid)

    # the Δλ, Δφ are approximate if ξ, η are not symmetric about 0
    if mod(Ny, 2) == 0
        extent_λ = maximum(rad2deg.(sum(grid.Δxᶜᶠᵃ[1:Nx, :], dims=1))) / grid.radius
    elseif mod(Ny, 2) == 1
        extent_λ = maximum(rad2deg.(sum(grid.Δxᶜᶜᵃ[1:Nx, :], dims=1))) / grid.radius
    end

    if mod(Nx, 2) == 0
        extent_φ = maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
    elseif mod(Nx, 2) == 1
        extent_φ = maximum(rad2deg.(sum(grid.Δyᶠᶜᵃ[:, 1:Ny], dims=2))) / grid.radius
    end

    return (extent_λ, extent_φ)
end

"""
    get_center_of_shell(grid::OSSG)

Return the latitude-longitude coordinates of the center of the shell `(λ_center, φ_center)`.
"""
function get_center_of_shell(grid)
    Nx, Ny, _ = size(grid)

    # find the indices that correspond to the center of the shell
    # ÷ ensures that expressions below work for both odd and even
    i_center = Nx÷2 + 1
    j_center = Ny÷2 + 1

    if mod(Nx, 2) == 0
        ℓx = Face()
    elseif mod(Nx, 2) == 1
        ℓx = Center()
    end

    if mod(Ny, 2) == 0
        ℓy = Face()
    elseif mod(Ny, 2) == 1
        ℓy = Center()
    end

    # latitude and longitudes of the shell's center
    λ_center = λnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())
    φ_center = φnode(i_center, j_center, 1, grid, ℓx, ℓy, Center())

    return (λ_center, φ_center)
end

function Base.show(io::IO, grid::OrthogonalSphericalShellGrid, withsummary=true)
    TX, TY, TZ = topology(grid)
    Nx, Ny, Nz = size(grid)

    Nx_face, Ny_face = total_length(Face(), TX(), Nx, 0), total_length(Face(), TY(), Ny, 0)

    λ₁, λ₂ = minimum(grid.λᶠᶠᵃ[1:Nx_face, 1:Ny_face]), maximum(grid.λᶠᶠᵃ[1:Nx_face, 1:Ny_face])
    φ₁, φ₂ = minimum(grid.φᶠᶠᵃ[1:Nx_face, 1:Ny_face]), maximum(grid.φᶠᶠᵃ[1:Nx_face, 1:Ny_face])
    z₁, z₂ = domain(topology(grid, 3)(), Nz, grid.zᵃᵃᶠ)

    (λ_center, φ_center) = get_center_of_shell(grid)
    (extent_λ, extent_φ) = get_extents_of_shell(grid)

    λ_center = round(λ_center, digits=4)
    φ_center = round(φ_center, digits=4)

    λ_center = ifelse(λ_center ≈ 0, 0.0, λ_center)
    φ_center = ifelse(φ_center ≈ 0, 0.0, φ_center)

    center_str = "centered at (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"

    if φ_center ≈ 90
        center_str = "centered at: North Pole, (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"
    end

    if φ_center ≈ -90
        center_str = "centered at: South Pole, (λ, φ) = (" * prettysummary(λ_center) * ", " * prettysummary(φ_center) * ")"
    end

    λ_summary = "$(TX)  extent $(prettysummary(extent_λ)) degrees"
    φ_summary = "$(TX)  extent $(prettysummary(extent_φ)) degrees"
    z_summary = domain_summary(TZ(), "z", z₁, z₂)

    longest = max(length(λ_summary), length(φ_summary), length(z_summary))

    padding_λ = length(λ_summary) < longest ? " "^(longest - length(λ_summary)) : ""
    padding_φ = length(φ_summary) < longest ? " "^(longest - length(φ_summary)) : ""

    λ_summary = "longitude: $(TX)  extent $(prettysummary(extent_λ)) degrees" * padding_λ *" " * coordinate_summary(rad2deg.(grid.Δxᶠᶠᵃ[1:Nx_face, 1:Ny_face] ./ grid.radius), "λ")
    φ_summary = "latitude:  $(TX)  extent $(prettysummary(extent_φ)) degrees" * padding_φ *" " * coordinate_summary(rad2deg.(grid.Δyᶠᶠᵃ[1:Nx_face, 1:Ny_face] ./ grid.radius), "φ")
    z_summary = "z:         " * dimension_summary(TZ(), "z", z₁, z₂, grid.Δzᵃᵃᶜ, longest - length(z_summary))

    if withsummary
        print(io, summary(grid), "\n")
    end

    return print(io, "├── ", center_str, "\n",
                     "├── ", λ_summary, "\n",
                     "├── ", φ_summary, "\n",
                     "└── ", z_summary)
end

@inline z_domain(grid::OrthogonalSphericalShellGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = domain(TZ, grid.Nz, grid.zᵃᵃᶠ)
@inline cpu_face_constructor_z(grid::ZRegOrthogonalSphericalShellGrid) = z_domain(grid)

function nodes(grid::OSSG, ℓx, ℓy, ℓz; reshape=false, with_halos=false)
    λ = λnodes(grid, ℓx, ℓy, ℓz; with_halos)
    φ = φnodes(grid, ℓx, ℓy, ℓz; with_halos)
    z = znodes(grid, ℓx, ℓy, ℓz; with_halos)

    if reshape
        # λ and φ are 2D arrays
        N = (size(λ)..., size(z)...)
        λ = Base.reshape(λ, N[1], Ν[2], 1)
        φ = Base.reshape(φ, N[1], N[2], 1)
        z = Base.reshape(z, 1, 1, N[3])
    end

    return (λ, φ, z)
end

@inline λnodes(grid::OSSG, ℓx::Face,   ℓy::Face, ; with_halos=false) = with_halos ? grid.λᶠᶠᵃ :
    view(grid.λᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Face,   ℓy::Center; with_halos=false) = with_halos ? grid.λᶠᶜᵃ :
    view(grid.λᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Center, ℓy::Face, ; with_halos=false) = with_halos ? grid.λᶜᶠᵃ :
    view(grid.λᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline λnodes(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) = with_halos ? grid.λᶜᶜᵃ :
    view(grid.λᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline φnodes(grid::OSSG, ℓx::Face,   ℓy::Face, ; with_halos=false) = with_halos ? grid.φᶠᶠᵃ :
    view(grid.φᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Face,   ℓy::Center; with_halos=false) = with_halos ? grid.φᶠᶜᵃ :
    view(grid.φᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Center, ℓy::Face, ; with_halos=false) = with_halos ? grid.φᶜᶠᵃ :
    view(grid.φᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline φnodes(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) = with_halos ? grid.φᶜᶜᵃ :
    view(grid.φᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline xnodes(grid::OSSG, ℓx, ℓy; with_halos=false) = grid.radius * deg2rad.(λnodes(grid, ℓx, ℓy; with_halos=with_halos)) .* hack_cosd.(φnodes(grid, ℓx, ℓy; with_halos=with_halos))
@inline ynodes(grid::OSSG, ℓx, ℓy; with_halos=false) = grid.radius * deg2rad.(φnodes(grid, ℓx, ℓy; with_halos=with_halos))

@inline znodes(grid::OSSG, ℓz::Face  ; with_halos=false) = with_halos ? grid.zᵃᵃᶠ :
    view(grid.zᵃᵃᶠ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))
@inline znodes(grid::OSSG, ℓz::Center; with_halos=false) = with_halos ? grid.zᵃᵃᶜ :
    view(grid.zᵃᵃᶜ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))

# convenience

"""
    λnodes(grid::OrthogonalSphericalShellGrid, ℓx, ℓy, ℓz, with_halos=false)

Return the positions over the interior nodes on a curvilinear `grid` in the ``λ``-direction
for the location `ℓλ`, `ℓφ`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

See [`znodes`](@ref) for examples.
"""
@inline λnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = λnodes(grid, ℓx, ℓy; with_halos)

"""
    φnodes(grid::AbstractCurvilinearGrid, ℓx, ℓy, ℓz, with_halos=false)

Return the positions over the interior nodes on a curvilinear `grid` in the ``φ``-direction
for the location `ℓλ`, `ℓφ`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

See [`znodes`](@ref) for examples.
"""
@inline φnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = φnodes(grid, ℓx, ℓy; with_halos)
@inline znodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = znodes(grid, ℓz    ; with_halos)
@inline xnodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = xnodes(grid, ℓx, ℓy; with_halos)
@inline ynodes(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = ynodes(grid, ℓx, ℓy; with_halos)

@inline λnode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.λᶜᶜᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.λᶠᶜᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.λᶜᶠᵃ[i, j]
@inline λnode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.λᶠᶠᵃ[i, j]

@inline φnode(i, j, grid::OSSG, ::Center, ::Center) = @inbounds grid.φᶜᶜᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Face  , ::Center) = @inbounds grid.φᶠᶜᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Center, ::Face  ) = @inbounds grid.φᶜᶠᵃ[i, j]
@inline φnode(i, j, grid::OSSG, ::Face  , ::Face  ) = @inbounds grid.φᶠᶠᵃ[i, j]

@inline xnode(i, j, grid::OSSG, ℓx, ℓy) = grid.radius * deg2rad(λnode(i, j, grid, ℓx, ℓy)) * hack_cosd((φnode(i, j, grid, ℓx, ℓy)))
@inline ynode(i, j, grid::OSSG, ℓx, ℓy) = grid.radius * deg2rad(φnode(i, j, grid, ℓx, ℓy))

@inline znode(k, grid::OSSG, ::Center) = @inbounds grid.zᵃᵃᶜ[k]
@inline znode(k, grid::OSSG, ::Face  ) = @inbounds grid.zᵃᵃᶠ[k]

# convenience
@inline λnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline φnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)
@inline znode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)
@inline xnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = xnode(i, j, grid, ℓx, ℓy)
@inline ynode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = ynode(i, j, grid, ℓx, ℓy)

# Definitions for node
@inline ξnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = λnode(i, j, grid, ℓx, ℓy)
@inline ηnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = φnode(i, j, grid, ℓx, ℓy)
@inline rnode(i, j, k, grid::OSSG, ℓx, ℓy, ℓz) = znode(k, grid, ℓz)

ξname(::OSSG) = :λ
ηname(::OSSG) = :φ
rname(::OSSG) = :z

#####
##### Grid spacings in x, y, z (in meters)
#####

@inline xspacings(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) =
    with_halos ? grid.Δxᶜᶜᵃ : view(grid.Δxᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::OSSG, ℓx::Face  , ℓy::Center; with_halos=false) =
    with_halos ? grid.Δxᶠᶜᵃ : view(grid.Δxᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::OSSG, ℓx::Center, ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δxᶜᶠᵃ : view(grid.Δxᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline xspacings(grid::OSSG, ℓx::Face  , ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δxᶠᶠᵃ : view(grid.Δxᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline yspacings(grid::OSSG, ℓx::Center, ℓy::Center; with_halos=false) =
    with_halos ? grid.Δyᶜᶜᵃ : view(grid.Δyᶜᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline yspacings(grid::OSSG, ℓx::Face  , ℓy::Center; with_halos=false) =
    with_halos ? grid.Δyᶠᶜᵃ : view(grid.Δyᶠᶜᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline yspacings(grid::OSSG, ℓx::Center, ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δyᶜᶠᵃ : view(grid.Δyᶜᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))
@inline yspacings(grid::OSSG, ℓx::Face  , ℓy::Face  ; with_halos=false) =
    with_halos ? grid.Δyᶠᶠᵃ : view(grid.Δyᶠᶠᵃ, interior_indices(ℓx, topology(grid, 1)(), grid.Nx), interior_indices(ℓy, topology(grid, 2)(), grid.Ny))

@inline zspacings(grid::OSSG,     ℓz::Center; with_halos=false) = with_halos ? grid.Δzᵃᵃᶜ :
    view(grid.Δzᵃᵃᶜ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))
@inline zspacings(grid::ZRegOSSG, ℓz::Center; with_halos=false) = grid.Δzᵃᵃᶜ
@inline zspacings(grid::OSSG,     ℓz::Face;   with_halos=false) = with_halos ? grid.Δzᵃᵃᶠ :
    view(grid.Δzᵃᵃᶠ, interior_indices(ℓz, topology(grid, 3)(), grid.Nz))
@inline zspacings(grid::ZRegOSSG, ℓz::Face;   with_halos=false) = grid.Δzᵃᵃᶠ

"""
    xspacings(grid, ℓx, ℓy, ℓz; with_halos=true)

Return the spacings over the interior nodes on `grid` in the ``x``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest xspacings
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(8, 15, 10), longitude=(-20, 60), latitude=(-10, 50), z=(-100, 0));

julia> xspacings(grid, Center(), Face(), Center())
16-element view(OffsetArray(::Vector{Float64}, -2:18), 1:16) with eltype Float64:
      1.0950562585518518e6
      1.1058578920188267e6
      1.1112718969963323e6
      1.1112718969963323e6
      1.1058578920188267e6
      1.0950562585518518e6
      1.0789196210678827e6
      1.0575265956426917e6
      1.0309814069457315e6
 999413.38046802
 962976.3124613502
 921847.720658409
 876227.979424229
 826339.3435524226
 772424.8654621692
 714747.2110712599
 
```
"""
@inline xspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = xspacings(grid, ℓx, ℓy; with_halos)

"""
    yspacings(grid, ℓx, ℓy, ℓz; with_halos=true)

Return the spacings over the interior nodes on `grid` in the ``y``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest yspacings
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(20, 15, 10), longitude=(0, 20), latitude=(-15, 15), z=(-100, 0));

julia> yspacings(grid, Center(), Center(), Center())    
222389.85328911748

```
"""
@inline yspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = yspacings(grid, ℓx, ℓy; with_halos)

"""
    zspacings(grid, ℓx, ℓy, ℓz; with_halos=true)

Return the spacings over the interior nodes on `grid` in the ``z``-direction for the location `ℓx`,
`ℓy`, `ℓz`. For `Bounded` directions, `Face` nodes include the boundary points.

```jldoctest zspacings
julia> using Oceananigans

julia> grid = LatitudeLongitudeGrid(size=(20, 15, 10), longitude=(0, 20), latitude=(-15, 15), z=(-100, 0));

julia> zspacings(grid, Center(), Center(), Center())
10.0

```
"""
@inline zspacings(grid::OSSG, ℓx, ℓy, ℓz; with_halos=false) = zspacings(grid, ℓz; with_halos)
