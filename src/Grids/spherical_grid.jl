#####
##### SphericalGrid: a horizontally curvilinear grid on the sphere whose coordinate lines need not be orthogonal
##### (Oceananigans issue #5808). It carries the coordinate, length and area arrays of `OrthogonalSphericalShellGrid`
##### plus the sine and cosine of the angle α between the local `x` and `y` coordinate lines at the four horizontal
##### staggers. With α = 90° everywhere it describes the same geometry as an `OrthogonalSphericalShellGrid`.
#####

"""
    AbstractShell

Abstract supertype for the radial treatment of a [`SphericalGrid`](@ref).
"""
abstract type AbstractShell end

"""
    ThinShell()

The thin-shell (shallow) approximation: horizontal lengths and areas are evaluated on the sphere of radius
`grid.radius` and do not vary with height. This is the behavior of `OrthogonalSphericalShellGrid`.
"""
struct ThinShell <: AbstractShell end

"""
    DeepShell()

Deep-shell geometry, in which horizontal lengths scale as `r / R` and areas as `(r / R)²`. Declared so that
the `shell` type parameter has its intended range; it is not implemented yet and every constructor rejects it.
"""
struct DeepShell <: AbstractShell end

"""
    SphericalGrid{FT, TX, TY, TZ, Z, Map, Shell, CC, FC, CF, FF, Arch, FT2, SZ}

A horizontally curvilinear grid on the sphere whose `x` and `y` coordinate lines may meet at any angle. It stores
the same coordinates (`λ`, `φ` in degrees), lengths (`Δx`, `Δy`, meters) and areas (`Az`, square meters) as
[`OrthogonalSphericalShellGrid`](@ref), at the four horizontal staggers `ᶜᶜᵃ`, `ᶠᶜᵃ`, `ᶜᶠᵃ`, `ᶠᶠᵃ`, plus

- `cosα` and `sinα` at the same four staggers, where `α` is the angle measured counterclockwise (seen from outside
  the sphere) from the local `+x` coordinate direction to the local `+y` coordinate direction. Both are stored,
  because `sinα = √(1 - cos²α)` loses precision near orthogonality;
- `shell`, a [`ThinShell`](@ref) or [`DeepShell`](@ref) (the latter is not implemented yet);
- `mapping`, the provenance of the horizontal coordinates (for example an [`EquiangularGnomonicMapping`](@ref)).
  The geometry never reads it; it records how the grid was made.

Primitive stencils (`δx`, `ℑx`, ...) on a `SphericalGrid` are plain index arithmetic, and connectivity (for example
across cubed-sphere panel seams) lives only in halo filling. Code that assumes orthogonal coordinate lines must check
[`isorthogonal`](@ref), which is `false` for a `SphericalGrid`.
"""
struct SphericalGrid{FT, TX, TY, TZ, Z, Map, Shell, CC, FC, CF, FF, Arch, FT2, SZ} <:
       AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Z, Arch, SZ}
    architecture :: Arch
       Nx :: Int
       Ny :: Int
       Nz :: Int
       Hx :: Int
       Hy :: Int
       Hz :: Int
       Lz :: FT2
     λᶜᶜᵃ :: CC
     λᶠᶜᵃ :: FC
     λᶜᶠᵃ :: CF
     λᶠᶠᵃ :: FF
     φᶜᶜᵃ :: CC
     φᶠᶜᵃ :: FC
     φᶜᶠᵃ :: CF
     φᶠᶠᵃ :: FF
        z :: Z
    Δxᶜᶜᵃ :: CC
    Δxᶠᶜᵃ :: FC
    Δxᶜᶠᵃ :: CF
    Δxᶠᶠᵃ :: FF
    Δyᶜᶜᵃ :: CC
    Δyᶠᶜᵃ :: FC
    Δyᶜᶠᵃ :: CF
    Δyᶠᶠᵃ :: FF
    Azᶜᶜᵃ :: CC
    Azᶠᶜᵃ :: FC
    Azᶜᶠᵃ :: CF
    Azᶠᶠᵃ :: FF
  cosαᶜᶜᵃ :: CC
  cosαᶠᶜᵃ :: FC
  cosαᶜᶠᵃ :: CF
  cosαᶠᶠᵃ :: FF
  sinαᶜᶜᵃ :: CC
  sinαᶠᶜᵃ :: FC
  sinαᶜᶠᵃ :: CF
  sinαᶠᶠᵃ :: FF
   radius :: FT2
    shell :: Shell
  mapping :: Map
end

function SphericalGrid{FT, TX, TY, TZ, SZ}(architecture::Arch, Nx, Ny, Nz, Hx, Hy, Hz, Lz::FT2,
                                           λᶜᶜᵃ::CC, λᶠᶜᵃ::FC, λᶜᶠᵃ::CF, λᶠᶠᵃ::FF,
                                           φᶜᶜᵃ::CC, φᶠᶜᵃ::FC, φᶜᶠᵃ::CF, φᶠᶠᵃ::FF, z::Z,
                                           Δxᶜᶜᵃ::CC, Δxᶠᶜᵃ::FC, Δxᶜᶠᵃ::CF, Δxᶠᶠᵃ::FF,
                                           Δyᶜᶜᵃ::CC, Δyᶠᶜᵃ::FC, Δyᶜᶠᵃ::CF, Δyᶠᶠᵃ::FF,
                                           Azᶜᶜᵃ::CC, Azᶠᶜᵃ::FC, Azᶜᶠᵃ::CF, Azᶠᶠᵃ::FF,
                                           cosαᶜᶜᵃ::CC, cosαᶠᶜᵃ::FC, cosαᶜᶠᵃ::CF, cosαᶠᶠᵃ::FF,
                                           sinαᶜᶜᵃ::CC, sinαᶠᶜᵃ::FC, sinαᶜᶠᵃ::CF, sinαᶠᶠᵃ::FF,
                                           radius::FT2, shell::Shell, mapping::Map) where {FT, TX, TY, TZ, SZ, Arch, FT2,
                                                                                         CC, FC, CF, FF, Z, Shell, Map}

    return SphericalGrid{FT, TX, TY, TZ, Z, Map, Shell, CC, FC, CF, FF, Arch, FT2, SZ}(architecture,
                                                                                       Nx, Ny, Nz, Hx, Hy, Hz, Lz,
                                                                                       λᶜᶜᵃ, λᶠᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                                                                       φᶜᶜᵃ, φᶠᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, z,
                                                                                       Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                                                       Δyᶜᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                                                                       Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                                                       cosαᶜᶜᵃ, cosαᶠᶜᵃ, cosαᶜᶠᵃ, cosαᶠᶠᵃ,
                                                                                       sinαᶜᶜᵃ, sinαᶠᶜᵃ, sinαᶜᶠᵃ, sinαᶠᶠᵃ,
                                                                                       radius, shell, mapping)
end

SphericalGrid{TX, TY, TZ}(architecture, Nx, Ny, Nz, Hx, Hy, Hz, Lz::FT, args...) where {TX, TY, TZ, FT} =
    SphericalGrid{FT, TX, TY, TZ, typeof(GridSize(Nx, Ny, Nz, Hx, Hy, Hz))}(architecture, Nx, Ny, Nz, Hx, Hy, Hz, Lz, args...)

const ZRegSphericalGrid = SphericalGrid{<:Any, <:Any, <:Any, <:Any, <:RegularVerticalCoordinate}

"""
    isorthogonal(grid)

`true` if the horizontal coordinate lines of `grid` are guaranteed to meet at right angles. Code that relies on
orthogonal metrics (for example the rotation between intrinsic and extrinsic vector components, or the stock model
tendencies) must check this trait and fail loudly when it is `false`.
"""
isorthogonal(::AbstractGrid) = true
isorthogonal(::SphericalGrid) = false

"""
    validate_orthogonal_grid(grid, caller)

Throw an `ArgumentError` naming `caller` if `isorthogonal(grid)` is `false`.
"""
function validate_orthogonal_grid(grid, caller)
    isorthogonal(grid) && return nothing
    throw(ArgumentError("$caller assumes orthogonal horizontal coordinate lines, but the grid " *
                        "($(summary(grid))) is not guaranteed to be orthogonal; see `isorthogonal`."))
end
