using Oceananigans.Grids: cpu_face_constructor_z, dimension_summary, domain, size_summary
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid, rotated_basis_angle
using Oceananigans.Utils: prettysummary

# Region 1 is the Yin panel and region 2 is the Yang panel
struct YinYangPartition <: AbstractPartition end

Base.length(::YinYangPartition) = 2
Base.summary(::YinYangPartition) = "YinYangPartition with (1: Yin, 2: Yang)"

struct YinYangConnectivity{M, I} <: AbstractConnectivity
    overlap :: Int               # cells by which each panel extends beyond its 270° × 90° core on every side
    core_size :: Tuple{Int, Int} # (Nλ, Nφ): cells in the core of each panel
    stencil_width :: Int         # width s of the s × s interpolation stencils
    masks :: M                   # `YinYangMasks` shared by both panels
    interpolation :: I           # `YinYangInterpolation` shared by both panels
end

Grids.on_architecture(arch, c::YinYangConnectivity) =
    YinYangConnectivity(c.overlap, c.core_size, c.stencil_width,
                        on_architecture(arch, c.masks),
                        on_architecture(arch, c.interpolation))

# Every edge of each panel is filled from the other panel
struct YinYangRegionalConnectivity <: AbstractConnectivity end

const yin_yang_regional_connectivity = (; west = YinYangRegionalConnectivity(), east = YinYangRegionalConnectivity(),
                                          south = YinYangRegionalConnectivity(), north = YinYangRegionalConnectivity())

@inline Utils.getregion(::YinYangConnectivity, r)  = yin_yang_regional_connectivity
@inline Utils._getregion(::YinYangConnectivity, r) = yin_yang_regional_connectivity

function Base.summary(c::YinYangConnectivity)
    Nλ, Nφ = c.core_size
    return string("YinYangConnectivity with ", Nλ, "×", Nφ, " core, ", c.overlap, "-cell overlap and ",
                  c.stencil_width, "×", c.stencil_width, " interpolation stencils")
end

Base.show(io::IO, c::YinYangConnectivity) = print(io, summary(c))

const YinYangGrid{FT, TX, TY, TZ, CZ} = MultiRegionGrid{FT, TX, TY, TZ, CZ, <:YinYangPartition}

const ImmersedYinYangGrid{FT, TX, TY, TZ, CZ} =
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:YinYangGrid{FT, TX, TY, TZ, CZ}, <:Any, <:Any, <:Any, <:AbstractSerialArchitecture}

const YinYangGridOfSomeKind{FT, TX, TY, TZ, CZ} =
    Union{YinYangGrid{FT, TX, TY, TZ, CZ}, ImmersedYinYangGrid{FT, TX, TY, TZ, CZ}}

const YinYangField{LX, LY, LZ} =
    Union{Field{LX, LY, LZ, <:Any, <:YinYangGrid},
          Field{LX, LY, LZ, <:Any, <:ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:YinYangGrid}}}

# The default overlap ⌈(2H + s) / 4⌉ for a horizontal halo H and stencil width s. It exceeds the ⌈s/4⌉ cells
# that the interpolation alone needs, so that the donor nodes lie an advection stencil away from their panel's halo.
default_yin_yang_overlap(halo, stencil_width = 4) = cld(2 * max(halo[1], halo[2]) + stencil_width, 4)

"""
    YinYangGrid(arch = CPU(), FT = Oceananigans.defaults.FloatType;
                size,
                z,
                halo = (3, 3, 3),
                stencil_width = 4,
                overlap = default_yin_yang_overlap(halo, stencil_width),
                radius = Oceananigans.defaults.planet_radius)

Return the Yin-Yang grid of [Kageyama and Sato (2004)](@cite KageyamaSato2004): a two-region `MultiRegionGrid`
of two identical `RotatedLatitudeLongitudeGrid` panels, Yin (region 1) and Yang (region 2), whose frames are
related by the map `(x, y, z) ↦ (-x, z, y)` of Cartesian coordinates. Yin's frame is the geographic frame.
Each panel integrates its interior cells and fills its horizontal halos by interpolation from the interior
of the other panel.

Keyword arguments
=================

- `size`: `(Nλ, Nφ, Nz)`, the number of cells in the 270° × 90° core of each panel and in the vertical,
          with `Nλ = 3Nφ`.
- `z`: the vertical coordinate.
- `halo`: the halo size.
- `stencil_width`: the width `s` of the `s × s` Lagrange stencils that interpolate the halos: `4` (cubic) or
                   `2` (bilinear). The halos are interpolated point values: they are not bounded by the donor
                   values, and they are second-order accurate for cell averages.
- `overlap`: the number of cells by which each panel extends beyond its core on every side.
- `radius`: the radius of the sphere.

Reductions over both horizontal dimensions count every point of the sphere once. `sum`, `mean`, `Integral`
and `Average` weight each point by the fraction of its cell that the panel owns (see `owned_weights`),
lumped from the adjacent cells at staggered locations, and `maximum`, `minimum`, `all`, `any` and `prod`
consider only owned points.

Example
=======

```jldoctest yinyang
using Oceananigans

grid = YinYangGrid(size=(48, 16, 4), z=(-1000, 0))

# output
54×22×4 YinYangGrid{Float64, FullyConnected, FullyConnected, Bounded} on CPU with 3×3×3 halo
├── core: 48×16 cells per panel spanning 270° × 90° with Δλ′ = Δφ′ = 5.625°
├── overlap: 3 cells (16.875°) beyond the core on every side, 4×4 interpolation stencils
├── panels: λ′ ∈ [-151.875, 151.875], φ′ ∈ [-61.875, 61.875]
├── Yin (region 1):  panel frame = geographic frame
├── Yang (region 2): (x, y, z) ↦ (-x, z, y), panel (λ′, φ′) = (0, 0) at (λ, φ) = (180, 0)
└── z: Bounded  z ∈ [-1000.0, 0.0] regularly spaced with Δz=250.0
```
"""
function YinYangGrid(arch::AbstractArchitecture = CPU(),
                     FT::DataType = Oceananigans.defaults.FloatType;
                     size,
                     z,
                     halo = (3, 3, 3),
                     stencil_width = 4,
                     overlap = default_yin_yang_overlap(halo, stencil_width),
                     radius = Oceananigans.defaults.planet_radius)

    Nλ, Nφ, Nz = size
    Hλ, Hφ = halo[1], halo[2]

    Nλ == 3Nφ || throw(ArgumentError("YinYangGrid requires size = (3Nφ, Nφ, Nz); got size = $size."))

    # The panels, including the extra halo cell of `RotatedLatitudeLongitudeGrid` metrics, must not reach their poles
    2 * (overlap + Hφ + 1) ≤ Nφ ||
        throw(ArgumentError("YinYangGrid panels with overlap = $overlap and halo = $halo reach the poles: increase Nφ = $Nφ."))

    λᵉ, φᵉ = yin_yang_panel_extents((Nλ, Nφ), overlap)
    panel_size = (Nλ + 2overlap, Nφ + 2overlap, Nz)

    # The tables are computed once on the CPU in Float64, from the panel coordinates shared by both panels
    panel_faces = yin_yang_panel_face_coordinates((Nλ, Nφ), overlap)
    masks = YinYangMasks(arch, FT, panel_faces, (Hλ, Hφ))
    interpolation = YinYangInterpolation(arch, FT, panel_faces, (Hλ, Hφ), stencil_width)
    connectivity = YinYangConnectivity(overlap, (Nλ, Nφ), stencil_width, masks, interpolation)
    topology = (FullyConnected, FullyConnected, Bounded)
    latitude = (-φᵉ, φᵉ)

    yin = RotatedLatitudeLongitudeGrid(arch, FT; size = panel_size, z, halo, radius, topology, latitude,
                                       longitude = (-λᵉ, λᵉ),
                                       north_pole = (0, 90))

    # Yang's unrotated longitude is its panel longitude plus 90°
    yang = RotatedLatitudeLongitudeGrid(arch, FT; size = panel_size, z, halo, radius, topology, latitude,
                                        longitude = (90 - λᵉ, 90 + λᵉ),
                                        north_pole = (90, 0))

    region_grids = MultiRegionObject((yin, yang))
    CZ = typeof(yin.z)

    return MultiRegionGrid{FT, topology..., CZ}(arch, YinYangPartition(), connectivity, region_grids)
end

# The half-widths `(λᵉ, φᵉ)` (degrees) of a Yin-Yang panel with `core_size = (Nλ, Nφ)` and `overlap`,
# in Float64 so that both panels get the same spacing in any precision.
yin_yang_panel_extents((Nλ, Nφ), overlap) = (135 + overlap * 90 / Nφ, 45 + overlap * 90 / Nφ)

# The uniformly spaced panel coordinates `(λᶠ, φᶠ)` (degrees) of the cell faces of a Yin-Yang panel.
function yin_yang_panel_face_coordinates((Nλ, Nφ), overlap)
    λᵉ, φᵉ = yin_yang_panel_extents((Nλ, Nφ), overlap)
    λᶠ = range(-λᵉ, λᵉ, length = Nλ + 2overlap + 1)
    φᶠ = range(-φᵉ, φᵉ, length = Nφ + 2overlap + 1)
    return λᶠ, φᶠ
end

function Grids.with_halo(new_halo, grid::YinYangGrid)
    connectivity = grid.connectivity
    Nλ, Nφ = connectivity.core_size
    Nz = size(grid, 3)
    yin = first(grid)
    z = cpu_face_constructor_z(yin)

    return YinYangGrid(architecture(grid), eltype(grid);
                       size = (Nλ, Nφ, Nz), z,
                       halo = new_halo,
                       stencil_width = connectivity.stencil_width,
                       overlap = connectivity.overlap,
                       radius = yin.radius)
end

yin_yang_connectivity(grid::YinYangGrid) = grid.connectivity
yin_yang_connectivity(grid::ImmersedYinYangGrid) = grid.underlying_grid.connectivity

yin_yang_masks(grid::YinYangGridOfSomeKind) = yin_yang_connectivity(grid).masks
yin_yang_interpolation(grid::YinYangGridOfSomeKind) = yin_yang_connectivity(grid).interpolation

"""
$(TYPEDSIGNATURES)

Return the fractions `w ∈ [0, 1]` of the area of the cells of the Yin-Yang `grid` that each panel owns,
as a `Field{Center, Center, Nothing}` whose two regions share one array, with `w = 0` in the halos. The
owned cells of the two panels tile the sphere: `Σ w Az` over the interior cells of both panels is `4πR²`
to rounding error.

The weights also give the seam budget of a flux-form scheme. A panel that updates every cell with the
divergence of its face fluxes `F` changes its owned content `Σ w V c` by `Σ F Δw` over its horizontal
faces, plus the boundary fluxes and the sources, with `Δw = w[i, j] - w[i-1, j]` across an x-face and
`Δw = w[i, j] - w[i, j-1]` across a y-face. Only the faces next to the curve that divides the sphere
between the panels have `Δw ≠ 0`, and the sums of the two panels cancel where the panels agree on the
flux through that curve.

```jldoctest
using Oceananigans
using Oceananigans.MultiRegion: owned_weights

grid = YinYangGrid(size=(48, 16, 1), z=(-1, 0))
summary(owned_weights(grid))

# output
"54×22×1 Field{Center, Center, Nothing} reduced over dims = (3,) on YinYangGrid on CPU"
```
"""
function owned_weights(grid::YinYangGridOfSomeKind)
    w = yin_yang_masks(grid).cell_weights
    return Field((Center(), Center(), nothing), grid, MultiRegionObject((w, w)), nothing, (:, :, :))
end

# The angle (radians) from the basis of the panel in whose frame `(λ, φ)` (degrees) are given to the basis of
# the other panel, whose pole lies at (90°, 0) in either frame
@inline yin_yang_rotation_angle(λ, φ) = rotated_basis_angle(λ, φ, 90, 0)

#####
##### Show
#####

Grids.grid_name(::YinYangGridOfSomeKind) = "YinYangGrid"

function Base.summary(grid::YinYangGridOfSomeKind{FT}) where FT
    nTX, nTY, nTZ = map(nameof, topology(grid))
    return string(size_summary(size(grid)),
                  " YinYangGrid{$FT, $nTX, $nTY, $nTZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function Base.show(io::IO, grid::YinYangGrid)
    connectivity = grid.connectivity
    Nλ, Nφ = connectivity.core_size
    δ = connectivity.overlap
    Δ = 90 / Nφ
    λᵉ, φᵉ = yin_yang_panel_extents(connectivity.core_size, δ)
    s = connectivity.stencil_width

    yin = first(grid)
    TZ = topology(yin, 3)
    Ωz = domain(TZ(), size(yin, 3), yin.z.cᵃᵃᶠ)

    return print(io, summary(grid), "\n",
                 "├── core: ", Nλ, "×", Nφ, " cells per panel spanning 270° × 90° with Δλ′ = Δφ′ = ", prettysummary(Δ), "°\n",
                 "├── overlap: ", δ, " cells (", prettysummary(δ * Δ), "°) beyond the core on every side, ",
                             s, "×", s, " interpolation stencils\n",
                 "├── panels: λ′ ∈ [", prettysummary(-λᵉ), ", ", prettysummary(λᵉ), "], ",
                             "φ′ ∈ [", prettysummary(-φᵉ), ", ", prettysummary(φᵉ), "]\n",
                 "├── Yin (region 1):  panel frame = geographic frame\n",
                 "├── Yang (region 2): (x, y, z) ↦ (-x, z, y), panel (λ′, φ′) = (0, 0) at (λ, φ) = (180, 0)\n",
                 "└── z: ", dimension_summary(TZ(), "z", Ωz, yin.z, 0))
end
