using Oceananigans.Grids: cpu_face_constructor_z, dimension_summary, domain, size_summary
using Oceananigans.OrthogonalSphericalShellGrids: RotatedLatitudeLongitudeGrid, rotated_basis_angle
using Oceananigans.Utils: prettysummary

struct YinYangConnectivity{M, I, RM, RH, E} <: AbstractConnectivity
    overlap :: Int               # cells by which each panel extends beyond its 270° × 90° core on every side
    core_size :: Tuple{Int, Int} # (Nλ, Nφ): cells in the core of each panel
    stencil_width :: Int         # width s of the s × s interpolation stencils
    masks :: M                   # `YinYangMasks` shared by both panels
    interpolation :: I           # `YinYangInterpolation` shared by both panels
    region_masks :: RM           # the `YinYangMasks` of every region, cut from those of its panel
    region_halos :: RH           # the `YinYangRegionHalos` of every region
    exchange :: E                # `nothing`, or the `YinYangExchange` of this rank's region of a distributed grid
end

Grids.on_architecture(arch, c::YinYangConnectivity) =
    YinYangConnectivity(c.overlap, c.core_size, c.stencil_width,
                        on_architecture(arch, c.masks), on_architecture(arch, c.interpolation),
                        on_architecture(arch, c.region_masks), on_architecture(arch, c.region_halos), c.exchange)

Base.summary(c::YinYangConnectivity) =
    string("YinYangConnectivity with ", c.core_size[1], "×", c.core_size[2], " core, ", c.overlap, "-cell overlap and ",
           c.stencil_width, "×", c.stencil_width, " interpolation stencils")

Base.show(io::IO, c::YinYangConnectivity) = print(io, summary(c))

# Every edge of each panel is filled from the other panel
struct YinYangRegionalConnectivity <: AbstractConnectivity end

const yin_yang_regional_connectivity = (; west = YinYangRegionalConnectivity(), east = YinYangRegionalConnectivity(),
                                          south = YinYangRegionalConnectivity(), north = YinYangRegionalConnectivity())

@inline Utils.getregion(::YinYangConnectivity, r)  = yin_yang_regional_connectivity
@inline Utils._getregion(::YinYangConnectivity, r) = yin_yang_regional_connectivity

const YinYangGrid{FT, TX, TY, TZ, CZ} = MultiRegionGrid{FT, TX, TY, TZ, CZ, <:SubregionPartition{<:YinYangPartition}}

const ImmersedYinYangGrid{FT, TX, TY, TZ, CZ} =
    ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:YinYangGrid{FT, TX, TY, TZ, CZ}, <:Any, <:Any, <:Any, <:AbstractSerialArchitecture}

const YinYangGridOfSomeKind{FT, TX, TY, TZ, CZ} =
    Union{YinYangGrid{FT, TX, TY, TZ, CZ}, ImmersedYinYangGrid{FT, TX, TY, TZ, CZ}}

const YinYangField{LX, LY, LZ} = Field{LX, LY, LZ, <:Any, <:YinYangGridOfSomeKind}

"""
    YinYangGrid(arch = CPU(), FT = Oceananigans.defaults.FloatType;
                size, z, halo = (3, 3, 3), stencil_width = 4,
                overlap = cld(2 * max(halo[1], halo[2]) + stencil_width, 4),
                radius = Oceananigans.defaults.planet_radius, partition = Partition(1))

Return the Yin-Yang grid of [Kageyama and Sato (2004)](@cite KageyamaSato2004): a two-region `MultiRegionGrid`
of two identical `RotatedLatitudeLongitudeGrid` panels, Yin (region 1) and Yang (region 2), whose frames are
related by the map `(x, y, z) ↦ (-x, z, y)` of Cartesian coordinates. Yin's frame is the geographic frame.
Each panel integrates its interior cells and fills its horizontal halos by `stencil_width × stencil_width`
Lagrange interpolation (`4`: cubic, `2`: bilinear) from the interior of the other panel.

Keyword arguments
=================

- `size`: `(Nλ, Nφ, Nz)`, the cells in the 270° × 90° core of each panel and in the vertical, with `Nλ = 3Nφ`.
- `z`: the vertical coordinate.
- `halo`: the halo size.
- `stencil_width`: the width of the interpolation stencils.
- `overlap`: the cells by which each panel extends beyond its core on every side. The default keeps the
             donor nodes an advection stencil away from their panel's halo.
- `radius`: the radius of the sphere.
- `partition`: `Partition(px, py)` splits each panel into `px × py` regions, for `2 px py` regions in all;
               `MultiRegionPartition(yin, yang)` splits each panel by its own `Partition`.

Horizontal velocities are stored in each panel's basis: `set!(model; u, v)` takes geographic components
unless `intrinsic_velocities = true`, and `PrescribedVelocityFields` takes panel components. Reductions over both horizontal dimensions count every point of the
sphere once, weighting `sum`, `mean`, `Integral` and `Average` by `owned_weights` and restricting `maximum`,
`minimum`, `all`, `any` and `prod` to owned points.

Example
=======

```jldoctest yinyang
using Oceananigans

grid = YinYangGrid(size=(48, 16, 4), z=(-1000, 0))

# output
54×22×4 YinYangGrid{Float64, Connected, Connected, Bounded} on CPU with 3×3×3 halo
├── core: 48×16 cells per panel spanning 270° × 90° with Δλ′ = Δφ′ = 5.625°
├── overlap: 3 cells (16.875°) beyond the core on every side, 4×4 interpolation stencils
├── panels: λ′ ∈ [-151.875, 151.875], φ′ ∈ [-61.875, 61.875]
├── Yin (region 1):  panel frame = geographic frame
├── Yang (region 2): (x, y, z) ↦ (-x, z, y), panel (λ′, φ′) = (0, 0) at (λ, φ) = (180, 0)
└── z: Bounded  z ∈ [-1000.0, 0.0] regularly spaced with Δz=250.0
```

References
==========
* Kageyama, A., & Sato, T. (2004). "'Yin-Yang grid': An overset grid in spherical geometry."
  Geochemistry, Geophysics, Geosystems, 5(9), Q09005.
"""
function YinYangGrid(arch::AbstractArchitecture = CPU(),
                     FT::DataType = Oceananigans.defaults.FloatType;
                     size,
                     z,
                     halo = (3, 3, 3),
                     stencil_width = 4,
                     overlap = cld(2 * max(halo[1], halo[2]) + stencil_width, 4),
                     radius = Oceananigans.defaults.planet_radius,
                     partition = Partition(1))

    Nλ, Nφ, Nz = size
    Nλ == 3Nφ || throw(ArgumentError("YinYangGrid requires size = (3Nφ, Nφ, Nz); got size = $size."))

    # The panels, including the extra halo cell of `RotatedLatitudeLongitudeGrid` metrics, must not reach their poles
    2 * (overlap + halo[2] + 1) ≤ Nφ ||
        throw(ArgumentError("YinYangGrid panels with overlap = $overlap and halo = $halo reach the poles: increase Nφ = $Nφ."))

    # The tables are computed once on the CPU in Float64, from the panel coordinates shared by both panels
    panel_faces = yin_yang_panel_face_coordinates((Nλ, Nφ), overlap)
    panel_size = (Nλ + 2overlap, Nφ + 2overlap)
    partition = yin_yang_partition(arch, partition, panel_size)
    layout = partition.layout

    # A distributed grid holds only the region of its rank, and exchanges the halos of the others
    masks = YinYangMasks(CPU(), FT, panel_faces, halo[1:2])
    interpolation = YinYangInterpolation(CPU(), FT, panel_faces, halo[1:2], stencil_width)
    halos = [YinYangRegionHalos(layout, r, interpolation, panel_size, halo[1:2]) for r in eachindex(layout)]
    local_regions = yin_yang_local_regions(arch, layout)
    exchange = yin_yang_exchange(arch, halos)
    child = child_architecture(arch)

    region_masks = MultiRegionObject(Tuple(YinYangMasks(region_cell_weights(layout, r, masks.cell_weights, halo[1:2])) for r in local_regions))
    region_halos = MultiRegionObject(Tuple(halos[r] for r in local_regions))
    connectivity = YinYangConnectivity(overlap, (Nλ, Nφ), stencil_width, on_architecture(child, masks),
                                       on_architecture(child, interpolation),
                                       on_architecture(child, region_masks), on_architecture(child, region_halos),
                                       on_architecture(child, exchange))

    λᶠ, φᶠ = panel_faces
    topology = (TwoSided{Connected, Connected}, TwoSided{Connected, Connected}, Bounded)

    # Yang's unrotated longitude is its panel longitude plus 90°
    function region_grid(r)
        (ox, oy), (nx, ny) = layout[r].offsets, layout[r].size
        λ₀ = layout[r].region == 1 ? 0 : 90
        north_pole = layout[r].region == 1 ? (0, 90) : (90, 0)
        longitude = (λ₀ + λᶠ[ox + 1], λ₀ + λᶠ[ox + nx + 1])
        latitude = (φᶠ[oy + 1], φᶠ[oy + ny + 1])
        return RotatedLatitudeLongitudeGrid(child, FT; size = (nx, ny, Nz), z, halo, radius, topology,
                                            latitude, longitude, north_pole)
    end

    region_grids = MultiRegionObject(Tuple(region_grid(r) for r in local_regions))
    yin = first(region_grids.regional_objects)

    return MultiRegionGrid{FT, topology..., typeof(yin.z)}(arch, partition, connectivity, region_grids)
end

yin_yang_partition(arch, p, panel_size) = yin_yang_partition(p, panel_size)
yin_yang_partition(p::Partition, panel_size) = yin_yang_partition(MultiRegionPartition(p, p), panel_size)
yin_yang_partition(p::SubregionPartition, panel_size) = p
yin_yang_local_regions(arch, layout) = eachindex(layout)
yin_yang_exchange(arch, halos) = nothing

function yin_yang_partition(p::MultiRegionPartition, panel_size)
    length(p.partitions) == 2 || throw(ArgumentError("A YinYangGrid has 2 regions; got $(length(p.partitions)) partitions."))
    return SubregionPartition(YinYangPartition(), region_splits(p), panel_size)
end

# The uniformly spaced panel coordinates `(λᶠ, φᶠ)` (degrees) of the cell faces of a Yin-Yang panel with
# `core_size = (Nλ, Nφ)` and `overlap`, in Float64 so that both panels get the same spacing in any precision.
function yin_yang_panel_face_coordinates((Nλ, Nφ), overlap)
    λᵉ, φᵉ = 135 + overlap * 90 / Nφ, 45 + overlap * 90 / Nφ
    return range(-λᵉ, λᵉ, length = Nλ + 2overlap + 1), range(-φᵉ, φᵉ, length = Nφ + 2overlap + 1)
end

function Grids.with_halo(new_halo, grid::YinYangGrid)
    c = grid.connectivity
    yin = first(grid)
    return YinYangGrid(architecture(grid), eltype(grid); size = (c.core_size..., size(grid, 3)), halo = new_halo,
                       z = cpu_face_constructor_z(yin), stencil_width = c.stencil_width, overlap = c.overlap, radius = yin.radius,
                       partition = grid.partition)
end

yin_yang_connectivity(grid::YinYangGrid) = grid.connectivity
yin_yang_connectivity(grid::ImmersedYinYangGrid) = grid.underlying_grid.connectivity

yin_yang_masks(grid::YinYangGridOfSomeKind) = yin_yang_connectivity(grid).masks
yin_yang_interpolation(grid::YinYangGridOfSomeKind) = yin_yang_connectivity(grid).interpolation
yin_yang_region_halos(grid::YinYangGridOfSomeKind) = yin_yang_connectivity(grid).region_halos

# The fractions `w ∈ [0, 1]` of the area of each cell that its panel owns, with `w = 0` in the panel halos. Halos at
# a split edge hold the sibling's values, which Face weights interpolate across; reductions cover interiors only.
# The owned cells of the two panels tile the sphere: `Σ w Az` over both panels is `4πR²`.
yin_yang_cell_weights(grid::YinYangGridOfSomeKind) =
    MultiRegionObject(map(masks -> masks.cell_weights, yin_yang_connectivity(grid).region_masks.regional_objects))

function owned_weights(grid::YinYangGridOfSomeKind)
    w = yin_yang_cell_weights(grid)
    return Field((Center(), Center(), nothing), grid, w, nothing, (:, :, :))
end

# The angle (radians) from the basis of the panel in whose frame `(λ, φ)` (degrees) are given to the basis of
# the other panel, whose pole lies at (90°, 0) in either frame
@inline yin_yang_rotation_angle(λ, φ) = rotated_basis_angle(λ, φ, 90, 0)

#####
##### Show
#####

Grids.grid_name(::YinYangGridOfSomeKind) = "YinYangGrid"

function Base.summary(grid::YinYangGridOfSomeKind{FT}) where FT
    nTX, nTY, nTZ = map(prettysummary, topology(grid))
    return string(size_summary(size(grid)), " YinYangGrid{$FT, $nTX, $nTY, $nTZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo")
end

function Base.show(io::IO, grid::YinYangGrid)
    c = grid.connectivity
    Nλ, Nφ = c.core_size
    Δ = 90 / Nφ
    λᶠ, φᶠ = yin_yang_panel_face_coordinates(c.core_size, c.overlap)

    yin = first(grid)
    TZ = topology(yin, 3)
    Ωz = domain(TZ(), size(yin, 3), yin.z.cᵃᵃᶠ)

    return print(io, summary(grid), "\n",
                 "├── core: ", Nλ, "×", Nφ, " cells per panel spanning 270° × 90° with Δλ′ = Δφ′ = ", prettysummary(Δ), "°\n",
                 "├── overlap: ", c.overlap, " cells (", prettysummary(c.overlap * Δ), "°) beyond the core on every side, ",
                             c.stencil_width, "×", c.stencil_width, " interpolation stencils\n",
                 "├── panels: λ′ ∈ [", prettysummary(λᶠ[1]), ", ", prettysummary(λᶠ[end]), "], ",
                             "φ′ ∈ [", prettysummary(φᶠ[1]), ", ", prettysummary(φᶠ[end]), "]\n",
                 "├── Yin (region 1):  panel frame = geographic frame\n",
                 "├── Yang (region 2): (x, y, z) ↦ (-x, z, y), panel (λ′, φ′) = (0, 0) at (λ, φ) = (180, 0)\n",
                 "└── z: ", dimension_summary(TZ(), "z", Ωz, yin.z, 0))
end
