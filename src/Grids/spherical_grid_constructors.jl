#####
##### Spherical geometry on unit vectors (CPU construction only)
#####

@inline dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
@inline cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
@inline norm3(a) = sqrt(dot3(a, a))
@inline normalize3(a) = (n = norm3(a); (a[1] / n, a[2] / n, a[3] / n))
@inline add3(a, b) = (a[1] + b[1], a[2] + b[2], a[3] + b[3])

"""
    spherical_unit_vector(λ, φ)

Cartesian unit vector of the point with longitude `λ` and latitude `φ` (degrees).
"""
spherical_unit_vector(λ, φ) = (cosd(φ) * cosd(λ), cosd(φ) * sind(λ), sind(φ))

"""
    longitude_latitude(v)

Longitude and latitude (degrees) of the direction of the Cartesian vector `v`.
"""
longitude_latitude(v) = (atand(v[2], v[1]), atand(v[3], hypot(v[1], v[2])))

"""
    central_angle(a, b)

Angle (radians) subtended at the centre of the sphere by the unit vectors `a` and `b`.
"""
central_angle(a, b) = atan(norm3(cross3(a, b)), dot3(a, b))

"""
    spherical_triangle_area(a, b, c)

Area of the spherical triangle with unit-vector vertices `a`, `b`, `c` on the unit sphere
(Van Oosterom & Strackee, 1983).
"""
spherical_triangle_area(a, b, c) = 2 * atan(abs(dot3(a, cross3(b, c))), 1 + dot3(a, b) + dot3(b, c) + dot3(c, a))

spherical_quadrilateral_area(a, b, c, d) = spherical_triangle_area(a, b, c) + spherical_triangle_area(a, c, d)

great_circle_midpoint(a, b) = normalize3(add3(a, b))
quadrilateral_centroid(a, b, c, d) = normalize3(add3(add3(a, b), add3(c, d)))

# Unit tangent at `q` of the great circle through `a` and `b`, pointing from `a` towards `b`.
great_circle_tangent(a, b, q) = normalize3(cross3(cross3(a, b), q))

# cos α and sin α of the angle from `tx` to `ty`, counterclockwise seen from outside the sphere at `q`.
coordinate_angle(tx, ty, q) = (dot3(tx, ty), dot3(cross3(tx, ty), q))

#####
##### Equiangular gnomonic map
#####

"""
    EquiangularGnomonicMapping(panel, rotation, ξᶠᵃᵃ, ηᵃᶠᵃ, ξᶜᵃᵃ, ηᵃᶜᵃ)

Provenance of a [`SphericalGrid`](@ref) panel of the gnomonic cubed sphere: the panel index, the `rotation` applied
to the reference panel (centred on the North Pole; `nothing` means no rotation), and the one-dimensional
computational angles `ξ`, `η` (radians, interior faces and centers) of its coordinate lines. Uniformly spaced angles
give the equiangular cubed sphere; other spacings (for example FV3's equal-edge grid) are gnomonic panels too.
"""
struct EquiangularGnomonicMapping{R, A}
    panel :: Int
    rotation :: R
    ξᶠᵃᵃ :: A
    ηᵃᶠᵃ :: A
    ξᶜᵃᵃ :: A
    ηᵃᶜᵃ :: A
end

Architectures.on_architecture(arch, m::EquiangularGnomonicMapping) =
    EquiangularGnomonicMapping(m.panel, m.rotation,
                               on_architecture(arch, m.ξᶠᵃᵃ), on_architecture(arch, m.ηᵃᶠᵃ),
                               on_architecture(arch, m.ξᶜᵃᵃ), on_architecture(arch, m.ηᵃᶜᵃ))

Adapt.adapt_structure(to, m::EquiangularGnomonicMapping) =
    EquiangularGnomonicMapping(m.panel, Adapt.adapt(to, m.rotation),
                               Adapt.adapt(to, m.ξᶠᵃᵃ), Adapt.adapt(to, m.ηᵃᶠᵃ),
                               Adapt.adapt(to, m.ξᶜᵃᵃ), Adapt.adapt(to, m.ηᵃᶜᵃ))

Base.summary(m::EquiangularGnomonicMapping) = string("EquiangularGnomonicMapping(panel ", m.panel, ")")

rotate(::Nothing, v) = v
rotate(rotation, v) = Tuple(rotation * collect(v))
inverse_rotate(::Nothing, v) = v
inverse_rotate(rotation, v) = Tuple(transpose(rotation) * collect(v))

"""
    equiangular_gnomonic_map(ξ, η, rotation = nothing)

Unit vector of the point with gnomonic angles `ξ`, `η` (radians, `|ξ|, |η| < π/2`) on the reference cube face
centred on the North Pole, `(tan ξ, tan η, 1) / √(1 + tan²ξ + tan²η)`, then rotated by `rotation`.
"""
function equiangular_gnomonic_map(ξ, η, rotation = nothing)
    X = tan(ξ)
    Y = tan(η)
    return rotate(rotation, normalize3((X, Y, one(X))))
end

"""
    inverse_equiangular_gnomonic_map(v, rotation = nothing)

Gnomonic angles `(ξ, η)` of the unit vector `v` on the panel obtained by `rotation` of the reference face.
"""
function inverse_equiangular_gnomonic_map(v, rotation = nothing)
    x, y, z = inverse_rotate(rotation, v)
    z > 0 || throw(ArgumentError("the point is not on the gnomonic panel's hemisphere"))
    return atan(x, z), atan(y, z)
end

#####
##### SphericalGrid constructors
#####

function allocate_horizontal_metric(FT, (LX, LY), topology, size, halo)
    metric = new_data(FT, CPU(), (LX, LY), topology, size, halo)
    fill!(metric, convert(FT, NaN))
    return metric
end

const horizontal_metric_names = (:λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ, :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ,
                                 :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
                                 :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ,
                                 :cosαᶜᶜᵃ, :cosαᶠᶜᵃ, :cosαᶜᶠᵃ, :cosαᶠᶠᵃ, :sinαᶜᶜᵃ, :sinαᶠᶜᵃ, :sinαᶜᶠᵃ, :sinαᶠᶠᵃ)

function horizontal_metric_location(name::Symbol)
    superscripts = collect(String(name))[end-2:end-1]
    LX = superscripts[1] == 'ᶜ' ? Center : Face
    LY = superscripts[2] == 'ᶜ' ? Center : Face
    return LX, LY
end

"""
    SphericalGrid(arch = CPU(), FT = Oceananigans.defaults.FloatType;
                  size, z,
                  vertices = nothing,
                  topology = (Bounded, Bounded, Bounded),
                  halo = (3, 3, 3),
                  radius = Oceananigans.defaults.planet_radius,
                  shell = ThinShell(),
                  mapping = nothing)

Return a [`SphericalGrid`](@ref) with `size = (Nx, Ny, Nz)` cells and vertical coordinate `z`.

If `vertices = (λ, φ)` is given, the horizontal geometry is computed from the cell-corner coordinates alone
(Oceananigans issue #5514). `λ` and `φ` are `(Nx + 1 + 2Hx) × (Ny + 1 + 2Hy)` arrays (degrees) of the corners
`i = 1 - Hx : Nx + 1 + Hx`, `j = 1 - Hy : Ny + 1 + Hy`, so that the halo geometry is determined too. With unit corner
vectors `P`:

- centers are the normalized sums of the four corners, `ᶠᶜ` and `ᶜᶠ` points the great-circle midpoints of the cell
  edges;
- lengths are great-circle distances between neighbouring nodes of the stagger's dual (for example `Δxᶠᶜᵃ` joins
  the centers `i - 1` and `i`, `Δxᶜᶠᵃ` the corners `i` and `i + 1`, `Δxᶜᶜᵃ` the midpoints of the west and east edges);
- areas are spherical-excess areas of the quadrilateral of dual nodes surrounding the point;
- `cosα`, `sinα` come from the tangents of the coordinate lines through the point.

A metric whose stencil reaches beyond the given corners (the outermost halo ring) copies its nearest computed value.
With `vertices = nothing` every horizontal array is allocated and filled with `NaN`, for the caller to fill.
"""
function SphericalGrid(arch::AbstractArchitecture = CPU(),
                       FT::DataType = Oceananigans.defaults.FloatType;
                       size,
                       z,
                       vertices = nothing,
                       topology = (Bounded, Bounded, Bounded),
                       halo = (3, 3, 3),
                       radius = Oceananigans.defaults.planet_radius,
                       shell = ThinShell(),
                       mapping = nothing)

    shell isa DeepShell && throw(ArgumentError("DeepShell is declared but not implemented yet; use ThinShell()."))

    topology = validate_topology(topology)
    TX, TY, TZ = topology
    Nx, Ny, Nz = size
    Hx, Hy, Hz = halo

    metrics = NamedTuple{horizontal_metric_names}(Tuple(allocate_horizontal_metric(FT, horizontal_metric_location(name),
                                                                                   (TX, TY), (Nx, Ny), (Hx, Hy))
                                                        for name in horizontal_metric_names))

    isnothing(vertices) || compute_spherical_metrics!(metrics, vertices, (Nx, Ny), (Hx, Hy), convert(FT, radius))

    Lz, zc = generate_coordinate(FT, topology, size, halo, z, :z, 3, CPU())

    grid = SphericalGrid{TX, TY, TZ}(CPU(), Nx, Ny, Nz, Hx, Hy, Hz, convert(FT, Lz),
                                     metrics.λᶜᶜᵃ, metrics.λᶠᶜᵃ, metrics.λᶜᶠᵃ, metrics.λᶠᶠᵃ,
                                     metrics.φᶜᶜᵃ, metrics.φᶠᶜᵃ, metrics.φᶜᶠᵃ, metrics.φᶠᶠᵃ, zc,
                                     metrics.Δxᶜᶜᵃ, metrics.Δxᶠᶜᵃ, metrics.Δxᶜᶠᵃ, metrics.Δxᶠᶠᵃ,
                                     metrics.Δyᶜᶜᵃ, metrics.Δyᶠᶜᵃ, metrics.Δyᶜᶠᵃ, metrics.Δyᶠᶠᵃ,
                                     metrics.Azᶜᶜᵃ, metrics.Azᶠᶜᵃ, metrics.Azᶜᶠᵃ, metrics.Azᶠᶠᵃ,
                                     metrics.cosαᶜᶜᵃ, metrics.cosαᶠᶜᵃ, metrics.cosαᶜᶠᵃ, metrics.cosαᶠᶠᵃ,
                                     metrics.sinαᶜᶜᵃ, metrics.sinαᶠᶜᵃ, metrics.sinαᶜᶠᵃ, metrics.sinαᶠᶠᵃ,
                                     convert(FT, radius), shell, mapping)

    return on_architecture(arch, grid)
end

function corner_unit_vectors(FT, (λ, φ), (Nx, Ny), (Hx, Hy))
    expected = (Nx + 1 + 2Hx, Ny + 1 + 2Hy)
    Base.size(λ) == expected && Base.size(φ) == expected ||
        throw(ArgumentError("vertices must be $(expected[1])×$(expected[2]) arrays covering the corners " *
                            "1-Hx:Nx+1+Hx × 1-Hy:Ny+1+Hy; got $(Base.size(λ)) and $(Base.size(φ))"))
    λ = parent(λ)
    φ = parent(φ)
    P = OffsetArray(Matrix{NTuple{3, FT}}(undef, expected...), 1-Hx:Nx+1+Hx, 1-Hy:Ny+1+Hy)
    for j in axes(P, 2), i in axes(P, 1)
        P[i, j] = spherical_unit_vector(λ[i+Hx, j+Hy], φ[i+Hx, j+Hy])
    end
    return P
end

function dual_points(f, irange, jrange, FT)
    points = OffsetArray(Matrix{NTuple{3, FT}}(undef, length(irange), length(jrange)), irange, jrange)
    for j in jrange, i in irange
        points[i, j] = f(i, j)
    end
    return points
end

# Fill `metric` at every index it stores: the value of `f` at the nearest index of `irange × jrange`.
function fill_metric!(f, metric, irange, jrange)
    for j in axes(metric, 2), i in axes(metric, 1)
        metric[i, j] = f(clamp(i, first(irange), last(irange)), clamp(j, first(jrange), last(jrange)))
    end
    return metric
end

function compute_spherical_metrics!(m, vertices, (Nx, Ny), (Hx, Hy), R)
    FT = eltype(m.Δxᶜᶜᵃ)
    P = corner_unit_vectors(FT, vertices, (Nx, Ny), (Hx, Hy))

    # node ranges: corners, centers, x-edge midpoints (ᶜᶠ) and y-edge midpoints (ᶠᶜ)
    If, Jf = 1-Hx:Nx+1+Hx, 1-Hy:Ny+1+Hy
    Ic, Jc = 1-Hx:Nx+Hx, 1-Hy:Ny+Hy

    C   = dual_points((i, j) -> quadrilateral_centroid(P[i, j], P[i+1, j], P[i+1, j+1], P[i, j+1]), Ic, Jc, FT)
    Mfc = dual_points((i, j) -> great_circle_midpoint(P[i, j], P[i, j+1]), If, Jc, FT)
    Mcf = dual_points((i, j) -> great_circle_midpoint(P[i, j], P[i+1, j]), Ic, Jf, FT)

    λ(v) = longitude_latitude(v)[1]
    φ(v) = longitude_latitude(v)[2]
    Ii, Ji = first(Ic)+1:last(Ic), first(Jc)+1:last(Jc) # centers with a western / southern neighbour
    Ip, Jp = first(If)+1:last(If)-1, first(Jf)+1:last(Jf)-1 # corners with both neighbours

    fill_metric!((i, j) -> λ(C[i, j]),   m.λᶜᶜᵃ, Ic, Jc)
    fill_metric!((i, j) -> λ(Mfc[i, j]), m.λᶠᶜᵃ, If, Jc)
    fill_metric!((i, j) -> λ(Mcf[i, j]), m.λᶜᶠᵃ, Ic, Jf)
    fill_metric!((i, j) -> λ(P[i, j]),   m.λᶠᶠᵃ, If, Jf)
    fill_metric!((i, j) -> φ(C[i, j]),   m.φᶜᶜᵃ, Ic, Jc)
    fill_metric!((i, j) -> φ(Mfc[i, j]), m.φᶠᶜᵃ, If, Jc)
    fill_metric!((i, j) -> φ(Mcf[i, j]), m.φᶜᶠᵃ, Ic, Jf)
    fill_metric!((i, j) -> φ(P[i, j]),   m.φᶠᶠᵃ, If, Jf)

    fill_metric!((i, j) -> R * central_angle(Mfc[i, j], Mfc[i+1, j]), m.Δxᶜᶜᵃ, Ic, Jc)
    fill_metric!((i, j) -> R * central_angle(C[i-1, j],  C[i, j]),    m.Δxᶠᶜᵃ, Ii, Jc)
    fill_metric!((i, j) -> R * central_angle(P[i, j],    P[i+1, j]),  m.Δxᶜᶠᵃ, Ic, Jf)
    fill_metric!((i, j) -> R * central_angle(Mcf[i-1, j], Mcf[i, j]), m.Δxᶠᶠᵃ, Ii, Jf)

    fill_metric!((i, j) -> R * central_angle(Mcf[i, j], Mcf[i, j+1]), m.Δyᶜᶜᵃ, Ic, Jc)
    fill_metric!((i, j) -> R * central_angle(P[i, j],   P[i, j+1]),   m.Δyᶠᶜᵃ, If, Jc)
    fill_metric!((i, j) -> R * central_angle(C[i, j-1], C[i, j]),     m.Δyᶜᶠᵃ, Ic, Ji)
    fill_metric!((i, j) -> R * central_angle(Mfc[i, j-1], Mfc[i, j]), m.Δyᶠᶠᵃ, If, Ji)

    R² = R * R
    fill_metric!((i, j) -> R² * spherical_quadrilateral_area(P[i, j], P[i+1, j], P[i+1, j+1], P[i, j+1]), m.Azᶜᶜᵃ, Ic, Jc)
    fill_metric!((i, j) -> R² * spherical_quadrilateral_area(Mcf[i-1, j], Mcf[i, j], Mcf[i, j+1], Mcf[i-1, j+1]), m.Azᶠᶜᵃ, Ii, Jc)
    fill_metric!((i, j) -> R² * spherical_quadrilateral_area(Mfc[i, j-1], Mfc[i+1, j-1], Mfc[i+1, j], Mfc[i, j]), m.Azᶜᶠᵃ, Ic, Ji)
    fill_metric!((i, j) -> R² * spherical_quadrilateral_area(C[i-1, j-1], C[i, j-1], C[i, j], C[i-1, j]), m.Azᶠᶠᵃ, Ii, Ji)

    angleᶜᶜ(i, j) = coordinate_angle(great_circle_tangent(Mfc[i, j], Mfc[i+1, j], C[i, j]),
                                     great_circle_tangent(Mcf[i, j], Mcf[i, j+1], C[i, j]), C[i, j])
    angleᶠᶜ(i, j) = coordinate_angle(great_circle_tangent(C[i-1, j], C[i, j], Mfc[i, j]),
                                     great_circle_tangent(P[i, j], P[i, j+1], Mfc[i, j]), Mfc[i, j])
    angleᶜᶠ(i, j) = coordinate_angle(great_circle_tangent(P[i, j], P[i+1, j], Mcf[i, j]),
                                     great_circle_tangent(C[i, j-1], C[i, j], Mcf[i, j]), Mcf[i, j])
    angleᶠᶠ(i, j) = coordinate_angle(great_circle_tangent(P[i-1, j], P[i+1, j], P[i, j]),
                                     great_circle_tangent(P[i, j-1], P[i, j+1], P[i, j]), P[i, j])

    fill_metric!((i, j) -> angleᶜᶜ(i, j)[1], m.cosαᶜᶜᵃ, Ic, Jc)
    fill_metric!((i, j) -> angleᶜᶜ(i, j)[2], m.sinαᶜᶜᵃ, Ic, Jc)
    fill_metric!((i, j) -> angleᶠᶜ(i, j)[1], m.cosαᶠᶜᵃ, Ii, Jc)
    fill_metric!((i, j) -> angleᶠᶜ(i, j)[2], m.sinαᶠᶜᵃ, Ii, Jc)
    fill_metric!((i, j) -> angleᶜᶠ(i, j)[1], m.cosαᶜᶠᵃ, Ic, Ji)
    fill_metric!((i, j) -> angleᶜᶠ(i, j)[2], m.sinαᶜᶠᵃ, Ic, Ji)
    fill_metric!((i, j) -> angleᶠᶠ(i, j)[1], m.cosαᶠᶠᵃ, Ip, Jp)
    fill_metric!((i, j) -> angleᶠᶠ(i, j)[2], m.sinαᶠᶠᵃ, Ip, Jp)

    return m
end

"""
    EquiangularGnomonicPanelGrid(arch = CPU(), FT = Oceananigans.defaults.FloatType;
                                 size, z,
                                 topology = (Bounded, Bounded, Bounded),
                                 halo = (3, 3, 3),
                                 radius = Oceananigans.defaults.planet_radius,
                                 panel = 1,
                                 rotation = nothing)

Return a [`SphericalGrid`](@ref) for one panel of the equiangular gnomonic cubed sphere: `Nx × Ny` cells with
uniformly spaced gnomonic angles `ξ, η ∈ [-π/4, π/4]` on the reference face centred on the North Pole, rotated by
`rotation`. The halo geometry continues the same gnomonic map beyond the panel edges (the panel's own coordinate
lines, not a neighbour panel's), which requires `Hx < Nx/2` and `Hy < Ny/2`.
"""
function EquiangularGnomonicPanelGrid(arch::AbstractArchitecture = CPU(),
                                      FT::DataType = Oceananigans.defaults.FloatType;
                                      size,
                                      z,
                                      topology = (Bounded, Bounded, Bounded),
                                      halo = (3, 3, 3),
                                      radius = Oceananigans.defaults.planet_radius,
                                      panel = 1,
                                      rotation = nothing)

    Nx, Ny, _ = size
    Hx, Hy, _ = halo
    (2Hx < Nx && 2Hy < Ny) ||
        throw(ArgumentError("the gnomonic halo continuation needs Hx < Nx/2 and Hy < Ny/2; got size $size, halo $halo"))

    Δξ = (π / 2) / Nx
    Δη = (π / 2) / Ny
    ξᶠ(i) = -π/4 + (i - 1) * Δξ
    ηᶠ(j) = -π/4 + (j - 1) * Δη

    λ = zeros(FT, Nx + 1 + 2Hx, Ny + 1 + 2Hy)
    φ = zeros(FT, Nx + 1 + 2Hx, Ny + 1 + 2Hy)
    for j in 1-Hy:Ny+1+Hy, i in 1-Hx:Nx+1+Hx
        λ[i+Hx, j+Hy], φ[i+Hx, j+Hy] = longitude_latitude(equiangular_gnomonic_map(ξᶠ(i), ηᶠ(j), rotation))
    end

    mapping = EquiangularGnomonicMapping(panel, rotation,
                                         [FT(ξᶠ(i)) for i in 1:Nx+1], [FT(ηᶠ(j)) for j in 1:Ny+1],
                                         [FT(ξᶠ(i) + Δξ / 2) for i in 1:Nx], [FT(ηᶠ(j) + Δη / 2) for j in 1:Ny])

    return SphericalGrid(arch, FT; size, z, topology, halo, radius, mapping, vertices = (λ, φ))
end

#####
##### Architecture, Adapt, show
#####

function Architectures.on_architecture(arch::AbstractSerialArchitecture, grid::SphericalGrid)
    names = (:λᶜᶜᵃ, :λᶠᶜᵃ, :λᶜᶠᵃ, :λᶠᶠᵃ, :φᶜᶜᵃ, :φᶠᶜᵃ, :φᶜᶠᵃ, :φᶠᶠᵃ, :z,
             :Δxᶜᶜᵃ, :Δxᶠᶜᵃ, :Δxᶜᶠᵃ, :Δxᶠᶠᵃ, :Δyᶜᶜᵃ, :Δyᶠᶜᵃ, :Δyᶜᶠᵃ, :Δyᶠᶠᵃ,
             :Azᶜᶜᵃ, :Azᶠᶜᵃ, :Azᶜᶠᵃ, :Azᶠᶠᵃ,
             :cosαᶜᶜᵃ, :cosαᶠᶜᵃ, :cosαᶜᶠᵃ, :cosαᶠᶠᵃ, :sinαᶜᶜᵃ, :sinαᶠᶜᵃ, :sinαᶜᶠᵃ, :sinαᶠᶠᵃ)

    arrays = Tuple(on_architecture(arch, getproperty(grid, name)) for name in names)
    TX, TY, TZ = topology(grid)

    return SphericalGrid{TX, TY, TZ}(arch, grid.Nx, grid.Ny, grid.Nz, grid.Hx, grid.Hy, grid.Hz, grid.Lz,
                                     arrays..., grid.radius, grid.shell, on_architecture(arch, grid.mapping))
end

function Adapt.adapt_structure(to, grid::SphericalGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ}
    a(x) = Adapt.adapt(to, x)
    return SphericalGrid{unwrapped_eltype(FT), TX, TY, TZ, Nothing}(nothing,
           grid.Nx, grid.Ny, grid.Nz, grid.Hx, grid.Hy, grid.Hz, a(grid.Lz),
           a(grid.λᶜᶜᵃ), a(grid.λᶠᶜᵃ), a(grid.λᶜᶠᵃ), a(grid.λᶠᶠᵃ),
           a(grid.φᶜᶜᵃ), a(grid.φᶠᶜᵃ), a(grid.φᶜᶠᵃ), a(grid.φᶠᶠᵃ), a(grid.z),
           a(grid.Δxᶜᶜᵃ), a(grid.Δxᶠᶜᵃ), a(grid.Δxᶜᶠᵃ), a(grid.Δxᶠᶠᵃ),
           a(grid.Δyᶜᶜᵃ), a(grid.Δyᶠᶜᵃ), a(grid.Δyᶜᶠᵃ), a(grid.Δyᶠᶠᵃ),
           a(grid.Azᶜᶜᵃ), a(grid.Azᶠᶜᵃ), a(grid.Azᶜᶠᵃ), a(grid.Azᶠᶠᵃ),
           a(grid.cosαᶜᶜᵃ), a(grid.cosαᶠᶜᵃ), a(grid.cosαᶜᶠᵃ), a(grid.cosαᶠᶠᵃ),
           a(grid.sinαᶜᶜᵃ), a(grid.sinαᶠᶜᵃ), a(grid.sinαᶜᶠᵃ), a(grid.sinαᶠᶠᵃ),
           a(grid.radius), grid.shell, a(grid.mapping))
end

shell_summary(::ThinShell) = "thin shell"
shell_summary(::DeepShell) = "deep shell"
mapping_summary(::Nothing) = "no mapping"
mapping_summary(mapping) = summary(mapping)

function Base.summary(grid::SphericalGrid)
    FT = eltype(grid)
    nTX, nTY, nTZ = map(prettysummary, topology(grid))
    return string(size_summary(grid), " SphericalGrid{$FT, $nTX, $nTY, $nTZ} on ", summary(architecture(grid)),
                  " with ", size_summary(halo_size(grid)), " halo (", shell_summary(grid.shell), ", ",
                  mapping_summary(grid.mapping), ")")
end

function Base.show(io::IO, grid::SphericalGrid, withsummary=true)
    withsummary && print(io, summary(grid), "\n")
    Nx, Ny, Nz = size(grid)
    TZ = topology(grid, 3)
    Ωz = domain(TZ(), Nz, grid.z.cᵃᵃᶠ)
    interior_cosα = @allowscalar view(grid.cosαᶜᶜᵃ, 1:Nx, 1:Ny)
    α_range = @allowscalar extrema(acosd.(clamp.(interior_cosα, -1, 1)))
    return print(io, "├── ", center_line_summary(grid), "\n",
                     "├── coordinate-line angle α at centers ∈ [", prettysummary(α_range[1]), ", ",
                     prettysummary(α_range[2]), "] degrees\n",
                     "└── z: ", dimension_summary(TZ(), "z", Ωz, grid.z, 0))
end
