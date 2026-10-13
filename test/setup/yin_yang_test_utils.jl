using Oceananigans: instantiated_location
using Oceananigans.Grids: λnode, φnode, halo_size
using Oceananigans.MultiRegion: yin_yang_panel_extents, yin_yang_transform, location_index_offset
using Oceananigans.OrthogonalSphericalShellGrids: rotated_basis_angle
using Oceananigans.Utils: getregion

const yin_yang_locations = ((Center(), Center()), (Face(), Center()), (Center(), Face()), (Face(), Face()))

location_field(ℓx, ℓy, grid) = Field{typeof(ℓx), typeof(ℓy), Center}(grid)

wrap_longitude(λ) = mod(λ + 180, 360) - 180

node_coordinates(i, j, grid, ℓx, ℓy) = (λnode(i, j, 1, grid, ℓx, ℓy, Center()),
                                        φnode(i, j, 1, grid, ℓx, ℓy, Center()))

north_pole(panel) = panel.conformal_mapping.north_pole

is_halo(i, j, grid) = !(1 ≤ i ≤ size(grid, 1) && 1 ≤ j ≤ size(grid, 2))

function halo_nodes(grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    return [(i, j) for j in 1-Hy:Ny+Hy for i in 1-Hx:Nx+Hx if is_halo(i, j, grid)]
end

convergence_rates(errors) = log2.(errors[1:end-1] ./ errors[2:end])

# The fractional indices at (ℓx, ℓy) on panel `r` of the geographic point (λ, φ)
function geographic_fractional_indices(λ, φ, grid, r, ℓx, ℓy)
    connectivity = grid.connectivity
    Δ = 90 / connectivity.core_size[2]
    λᵉ, φᵉ = yin_yang_panel_extents(connectivity.core_size, connectivity.overlap)
    λ′, φ′ = r == 1 ? (wrap_longitude(λ), φ) : yin_yang_transform(λ, φ)
    i★ = (λ′ + λᵉ) / Δ + 1 - location_index_offset(ℓx)
    j★ = (φ′ + φᵉ) / Δ + 1 - location_index_offset(ℓy)
    return i★, j★
end

regional_parents(field) = Tuple(Array(parent(getregion(field, r))) for r in 1:2)

# A smooth function of the geographic coordinates (degrees) with Y₃²- and Y₄¹-like structure
spherical_harmonics(λ, φ) = cosd(φ)^2 * sind(φ) * cosd(2λ) + cosd(φ) * (7sind(φ)^3 - 3sind(φ)) * sind(λ - 40)

# Geographic (east, north) velocity of solid-body rotation about the axis tilted by α from the pole.
# At the poles, these are the limits along the meridian λ.
function solid_body_velocity(λ, φ; α = π/4)
    u = cosd(φ) * cos(α) + sind(φ) * cosd(λ) * sin(α)
    v = -sind(λ) * sin(α)
    return u, v
end

# The components of the geographic `velocity` in the basis of the panel with `north_pole`
function panel_components(velocity, λ, φ, north_pole)
    uₑ, vₑ = velocity(λ, φ)
    θ = rotated_basis_angle(λ, φ, north_pole...)
    return uₑ * cos(θ) - vₑ * sin(θ), uₑ * sin(θ) + vₑ * cos(θ)
end

function set_panel_components!(u, v, velocity)
    for r in 1:2
        pole = north_pole(u.grid[r])
        set!(getregion(u, r), (λ, φ, z) -> panel_components(velocity, λ, φ, pole)[1])
        set!(getregion(v, r), (λ, φ, z) -> panel_components(velocity, λ, φ, pole)[2])
    end
    return nothing
end

# The largest difference between `field` and `exact(λ, φ, r)` over the `nodes(grid)` of both panels `r`
function panel_error(field, exact; nodes = halo_nodes)
    grid = field.grid
    ℓx, ℓy, _ = instantiated_location(field)
    error = 0.0
    for r in 1:2
        panel = on_architecture(CPU(), grid[r])
        data = on_architecture(CPU(), getregion(field, r))
        for (i, j) in nodes(grid)
            λ, φ = node_coordinates(i, j, panel, ℓx, ℓy)
            error = max(error, maximum(abs, data[i, j, k] - exact(λ, φ, r) for k in 1:size(field, 3)))
        end
    end
    return error
end

panel_u(velocity, grid) = (λ, φ, r) -> panel_components(velocity, λ, φ, north_pole(grid[r]))[1]
panel_v(velocity, grid) = (λ, φ, r) -> panel_components(velocity, λ, φ, north_pole(grid[r]))[2]
