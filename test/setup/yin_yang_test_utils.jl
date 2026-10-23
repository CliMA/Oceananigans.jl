using Oceananigans: instantiated_location, TracerTendencyCallsite
using Oceananigans.Grids: λnode, φnode, halo_size, immersed_inactive_node
using Oceananigans.MultiRegion: yin_yang_panel_face_coordinates, yin_yang_transform, panel_frame, panel_fractional_indices,
                                active_reference_volume
using Oceananigans.Operators: Azᶜᶜᶜ, Vᶜᶜᶜ
using Oceananigans.OrthogonalSphericalShellGrids: rotated_basis_angle
using Oceananigans.Utils: getregion

const yin_yang_locations = ((Center(), Center()), (Face(), Center()), (Center(), Face()), (Face(), Face()))

location_field(ℓx, ℓy, grid) = Field{typeof(ℓx), typeof(ℓy), Center}(grid)
wrap_longitude(λ) = mod(λ + 180, 360) - 180
node_coordinates(i, j, grid, ℓx, ℓy) = (λnode(i, j, 1, grid, ℓx, ℓy, Center()),
                                        φnode(i, j, 1, grid, ℓx, ℓy, Center()))
north_pole(panel) = panel.conformal_mapping.north_pole
convergence_rates(errors) = log2.(errors[1:end-1] ./ errors[2:end])
regional_parents(field) = Tuple(Array(parent(getregion(field, r))) for r in 1:2)
interior_nodes(grid) = [(i, j) for j in 1:size(grid, 2) for i in 1:size(grid, 1)]

halo_nodes(grid) = Oceananigans.MultiRegion.halo_nodes(size(grid)[1:2], halo_size(grid)[1:2])

# The fractional indices at (ℓx, ℓy) on panel `r` of the geographic point (λ, φ)
function geographic_fractional_indices(λ, φ, grid, r, ℓx, ℓy)
    connectivity = grid.connectivity
    frame = panel_frame(yin_yang_panel_face_coordinates(connectivity.core_size, connectivity.overlap)...)
    λ′, φ′ = r == 1 ? (wrap_longitude(λ), φ) : yin_yang_transform(λ, φ)
    return panel_fractional_indices(λ′, φ′, frame, (ℓx, ℓy))
end

# A smooth function of the geographic coordinates (degrees) with Y₃²- and Y₄¹-like structure
spherical_harmonics(λ, φ) = cosd(φ)^2 * sind(φ) * cosd(2λ) + cosd(φ) * (7sind(φ)^3 - 3sind(φ)) * sind(λ - 40)

# Geographic (east, north) velocity of solid-body rotation about the axis tilted by α from the pole
solid_body_velocity(λ, φ; α = π/4) = (cosd(φ) * cos(α) + sind(φ) * cosd(λ) * sin(α), -sind(λ) * sin(α))

# The components of the geographic `velocity` in the basis of the panel with `north_pole`
function panel_components(velocity, λ, φ, north_pole)
    uₑ, vₑ = velocity(λ, φ)
    θ = rotated_basis_angle(λ, φ, north_pole...)
    return uₑ * cos(θ) - vₑ * sin(θ), uₑ * sin(θ) + vₑ * cos(θ)
end

panel_component(n, velocity, grid) = (λ, φ, r) -> panel_components(velocity, λ, φ, north_pole(grid[r]))[n]
panel_u(velocity, grid) = panel_component(1, velocity, grid)
panel_v(velocity, grid) = panel_component(2, velocity, grid)

function set_panel_components!(u, v, velocity)
    for (n, field) in enumerate((u, v)), r in 1:2
        component = panel_component(n, velocity, field.grid)
        set!(getregion(field, r), (λ, φ, z) -> component(λ, φ, r))
    end
    return nothing
end

# The largest difference between `field` and `exact(λ, φ, r)` over the `nodes(grid)` of both panels `r`
function panel_error(field, exact; nodes = halo_nodes)
    ℓx, ℓy, _ = instantiated_location(field)
    error = 0.0
    for r in 1:2
        panel = on_architecture(CPU(), field.grid[r])
        data = on_architecture(CPU(), getregion(field, r))
        for (i, j) in nodes(field.grid)
            λ, φ = node_coordinates(i, j, panel, ℓx, ℓy)
            error = max(error, maximum(abs, data[i, j, k] - exact(λ, φ, r) for k in 1:size(field, 3)))
        end
    end
    return error
end

# Active contents and volumes, summed so that every point of a Yin-Yang grid counts once
@inline active(i, j, k, grid) = !immersed_inactive_node(i, j, k, grid, Center(), Center(), Center())
@inline active_content(i, j, k, grid, c) = ifelse(active(i, j, k, grid), Vᶜᶜᶜ(i, j, k, grid) * c[i, j, k], zero(grid))
@inline top_area(i, j, k, grid) = ifelse((k == size(grid, 3)) & active(i, j, k, grid), Azᶜᶜᶜ(i, j, k, grid), zero(grid))
@inline top_volume(i, j, k, grid, η) = @inbounds top_area(i, j, k, grid) * η[i, j, size(grid, 3) + 1]

owned_sum(f, grid, args...) = sum(KernelFunctionOperation{Center, Center, Center}(f, grid, args...))
active_content(c) = owned_sum(active_content, c.grid, c)

# The reference volume plus the volume of the free surface, for static and moving vertical coordinates
active_volume(model) = owned_sum(active_reference_volume, model.grid) +
                       owned_sum(top_volume, model.grid, model.free_surface.displacement.data)

# The callbacks with which `time_step!(model, Δt; callbacks)` remove the content that the panels add
conservation_fixer_callbacks(model) =
    (Callback(YinYangConservationFixer(model), callsite = TracerTendencyCallsite()),)
