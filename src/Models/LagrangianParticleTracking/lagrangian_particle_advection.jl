using Oceananigans.Utils: instantiate, KernelParameters
using Oceananigans.Models: total_velocities
using Oceananigans.Fields: interpolator, FractionalIndices

#####
##### Boundary conditions for Lagrangian particles
#####

# Functions for bouncing particles off walls to the right and left
@inline  bounce_left(x, xᴿ, Cʳ) = xᴿ - Cʳ * (x - xᴿ)
@inline bounce_right(x, xᴸ, Cʳ) = xᴸ + Cʳ * (xᴸ - x)

"""
    enforce_boundary_conditions(::Bounded, x, xᴸ, xᴿ, Cʳ)

Return a new particle position if the particle position `x`
is outside the Bounded interval `(xᴸ, xᴿ)` by bouncing the particle off
the interval edge with coefficient of restitution `Cʳ).
"""
@inline enforce_boundary_conditions(::Bounded, x, xᴸ, xᴿ, Cʳ) = ifelse(x > xᴿ, bounce_left(x, xᴿ, Cʳ),
                                                                ifelse(x < xᴸ, bounce_right(x, xᴸ, Cʳ), x))

"""
    enforce_boundary_conditions(::Periodic, x, xᴸ, xᴿ, Cʳ)

Return a new particle position if the particle position `x`
is outside the Periodic interval `(xᴸ, xᴿ)`.
"""
@inline enforce_boundary_conditions(::Periodic, x, xᴸ, xᴿ, Cʳ) = ifelse(x > xᴿ, xᴸ + (x - xᴿ),
                                                                 ifelse(x < xᴸ, xᴿ - (xᴸ - x), x))

"""
    enforce_boundary_conditions(::Flat, x, xᴸ, xᴿ, Cʳ)

Do nothing on Flat dimensions.
"""
@inline enforce_boundary_conditions(::Flat, x, xᴸ, xᴿ, Cʳ) = x

const f = Face()
const c = Center()

"""
    immersed_boundary_topology(grid_topology)

Unless `Flat`, immersed boundaries are treated as `Bounded` regardless of underlying grid topology.
"""
immersed_boundary_topology(grid_topology) = ifelse(grid_topology == Flat, Flat(), Bounded())

"""
    bounce_immersed_particle((x, y, z), grid, restitution, previous_particle_indices)

Return a new particle position if the position `(x, y, z)` lies in an immersed cell by
bouncing the particle off the immersed boundary with a coefficient or `restitution`.
"""
@inline function bounce_immersed_particle((x, y, z), ibg, restitution, previous_particle_indices)
    X = flattened_node((x, y, z), ibg)

    # Determine current particle cell from the interfaces
    fi = FractionalIndices(X, ibg.underlying_grid, f, f, f)

    i, i⁺, _ = interpolator(fi.i)
    j, j⁺, _ = interpolator(fi.j)
    k, k⁺, _ = interpolator(fi.k)

    # Determine whether particle was _previously_ in a non-immersed cell
    i⁻, j⁻, k⁻ = previous_particle_indices

    tx, ty, tz = map(immersed_boundary_topology, topology(ibg))

    # Right bounds of the previous cell
    xᴿ = ξnode(i⁺, j,  k, ibg, f, f, f)
    yᴿ = ηnode(i,  j⁺, k, ibg, f, f, f)
    zᴿ = rnode(i,  j,  k⁺, ibg, f, f, f)

    # Left bounds of the previous cell
    xᴸ = ξnode(i⁻, j⁻, k⁻, ibg, f, f, f)
    yᴸ = ηnode(i⁻, j⁻, k⁻, ibg, f, f, f)
    zᴸ = rnode(i⁻, j⁻, k⁻, ibg, f, f, f)

    Cʳ = restitution

    xb⁺ = enforce_boundary_conditions(tx, x, xᴸ, xᴿ, Cʳ)
    yb⁺ = enforce_boundary_conditions(ty, y, yᴸ, yᴿ, Cʳ)
    zb⁺ = enforce_boundary_conditions(tz, z, zᴸ, zᴿ, Cʳ)

    immersed = immersed_cell(i⁺, j⁺, k⁺, ibg)
    x⁺ = ifelse(immersed, xb⁺, x)
    y⁺ = ifelse(immersed, yb⁺, y)
    z⁺ = ifelse(immersed, zb⁺, z)

    return (x⁺, y⁺, z⁺)
end

"""
    rightmost_interface_index(topology, N)

Return the index of the rightmost cell interface for a grid with `topology` and `N` cells.
"""
rightmost_interface_index(::Bounded, N)  = N + 1
rightmost_interface_index(::Periodic, N) = N + 1
rightmost_interface_index(::Flat, N) = N

"""
    advect_particle((x, y, z), p, restitution, grid, Δt, velocities)

Return new position `(x⁺, y⁺, z⁺)` for a particle at current position (x, y, z),
given `velocities`, time-step `Δt, and coefficient of `restitution`.
"""
@inline function advect_particle((x, y, z), particles, p, restitution, grid, Δt, velocities)
    X = flattened_node((x, y, z), grid)

    # Obtain current particle indices, looking at the interfaces
    fi = FractionalIndices(X, grid, f, f, f)

    i, i⁺, _ = interpolator(fi.i)
    j, j⁺, _ = interpolator(fi.j)
    k, k⁺, _ = interpolator(fi.k)

    current_particle_indices = (i, j, k)

    uf = interpolate(X, velocities.u, (f, c, c), grid)
    vf = interpolate(X, velocities.v, (c, f, c), grid)
    wf = interpolate(X, velocities.w, (c, c, f), grid)

    # Interpolate velocity to particle position
    up = particle_u_velocity(particles, p, uf)
    vp = particle_v_velocity(particles, p, vf)
    wp = particle_w_velocity(particles, p, wf)

    # Advect particles, calculating the advection metric for a curvilinear grid.
    # Note that all supported grids use length coordinates in the vertical, so we do not
    # transform the vertical velocity nor invoke the k-index.
    ξ = x_metric(i, j, grid)
    η = y_metric(i, j, grid)

    x⁺ = x + ξ * up * Δt
    y⁺ = y + η * vp * Δt
    z⁺ = z +     wp * Δt

    # Satisfy boundary conditions for particles: bounce off walls, travel over periodic boundaries.
    tx, ty, tz = map(instantiate, topology(grid))
    Nx, Ny, Nz = size(grid)

    # Find index of the "rightmost" cell interface
    iᴿ = rightmost_interface_index(tx, Nx)
    jᴿ = rightmost_interface_index(ty, Ny)
    kᴿ = rightmost_interface_index(tz, Nz)

    xᴸ = ξnode(1, j, k, grid, f, f, f)
    yᴸ = ηnode(i, 1, k, grid, f, f, f)
    zᴸ = rnode(i, j, 1, grid, f, f, f)

    xᴿ = ξnode(iᴿ, j,  k,  grid, f, f, f)
    yᴿ = ηnode(i,  jᴿ, k,  grid, f, f, f)
    zᴿ = rnode(i,  j,  kᴿ, grid, f, f, f)

    # Enforce boundary conditions for particles.
    Cʳ = restitution
    x⁺ = enforce_boundary_conditions(tx, x⁺, xᴸ, xᴿ, Cʳ)
    y⁺ = enforce_boundary_conditions(ty, y⁺, yᴸ, yᴿ, Cʳ)
    z⁺ = enforce_boundary_conditions(tz, z⁺, zᴸ, zᴿ, Cʳ)

    if grid isa ImmersedBoundaryGrid
        previous_particle_indices = current_particle_indices # particle has been advected
        (x⁺, y⁺, z⁺) = bounce_immersed_particle((x⁺, y⁺, z⁺), grid, Cʳ, previous_particle_indices)
    end

    return (x⁺, y⁺, z⁺)
end

@inline particle_u_velocity(particles, p, uf) = uf
@inline particle_v_velocity(particles, p, vf) = vf
@inline particle_w_velocity(particles, p, wf) = wf

# Calculate the metric for particle advection according to the coordinate system of the `grid`:
#     * Unity metric for `RectilinearGrid` / Cartesian coordinates
#     * Sphere metric for `LatitudeLongitudeGrid` and geographic coordinates
@inline x_metric(i, j, grid::RectilinearGrid) = 1
@inline x_metric(i, j, grid::LatitudeLongitudeGrid{FT}) where FT = @inbounds 1 / (grid.radius * hack_cosd(grid.φᵃᶜᵃ[j])) * FT(360 / 2π)
@inline x_metric(i, j, grid::ImmersedBoundaryGrid) = x_metric(i, j, grid.underlying_grid)

@inline y_metric(i, j, grid::RectilinearGrid) = 1
@inline y_metric(i, j, grid::LatitudeLongitudeGrid{FT}) where FT = 1 / grid.radius * FT(360 / 2π)
@inline y_metric(i, j, grid::ImmersedBoundaryGrid) = y_metric(i, j, grid.underlying_grid)

@kernel function _advect_particles!(particles, restitution, grid::AbstractGrid, Δt, velocities)
    p = @index(Global)

    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]
    end

    x⁺, y⁺, z⁺ = advect_particle((x, y, z), particles, p, restitution, grid, Δt, velocities)

    @inbounds begin
        particles.x[p] = x⁺
        particles.y[p] = y⁺
        particles.z[p] = z⁺
    end
end

function advect_lagrangian_particles!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    parameters = KernelParameters(1:length(particles))

    launch!(arch, grid, parameters,
            _advect_particles!,
            particles.properties, particles.restitution, model.grid, Δt, total_velocities(model))

    return nothing
end
