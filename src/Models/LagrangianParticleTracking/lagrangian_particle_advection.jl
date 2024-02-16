using Oceananigans.Utils: instantiate 
using Oceananigans.Models: total_velocities

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

const f = Face()
const c = Center()

"""
    bounce_immersed_particle((x, y, z), grid, restitution, previous_particle_indices)

Return a new particle position if the position `(x, y, z)` lies in an immersed cell by
bouncing the particle off the immersed boundary with a coefficient or `restitution`.
"""
@inline function bounce_immersed_particle((x, y, z), ibg, restitution, previous_particle_indices)
    X = flattened_node((x, y, z), ibg)

    # Determine current particle cell
    i, j, k = fractional_indices(X, ibg.underlying_grid, (c, c, c))
    i = Base.unsafe_trunc(Int, i)
    j = Base.unsafe_trunc(Int, j)
    k = Base.unsafe_trunc(Int, k)
   
    if immersed_cell(i, j, k, ibg)
        # Determine whether particle was _previously_ in a non-immersed cell
        i⁻, j⁻, k⁻ = previous_particle_indices
       
        if !immersed_cell(i⁻, j⁻, k⁻, ibg)
            # Left-right bounds of the previous, non-immersed cell
            xᴿ, yᴿ, zᴿ = node(i⁻+1, j⁻+1, k⁻+1, ibg, f, f, f)
            xᴸ, yᴸ, zᴸ = node(i⁻,   j⁻,   k⁻,   ibg, f, f, f)

            Cʳ = restitution
            x⁺ = enforce_boundary_conditions(Bounded(), x, xᴸ, xᴿ, Cʳ)    
            y⁺ = enforce_boundary_conditions(Bounded(), y, yᴸ, yᴿ, Cʳ)    
            z⁺ = enforce_boundary_conditions(Bounded(), z, zᴸ, zᴿ, Cʳ)

        end
    end

    return x⁺, y⁺, z⁺
end

"""
    advect_particle((x, y, z), p, restitution, grid, Δt, velocities)

Return new position `(x⁺, y⁺, z⁺)` for a particle at current position (x, y, z),
given `velocities`, time-step `Δt, and coefficient of `restitution`.
"""
@inline function advect_particle((x, y, z), p, restitution, grid, Δt, velocities)
    X = flattened_node((x, y, z), grid)

    # Obtain current particle indices
    i, j, k = fractional_indices(X, grid, c, c, c)
    i = Base.unsafe_trunc(Int, i)
    j = Base.unsafe_trunc(Int, j)
    k = Base.unsafe_trunc(Int, k)

    current_particle_indices = (i, j, k)

    # Interpolate velocity to particle position
    u = interpolate(X, velocities.u, (f, c, c), grid)
    v = interpolate(X, velocities.v, (c, f, c), grid)
    w = interpolate(X, velocities.w, (c, c, f), grid)

    # Advect particles, calculating the advection metric for a curvilinear grid.
    # Note that all supported grids use length coordinates in the vertical, so we do not
    # transform the vertical velocity nor invoke the k-index.
    ξ = x_metric(i, j, grid) 
    η = y_metric(i, j, grid)

    x⁺ = x + ξ * u * Δt
    y⁺ = y + η * v * Δt
    z⁺ = z + w * Δt

    # Satisfy boundary conditions for particles: bounce off walls, travel over periodic boundaries.
    tx, ty, tz = map(instantiate, topology(grid))
    Nx, Ny, Nz = size(grid)

    # Find index of the "rightmost" cell interface
    iᴿ = length(f, tx, Nx)
    jᴿ = length(f, ty, Ny)
    kᴿ = length(f, tz, Nz)

    xᴸ = xnode(1, j, k, grid, f, f, f)
    yᴸ = ynode(i, 1, k, grid, f, f, f)
    zᴸ = znode(i, j, 1, grid, f, f, f)

    xᴿ = xnode(iᴿ, j, k, grid, f, f, f)
    yᴿ = ynode(i, jᴿ, k, grid, f, f, f)
    zᴿ = znode(i, j, kᴿ, grid, f, f, f)

    # Enforce boundary conditions for particles.
    Cʳ = restitution
    x⁺ = enforce_boundary_conditions(tx, x⁺, xᴸ, xᴿ, Cʳ)
    y⁺ = enforce_boundary_conditions(ty, y⁺, yᴸ, yᴿ, Cʳ)
    z⁺ = enforce_boundary_conditions(tz, z⁺, zᴸ, zᴿ, Cʳ)
    if grid isa ImmersedBoundaryGrid
        previous_particle_indices = current_particle_indices # particle has been advected
        x⁺, y⁺, z⁺ = bounce_immersed_particle((x⁺, y⁺, z⁺), grid, Cʳ, previous_particle_indices)
    end

    return (x⁺, y⁺, z⁺)
end

# Calculate the metric for particle advection according to the coordinate system of the `grid`:
#     * Unity metric for `RectilinearGrid` / Cartesian coordinates
#     * Sphere metric for `LatitudeLongitudeGrid` and geographic coordinates
@inline x_metric(i, j, grid::RectilinearGrid) = 1
@inline x_metric(i, j, grid::LatitudeLongitudeGrid{FT}) where FT = @inbounds 1 / (grid.radius * hack_cosd(grid.φᶜᶜᵃ[j])) * FT(360 / 2π)

@inline y_metric(i, j, grid::RectilinearGrid) = 1
@inline y_metric(i, j, grid::LatitudeLongitudeGrid{FT}) where FT = 1 / grid.radius * FT(360 / 2π)

@kernel function _advect_particles!(particles, restitution, grid::AbstractUnderlyingGrid, Δt, velocities) 
    p = @index(Global)

    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]
    end

    x⁺, y⁺, z⁺ = advect_particle((x, y, z), p, restitution, grid, Δt, velocities) 

    @inbounds begin
        particles.x[p] = x⁺ 
        particles.y[p] = y⁺ 
        particles.z[p] = z⁺ 
    end
end

function advect_lagrangian_particles!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    workgroup = min(length(particles), 256)
    worksize = length(particles)

    advect_particles_kernel! = _advect_particles!(device(arch), workgroup, worksize)
    advect_particles_kernel!(particles.properties, particles.restitution, model.grid, Δt, total_velocities(model))

    return nothing
end

