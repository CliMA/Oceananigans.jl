using Oceananigans.Models.NonhydrostaticModels: NonhydrostaticModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

#####
##### Boundary conditions for Lagrangian particles
#####

@inline  bounce_left(x, xᴿ, Cʳ) = xᴿ - Cʳ * (x - xᴿ)
@inline bounce_right(x, xᴸ, Cʳ) = xᴸ + Cʳ * (xᴸ - x)

"""
    enforce_boundary_conditions(x, xᴸ, xᴿ, ::Bounded)

If a particle with position `x` and domain `xᴸ < x < xᴿ` goes through the edge of the domain
along a `Bounded` dimension, put them back at the wall.
"""
@inline enforce_boundary_conditions(::Bounded, x, xᴸ, xᴿ, Cʳ) = ifelse(x > xᴿ, bounce_left(x, xᴿ, Cʳ),
                                                                ifelse(x < xᴸ, bounce_right(x, xᴸ, Cʳ), x))

"""
    enforce_boundary_conditions(x, xᴸ, xᴿ, ::Periodic)

If a particle with position `x` and domain `xᴸ < x < xᴿ` goes through the edge of the domain
along a `Periodic` dimension, put them on the other side.
"""
@inline enforce_boundary_conditions(::Periodic, x, xᴸ, xᴿ, Cʳ) = ifelse(x > xᴿ, xᴸ + (x - xᴿ),
                                                                 ifelse(x < xᴸ, xᴿ - (xᴸ - x), x))
                                                                          

const f = Face()
const c = Center()

"""
    bounce_immersed_particle((x, y, z), grid, restitution, previous_particle_indices)

If a particle with position `x, y, z` is inside and immersed boundary, correct the 
position based on the previous position (we bounce back a certain restitution from the old cell)
"""
@inline function bounce_immersed_particle((x, y, z), grid, restitution, previous_particle_indices)
    # Determine current particle cell
    i, j, k = fractional_indices(x, y, z, (c, c, c), grid.underlying_grid)
    i = Base.unsafe_trunc(Int, i)
    j = Base.unsafe_trunc(Int, j)
    k = Base.unsafe_trunc(Int, k)
   
    if immersed_cell(i, j, k, grid)
        # Determine previous particle indices
        i⁻, j⁻, k⁻ = previous_particle_indices
        i⁻ = Base.unsafe_trunc(Int, i⁻)
        j⁻ = Base.unsafe_trunc(Int, j⁻)
        k⁻ = Base.unsafe_trunc(Int, k⁻)
       
        if !immersed_cell(i⁻, j⁻, k⁻, grid)
            xᴿ, yᴿ, zᴿ = node(i+1, j+1, k+1, grid, f, f, f)
            xᴸ, yᴸ, zᴸ = node(i, j, k, grid, f, f, f)

            # What if we bounce too far? Hmm.
            Cʳ = restitution
            x⁺ = enforce_boundary_conditions(Bounded(), x, xᴸ, xᴿ, Cʳ)    
            y⁺ = enforce_boundary_conditions(Bounded(), y, yᴸ, yᴿ, Cʳ)    
            z⁺ = enforce_boundary_conditions(Bounded(), z, zᴸ, zᴿ, Cʳ)    
        end
    end

    return x⁺, y⁺, z⁺
end

"""
    advect_particle((x, y, z), p, grid, restitution, velocities, Δt)

Return new position `(x⁺, y⁺, z⁺)` for a particle at current position (x, y, z),
given `velocities`, time-step `Δt, and coefficient of `restitution`.
"""
@inline function advect_particle((x, y, z), grid, restitution, velocities, Δt)
    current_particle_indices = fractional_indices(x, y, z, (c, c, c), grid)

    # Interpolate velocity to particle position
    u = interpolate(velocities.u, f, c, c, grid, x, y, z)
    v = interpolate(velocities.v, c, f, c, grid, x, y, z)
    w = interpolate(velocities.w, c, c, f, grid, x, y, z)

    # We need the j-index of the particle in order to compute
    # metrics for advection on curvilinear grids.
    # Note: to support curvilinear grids other than LatitudeLongitudeGrid,
    # we will also need the i-index.
    i, j, k = fractional_indices(x, y, z, (c, c, c), grid)
    i = Base.unsafe_trunc(Int, i)
    j = Base.unsafe_trunc(Int, j)
    k = Base.unsafe_trunc(Int, k)

    # Transform Cartesian velocities into grid-dependent particle coordinate system.
    # Note that all supported grids use length coordinates in the vertical, so we do not
    # transform the vertical velocity nor invoke the k-index.
    ξ = x_metric(i, j, grid) 
    η = y_metric(i, j, grid) 
        
    x⁺ = x + ξ * uₚ * Δt
    y⁺ = y + η * uₚ * Δt
    z⁺ = z + wₚ * Δt

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
@inline x_metric(i, j, grid::RectilinearGrid, u) = 1
@inline x_metric(i, j, grid::LatitudeLongitudeGrid{FT}, u) where FT = 1 / (grid.radius * hack_cosd(grid.φᵃᶜᵃ[j])) * FT(360 / 2π)

@inline y_metric(i, j, grid::RectilinearGrid) = v
@inline y_metric(i, j, grid::LatitudeLongitudeGrid{FT}) where FT = v / grid.radius * FT(360 / 2π)

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

total_velocities(model::NonhydrostaticModel) = (u = SumOfArrays{2}(model.velocities.u, model.background_fields.velocities.u),
                                                v = SumOfArrays{2}(model.velocities.v, model.background_fields.velocities.v),
                                                w = SumOfArrays{2}(model.velocities.w, model.background_fields.velocities.w))

total_velocities(model::HydrostaticFreeSurfaceModel) = model.velocities

function advect_lagrangian_particles!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    workgroup = min(length(particles), 256)
    worksize = length(particles)

    advect_particles_kernel! = _advect_particles!(device(arch), workgroup, worksize)

    event = advect_particles_kernel!(particles.properties,
                                     particles.restitution,
                                     grid,
                                     Δt,
                                     total_velocities(model),
                                     dependencies = device_event(arch))

    wait(device(arch), event)

    return nothing
end

