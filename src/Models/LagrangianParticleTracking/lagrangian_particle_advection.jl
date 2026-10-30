using Oceananigans.Utils: instantiate, KernelParameters
using Oceananigans.Models: total_velocities
using Oceananigans.Fields: interpolator, FractionalIndices

const f = Face()
const c = Center()

"""
$(TYPEDSIGNATURES)

Unless `Flat`, immersed boundaries are treated as `Bounded` regardless of underlying grid topology.
"""
immersed_boundary_topology(grid_topology) = ifelse(grid_topology == Flat, Flat(), Bounded())

"""
$(TYPEDSIGNATURES)

Return a new particle position if the position `(x, y, z)` lies in an immersed cell, and whether
the particle was absorbed, according to the immersed boundary condition `bc`.

A [`ReflectingBoundary`](@ref) reflects the particle back into the cell with indices
`previous_particle_indices` that it occupied before it was advected, off the faces of that cell.
The reflection is computed from the unwrapped position `(x̃, ỹ, z̃)` of the particle, i.e. its
advected position before domain boundary conditions were applied. An [`AbsorbingBoundary`](@ref)
places the particle on the faces of its previous cell and flags it as absorbed.
"""
@inline function bounce_immersed_particle((x, y, z), (x̃, ỹ, z̃), ibg, bc, previous_particle_indices)
    X = flattened_node((x, y, z), ibg)

    # Determine the cell the particle is in now from the interfaces
    fi = FractionalIndices(X, ibg.underlying_grid, f, f, f)

    i, _, _ = interpolator(fi.i)
    j, _, _ = interpolator(fi.j)
    k, _, _ = interpolator(fi.k)

    immersed = immersed_cell(i, j, k, ibg)

    # Indices of the cell the particle occupied _before_ it was advected
    i⁻, j⁻, k⁻ = previous_particle_indices

    # Left bounds of the previous cell
    xᴸ = ξnode(i⁻, j⁻, k⁻, ibg, f, f, f)
    yᴸ = ηnode(i⁻, j⁻, k⁻, ibg, f, f, f)
    zᴸ = rnode(i⁻, j⁻, k⁻, ibg, f, f, f)

    # Right bounds of the previous cell
    xᴿ = ξnode(i⁻ + 1, j⁻,     k⁻,     ibg, f, f, f)
    yᴿ = ηnode(i⁻,     j⁻ + 1, k⁻,     ibg, f, f, f)
    zᴿ = rnode(i⁻,     j⁻,     k⁻ + 1, ibg, f, f, f)

    # Return the particle to the previous cell
    tx, ty, tz = map(immersed_boundary_topology, topology(ibg))

    xb⁺, _ = enforce_boundary_conditions(tx, x̃, xᴸ, xᴿ, bc, bc)
    yb⁺, _ = enforce_boundary_conditions(ty, ỹ, yᴸ, yᴿ, bc, bc)
    zb⁺, _ = enforce_boundary_conditions(tz, z̃, zᴸ, zᴿ, bc, bc)

    x⁺ = ifelse(immersed, xb⁺, x)
    y⁺ = ifelse(immersed, yb⁺, y)
    z⁺ = ifelse(immersed, zb⁺, z)
    absorbed = immersed & absorbs(bc)

    return (x⁺, y⁺, z⁺), absorbed
end

"""
$(TYPEDSIGNATURES)

Return the index of the rightmost cell interface for a grid with `topology` and `N` cells.
"""
rightmost_interface_index(::Bounded, N)  = N + 1
rightmost_interface_index(::Periodic, N) = N + 1
rightmost_interface_index(::Flat, N) = N


"""
$(TYPEDSIGNATURES)

Return the indices `(i, j, k)` of the cell that contains a particle at position `(x, y, z)`,
and the velocity `(ẋ, ẏ, ż)` of the particle in the coordinates of `grid`; for example,
in degrees per second for the horizontal coordinates of a `LatitudeLongitudeGrid`.

The particle velocity is the fluid `velocities` interpolated to the particle position and
then modified by `particle_u_velocity`, `particle_v_velocity`, and `particle_w_velocity`.
"""
@inline function particle_coordinate_velocity((x, y, z), particles, p, grid, velocities)
    X = flattened_node((x, y, z), grid)

    # Obtain current particle indices, looking at the interfaces
    fi = FractionalIndices(X, grid, f, f, f)

    i, _, _ = interpolator(fi.i)
    j, _, _ = interpolator(fi.j)
    k, _, _ = interpolator(fi.k)

    uf = interpolate(X, velocities.u, (f, c, c), grid)
    vf = interpolate(X, velocities.v, (c, f, c), grid)
    wf = interpolate(X, velocities.w, (c, c, f), grid)

    # Interpolate velocity to particle position
    up = particle_u_velocity(particles, p, uf)
    vp = particle_v_velocity(particles, p, vf)
    wp = particle_w_velocity(particles, p, wf)

    # Transform to coordinate velocities with the metric of a curvilinear grid.
    # Note that all supported grids use length coordinates in the vertical, so we do not
    # transform the vertical velocity nor invoke the k-index.
    ξ = x_metric(i, j, grid)
    η = y_metric(i, j, grid)

    return (i, j, k), (ξ * up, η * vp, wp)
end

"""
$(TYPEDSIGNATURES)

Return the position `(x⁺, y⁺, z⁺)` of a particle that was advected from the cell with
indices `previous_particle_indices` to the position `(x̃, ỹ, z̃)`, after applying particle
`boundary_conditions`, and whether the particle was absorbed at an [`AbsorbingBoundary`](@ref).
"""
@inline function enforce_particle_boundary_conditions((x̃, ỹ, z̃), previous_particle_indices, boundary_conditions, grid)
    i, j, k = previous_particle_indices

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
    bcs = boundary_conditions
    x⁺, absorbed_x = enforce_boundary_conditions(tx, x̃, xᴸ, xᴿ, bcs.west, bcs.east)
    y⁺, absorbed_y = enforce_boundary_conditions(ty, ỹ, yᴸ, yᴿ, bcs.south, bcs.north)
    z⁺, absorbed_z = enforce_boundary_conditions(tz, z̃, zᴸ, zᴿ, bcs.bottom, bcs.top)
    absorbed = absorbed_x | absorbed_y | absorbed_z

    if grid isa ImmersedBoundaryGrid
        (x⁺, y⁺, z⁺), absorbed_immersed = bounce_immersed_particle((x⁺, y⁺, z⁺), (x̃, ỹ, z̃), grid, bcs.immersed,
                                                                   previous_particle_indices)
        absorbed = absorbed | absorbed_immersed
    end

    return (x⁺, y⁺, z⁺), absorbed
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

# The vertical position of drogued particles is fixed at `depths`; see `DroguedParticleDynamics`.
drogue_depths(dynamics) = nothing

@inline particle_z(particles, p, ::Nothing) = @inbounds particles.z[p]
@inline particle_z(particles, p, depths) = @inbounds depths[p]

@inline store_particle_z!(particles, p, z, ::Nothing) = @inbounds particles.z[p] = z
@inline store_particle_z!(particles, p, z, depths) = nothing

# Advance `particles` through one `stage` of a time-stepping scheme. `state` holds per-particle
# storage for the scheme (or `nothing`), and `absorbed` flags the particles that were absorbed at an
# `AbsorbingBoundary` during the current time step (or `nothing`). Absorbed particles stop
# moving until they are removed at the end of the time step.
@kernel function _advect_particles!(particles, state, stage, boundary_conditions, absorbed,
                                    grid::AbstractGrid, Δt, velocities, depths)
    p = @index(Global)

    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
    end

    z = particle_z(particles, p, depths)

    indices, Ẋ = particle_coordinate_velocity((x, y, z), particles, p, grid, velocities)
    X̃ = advance_particle_position(stage, (x, y, z), Ẋ, Δt, state, p)
    (x⁺, y⁺, z⁺), absorbed_now = enforce_particle_boundary_conditions(X̃, indices, boundary_conditions, grid)

    absorbed_before = previously_absorbed(absorbed, p)

    @inbounds begin
        particles.x[p] = ifelse(absorbed_before, x, x⁺)
        particles.y[p] = ifelse(absorbed_before, y, y⁺)
    end

    store_particle_z!(particles, p, ifelse(absorbed_before, z, z⁺), depths)
    record_absorption!(absorbed, p, absorbed_before | absorbed_now)
end

"""
$(TYPEDSIGNATURES)

Advance the positions of `particles` through one `stage` of a time-stepping scheme, using the
velocities of `model`. `stage` is returned by `particle_stage` and determines the scheme
through `advance_particle_position`. `Δt` is the full time step.
"""
function advect_lagrangian_particles!(particles, model, Δt, stage)
    grid = model.grid
    arch = architecture(grid)
    Np = length(particles)
    Np == 0 && return nothing

    launch!(arch, grid, KernelParameters(1:Np),
            _advect_particles!,
            particles.properties, time_stepper_state(particles.timestepper), stage,
            particles.boundary_conditions, absorbed_flags(particles.removal),
            grid, Δt, total_velocities(model), drogue_depths(particles.dynamics))

    return nothing
end
