using Oceananigans.Utils: instantiate
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

Return the new position `(x⁺, y⁺, z⁺)` of a particle at position `(x, y, z)`, given `velocities`,
time-step `Δt`, and particle `boundary_conditions`, and whether the particle was absorbed
at an [`AbsorbingBoundary`](@ref).
"""
@inline function advect_particle((x, y, z), particles, p, boundary_conditions, grid, Δt, velocities)
    X = flattened_node((x, y, z), grid)

    # Obtain current particle indices, looking at the interfaces
    fi = FractionalIndices(X, grid, f, f, f)

    i, _, _ = interpolator(fi.i)
    j, _, _ = interpolator(fi.j)
    k, _, _ = interpolator(fi.k)

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

    # Advected position, before boundary conditions are applied
    x̃ = x + ξ * up * Δt
    ỹ = y + η * vp * Δt
    z̃ = z +     wp * Δt

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
        previous_particle_indices = current_particle_indices # particle has been advected
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

"""
$(TYPEDSIGNATURES)

Advect particle `p` of a species with properties `particles` by `Δt` with a forward Euler step.
`flags` are the transfer flags of the species (or `nothing`; see [`ParticleTransfers`](@ref)),
and particles absorbed at an [`AbsorbingBoundary`](@ref) are flagged for removal.
"""
@inline function advect_particle!((particles, flags), p, (boundary_conditions, grid, Δt, velocities))
    @inbounds begin
        x = particles.x[p]
        y = particles.y[p]
        z = particles.z[p]
    end

    (x⁺, y⁺, z⁺), absorbed = advect_particle((x, y, z), particles, p, boundary_conditions, grid, Δt, velocities)

    @inbounds begin
        particles.x[p] = x⁺
        particles.y[p] = y⁺
        particles.z[p] = z⁺
    end

    record_absorption!(flags, p, absorbed)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Advect the particles of all species by `Δt` with the velocities of `model`, in a single kernel launch
(see [`SuperrangeLaunch`](@ref)).
"""
function advect_lagrangian_particles!(particles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    properties = particles.properties
    species = particle_species(properties)
    flags = species_tuple(properties, transfer_flags(particles.transfers))

    species_args = map(tuple, species, flags)
    shared_args = (particles.boundary_conditions, grid, Δt, total_velocities(model))
    launch_species!(arch, advect_particle!, map(length, species), species_args, shared_args)

    return nothing
end
