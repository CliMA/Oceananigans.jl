# Halo filling algorithm for arbitrary flux, gradient, and value boundary conditions.
# For these we fill halos so that no-fluxes are added to source terms when the PDE / 
# associated differential operators are applied on boundary elements.
# Fluxes associated with the boundary condition are added in a separate step in the time-stepping
# algorithm.
 fill_west_halo!(ϕ, ::BC, H, N) = @views @. ϕ.parent[1:H, :, :] = ϕ.parent[1+H:1+H,  :, :]
fill_south_halo!(ϕ, ::BC, H, N) = @views @. ϕ.parent[:, 1:H, :] = ϕ.parent[:, 1+H:1+H,  :]
  fill_top_halo!(ϕ, ::BC, H, N) = @views @. ϕ.parent[:, :, 1:H] = ϕ.parent[:, :,  1+H:1+H]

  fill_east_halo!(ϕ, ::BC, H, N) = @views @. ϕ.parent[N+H+1:N+2H, :, :] = ϕ.parent[N+H:N+H, :, :]
 fill_north_halo!(ϕ, ::BC, H, N) = @views @. ϕ.parent[:, N+H+1:N+2H, :] = ϕ.parent[:, N+H:N+H, :]
fill_bottom_halo!(ϕ, ::BC, H, N) = @views @. ϕ.parent[:, :, N+H+1:N+2H] = ϕ.parent[:, :, N+H:N+H]

# Halo filling for periodic boundary conditions
const PBC = BoundaryCondition{<:Periodic}
 fill_west_halo!(ϕ, ::PBC, H, N) = @views @. ϕ.parent[1:H, :, :] = ϕ.parent[N+1:N+H, :, :]
fill_south_halo!(ϕ, ::PBC, H, N) = @views @. ϕ.parent[:, 1:H, :] = ϕ.parent[:, N+1:N+H, :]
  fill_top_halo!(ϕ, ::PBC, H, N) = @views @. ϕ.parent[:, :, 1:H] = ϕ.parent[:, :, N+1:N+H]

  fill_east_halo!(ϕ, ::PBC, H, N) = @views @. ϕ.parent[N+H+1:N+2H, :, :] = ϕ.parent[1+H:2H, :, :]
 fill_north_halo!(ϕ, ::PBC, H, N) = @views @. ϕ.parent[:, N+H+1:N+2H, :] = ϕ.parent[:, 1+H:2H, :]
fill_bottom_halo!(ϕ, ::PBC, H, N) = @views @. ϕ.parent[:, :, N+H+1:N+2H] = ϕ.parent[:, :, 1+H:2H]

# Halo filling for no-penetration boundary conditions. Recall that, by convention,
# the first grid point in an array with no penetration boundary condition
# lies on the boundary, where as the final grid point lies in the domain.
const NPBC = BoundaryCondition{<:NoPenetration}
 fill_west_halo!(ϕ, ::NPBC, H, N) = @views @. ϕ.parent[1:H+1, :, :] = 0
fill_south_halo!(ϕ, ::NPBC, H, N) = @views @. ϕ.parent[:, 1:H+1, :] = 0
  fill_top_halo!(ϕ, ::NPBC, H, N) = @views @. ϕ.parent[:, :, 1:H+1] = 0

  fill_east_halo!(ϕ, ::NPBC, H, N) = @views @. ϕ.parent[N+H+1:N+2H, :, :] = 0
 fill_north_halo!(ϕ, ::NPBC, H, N) = @views @. ϕ.parent[:, N+H+1:N+2H, :] = 0
fill_bottom_halo!(ϕ, ::NPBC, H, N) = @views @. ϕ.parent[:, :, N+H+1:N+2H] = 0

"Fill east and west halo regions."
function fill_x_halo_regions!(ϕ, bcs, grid)
    fill_west_halo!(ϕ, bcs.left, grid.Hx, grid.Nx)
    fill_east_halo!(ϕ, bcs.right, grid.Hx, grid.Nx)
    return nothing
end

"Fill north and south halo regions."
function fill_y_halo_regions!(ϕ, bcs, grid)
    fill_south_halo!(ϕ, bcs.left, grid.Hy, grid.Ny)
    fill_north_halo!(ϕ, bcs.right, grid.Hy, grid.Ny)
    return nothing
end

"Fill top and bottom halo regions."
function fill_z_halo_regions!(ϕ, bcs, grid)
    fill_top_halo!(ϕ, bcs.left, grid.Hz, grid.Nz)
    fill_bottom_halo!(ϕ, bcs.right, grid.Hz, grid.Nz)
    return nothing
end

"Fill halo regions in x, y, and z for a given field."
function fill_halo_regions!(field::AbstractArray, fieldbcs, grid)
    fill_x_halo_regions!(field, fieldbcs.x, grid)
    fill_y_halo_regions!(field, fieldbcs.y, grid)
    #fill_z_halo_regions!(field, fieldbcs.z, grid)
    return nothing
end

"""
    fill_halo_regions!(fields, bcs, grid)

Fill halo regions for all fields in the tuple `fields` according 
to the corresponding tuple of `bcs`.
"""
function fill_halo_regions!(fields::NamedTuple, bcs, grid)
    for i = 1:length(fields)
        fill_halo_regions!(fields[i], bcs[i], grid)
    end
    return nothing
end
