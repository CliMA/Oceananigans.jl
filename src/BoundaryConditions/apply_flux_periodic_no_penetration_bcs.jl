#####
##### Halo filling for flux, periodic, and no-penetration boundary conditions.
#####

# For flux boundary conditions we fill halos as for a *no-flux* boundary condition, and add the
# flux divergence associated with the flux boundary condition in a separate step. Note that
# ranges are used to reference the data copied into halos, as this produces views of the correct
# dimension (eg size = (1, Ny, Nz) for the west halos).

  _fill_west_halo!(c, ::FBC, H, N) = @views @. c.parent[1:H, :, :] = c.parent[1+H:1+H,  :, :]
 _fill_south_halo!(c, ::FBC, H, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, 1+H:1+H,  :]
_fill_bottom_halo!(c, ::FBC, H, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :,  1+H:1+H]

 _fill_east_halo!(c, ::FBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[N+H:N+H, :, :]
_fill_north_halo!(c, ::FBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, N+H:N+H, :]
  _fill_top_halo!(c, ::FBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, N+H:N+H]

# Periodic boundary conditions
  _fill_west_halo!(c, ::PBC, H, N) = @views @. c.parent[1:H, :, :] = c.parent[N+1:N+H, :, :]
 _fill_south_halo!(c, ::PBC, H, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, N+1:N+H, :]
_fill_bottom_halo!(c, ::PBC, H, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :, N+1:N+H]

 _fill_east_halo!(c, ::PBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[1+H:2H, :, :]
_fill_north_halo!(c, ::PBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, 1+H:2H, :]
  _fill_top_halo!(c, ::PBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, 1+H:2H]

# Recall that, by convention, the first grid point (k=1) in an array with a no-penetration boundary
# condition lies on the boundary, where as the last grid point (k=Nz) lies in the domain.

  _fill_west_halo!(c, ::NPBC, H, N) = @views @. c.parent[1:1+H, :, :] = 0
 _fill_south_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, 1:1+H, :] = 0
_fill_bottom_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, :, 1:1+H] = 0

 _fill_east_halo!(c, ::NPBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = 0
_fill_north_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = 0
  _fill_top_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = 0

# Generate functions that implement flux, periodic, and no-penetration boundary conditions
sides = (:west, :east, :south, :north, :top, :bottom)
coords = (:x, :x, :y, :y, :z, :z)

for (x, side) in zip(coords, sides)
    outername = Symbol(:fill_, side, :_halo!)
    innername = Symbol(:_fill_, side, :_halo!)
    H = Symbol(:H, x)
    N = Symbol(:N, x)
    @eval begin
        $outername(c, bc::Union{FBC, PBC, NPBC}, arch::AbstractArchitecture, grid::AbstractGrid, args...) =
            $innername(c, bc, grid.$(H), grid.$(N))
    end
end
