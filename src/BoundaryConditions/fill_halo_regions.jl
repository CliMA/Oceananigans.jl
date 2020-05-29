#####
##### General halo filling functions
#####

@inline fill_halo_regions!(::Nothing, args...) = nothing

"""
    fill_halo_regions!(fields, arch)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
@inline function fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)
    for field in fields
        fill_halo_regions!(field, arch, args...)
    end
    return nothing
end

@inline fill_halo_regions!(field, arch, args...) =
    fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, args...)

"Fill halo regions in x, y, and z for a given field."
@inline function fill_halo_regions!(c::AbstractArray, fieldbcs, arch, args...)
      fill_west_halo!(c, fieldbcs.x.left,   arch, args...)
      fill_east_halo!(c, fieldbcs.x.right,  arch, args...)
     fill_south_halo!(c, fieldbcs.y.left,   arch, args...)
     fill_north_halo!(c, fieldbcs.y.right,  arch, args...)
    fill_bottom_halo!(c, fieldbcs.z.bottom, arch, args...)
       fill_top_halo!(c, fieldbcs.z.top,    arch, args...)
    return nothing
end

#####
##### Halo filling for flux, periodic, and no-penetration boundary conditions.
#####

# For flux boundary conditions we fill halos as for a *no-flux* boundary condition, and add the
# flux divergence associated with the flux boundary condition in a separate step. Note that
# ranges are used to reference the data copied into halos, as this produces views of the correct
# dimension (eg size = (1, Ny, Nz) for the west halos).

  @inline _fill_west_halo!(c, ::FBC, H, N) = @views @. c.parent[1:H, :, :] = c.parent[1+H:1+H,  :, :]
 @inline _fill_south_halo!(c, ::FBC, H, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, 1+H:1+H,  :]
@inline _fill_bottom_halo!(c, ::FBC, H, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :,  1+H:1+H]

 @inline _fill_east_halo!(c, ::FBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[N+H:N+H, :, :]
@inline _fill_north_halo!(c, ::FBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, N+H:N+H, :]
  @inline _fill_top_halo!(c, ::FBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, N+H:N+H]

# Periodic boundary conditions
  @inline _fill_west_halo!(c, ::PBC, H, N) = @views @. c.parent[1:H, :, :] = c.parent[N+1:N+H, :, :]
 @inline _fill_south_halo!(c, ::PBC, H, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, N+1:N+H, :]
@inline _fill_bottom_halo!(c, ::PBC, H, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :, N+1:N+H]

 @inline _fill_east_halo!(c, ::PBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[1+H:2H, :, :]
@inline _fill_north_halo!(c, ::PBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, 1+H:2H, :]
  @inline _fill_top_halo!(c, ::PBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, 1+H:2H]

# Generate functions that implement flux and periodic boundary conditions
sides = (:west, :east, :south, :north, :top, :bottom)
coords = (:x, :x, :y, :y, :z, :z)

for (x, side) in zip(coords, sides)
    outername = Symbol(:fill_, side, :_halo!)
    innername = Symbol(:_fill_, side, :_halo!)
    H = Symbol(:H, x)
    N = Symbol(:N, x)
    @eval begin
        @inline $outername(c, bc::Union{FBC, PBC}, arch, grid, args...) =
            $innername(c, bc, grid.$(H), grid.$(N))
    end
end
