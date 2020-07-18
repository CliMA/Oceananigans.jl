#####
##### General halo filling functions
#####

fill_halo_regions!(::Nothing, args...) = []

"""
    fill_halo_regions!(fields, arch)

Fill halo regions for each field in the tuple `fields` according to their boundary
conditions, possibly recursing into `fields` if it is a nested tuple-of-tuples.
"""
function fill_halo_regions!(fields::Union{Tuple, NamedTuple}, arch, args...)

    for field in fields
        fill_halo_regions!(field, arch, args...)
    end

    return nothing
end

fill_halo_regions!(field, arch, args...) =
    fill_halo_regions!(field.data, field.boundary_conditions, arch, field.grid, args...)

"Fill halo regions in x, y, and z for a given field."
function fill_halo_regions!(c::AbstractArray, fieldbcs, arch, grid, args...)

      west_event =   fill_west_halo!(c, fieldbcs.x.left,   arch, grid, args...)
      east_event =   fill_east_halo!(c, fieldbcs.x.right,  arch, grid, args...)
     south_event =  fill_south_halo!(c, fieldbcs.y.left,   arch, grid, args...)
     north_event =  fill_north_halo!(c, fieldbcs.y.right,  arch, grid, args...)
    bottom_event = fill_bottom_halo!(c, fieldbcs.z.bottom, arch, grid, args...)
       top_event =    fill_top_halo!(c, fieldbcs.z.top,    arch, grid, args...)

    events = [west_event, east_event, south_event, north_event, bottom_event, top_event]
    events = filter(e -> typeof(e) <: Base.Event, events)
    wait(device(arch), MultiEvent(Tuple(events)))

    return nothing
end

#####
##### Halo filling for flux and periodic boundary conditions
#####

# For flux boundary conditions we fill halos as for a *no-flux* boundary condition, and add the
# flux divergence associated with the flux boundary condition in a separate step. Note that
# ranges are used to reference the data copied into halos, as this produces views of the correct
# dimension (eg size = (1, Ny, Nz) for the west halos).

  fill_west_halo!(c, ::FBC, H::Int, N) = @views @. c.parent[1:H, :, :] = c.parent[1+H:1+H,  :, :]
 fill_south_halo!(c, ::FBC, H::Int, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, 1+H:1+H,  :]
fill_bottom_halo!(c, ::FBC, H::Int, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :,  1+H:1+H]

 fill_east_halo!(c, ::FBC, H::Int, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[N+H:N+H, :, :]
fill_north_halo!(c, ::FBC, H::Int, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, N+H:N+H, :]
  fill_top_halo!(c, ::FBC, H::Int, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, N+H:N+H]

# Periodic boundary conditions
  fill_west_halo!(c, ::PBC, H::Int, N) = @views @. c.parent[1:H, :, :] = c.parent[N+1:N+H, :, :]
 fill_south_halo!(c, ::PBC, H::Int, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, N+1:N+H, :]
fill_bottom_halo!(c, ::PBC, H::Int, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :, N+1:N+H]

 fill_east_halo!(c, ::PBC, H::Int, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[1+H:2H, :, :]
fill_north_halo!(c, ::PBC, H::Int, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, 1+H:2H, :]
  fill_top_halo!(c, ::PBC, H::Int, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, 1+H:2H]

# Generate functions that implement flux and periodic boundary conditions
sides = (:west, :east, :south, :north, :top, :bottom)
coords = (:x, :x, :y, :y, :z, :z)

for (x, side) in zip(coords, sides)
    name = Symbol(:fill_, side, :_halo!)
    H = Symbol(:H, x)
    N = Symbol(:N, x)
    @eval begin
        $name(c, bc::Union{FBC, PBC}, arch, grid, args...) =
            $name(c, bc, grid.$(H), grid.$(N))
    end
end
