#####
##### Halo filling for flux, periodic, and no-penetration boundary conditions.
#####

# For flux boundary conditions we fill halos as for a *no-flux* boundary condition, and add the
# flux divergence associated with the flux boundary condition in a separate step. Note that
# ranges are used to reference the data copied into halos, as this produces views of the correct
# dimension (eg size = (1, Ny, Nz) for the west halos).

 _fill_west_halo!(c, ::FBC, H, N) = @views @. c.parent[1:H, :, :] = c.parent[1+H:1+H,  :, :]
_fill_south_halo!(c, ::FBC, H, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, 1+H:1+H,  :]
  _fill_top_halo!(c, ::FBC, H, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :,  1+H:1+H]

  _fill_east_halo!(c, ::FBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[N+H:N+H, :, :]
 _fill_north_halo!(c, ::FBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, N+H:N+H, :]
_fill_bottom_halo!(c, ::FBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, N+H:N+H]

# Periodic boundary conditions
 _fill_west_halo!(c, ::PBC, H, N) = @views @. c.parent[1:H, :, :] = c.parent[N+1:N+H, :, :]
_fill_south_halo!(c, ::PBC, H, N) = @views @. c.parent[:, 1:H, :] = c.parent[:, N+1:N+H, :]
  _fill_top_halo!(c, ::PBC, H, N) = @views @. c.parent[:, :, 1:H] = c.parent[:, :, N+1:N+H]

  _fill_east_halo!(c, ::PBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = c.parent[1+H:2H, :, :]
 _fill_north_halo!(c, ::PBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = c.parent[:, 1+H:2H, :]
_fill_bottom_halo!(c, ::PBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = c.parent[:, :, 1+H:2H]

# Recall that, by convention, the first grid point in an array with no penetration boundary
# condition lies on the boundary, where as the final grid point lies in the domain.

 _fill_west_halo!(c, ::NPBC, H, N) = @views @. c.parent[1:H+1, :, :] = 0
_fill_south_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, 1:H+1, :] = 0
  _fill_top_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, :, 1:H+1] = 0

  _fill_east_halo!(c, ::NPBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = 0
 _fill_north_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = 0
_fill_bottom_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = 0

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

#####
##### Halo filling for value and gradient boundary conditions
#####

@inline linearly_extrapolate(c₀, ∇c, Δ) = c₀ + ∇c * Δ

@inline top_gradient(bc::GBC, c¹, Δ, i, j, args...) = getbc(bc, i, j, args...)
@inline bottom_gradient(bc::GBC, cᴺ, Δ, i, j, args...) = getbc(bc, i, j, args...)

@inline top_gradient(bc::VBC, c¹, Δ, i, j, args...) =    ( getbc(bc, i, j, args...) - c¹ ) / (Δ/2)
@inline bottom_gradient(bc::VBC, cᴺ, Δ, i, j, args...) = ( cᴺ - getbc(bc, i, j, args...) ) / (Δ/2)

function _fill_top_halo!(c, bc::Union{VBC, GBC}, grid, args...)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds ∇c = top_gradient(bc, c[i, j, 1], grid.Δz, i, j, grid, args...)
            @unroll for k in (1-grid.Hz):0
                Δ = (1-k) * grid.Δz
                @inbounds c[i, j, k] = linearly_extrapolate(c[i, j, 1], ∇c, Δ)
            end
        end
    end
    return
end

function _fill_bottom_halo!(c, bc::Union{VBC, GBC}, grid, args...)
    @loop for j in (1:grid.Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
        @loop for i in (1:grid.Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
            @inbounds ∇c = bottom_gradient(bc, c[i, j, grid.Nz], grid.Δz, i, j, grid, args...)
            @unroll for k in grid.Nz+1:grid.Nz+grid.Hz
                Δ = (grid.Nz-k) * grid.Δz # separation between bottom grid cell and halo is negative
                @inbounds c[i, j, k] = linearly_extrapolate(c[i, j, grid.Nz], ∇c, Δ)
            end
        end
    end
    return
end

function fill_top_halo!(c, bc::Union{VBC, GBC}, arch, grid, args...)
    @launch device(arch) config=launch_config(grid, :xy) _fill_top_halo!(c, bc, grid, args...)
    return
end

function fill_bottom_halo!(c, bc::Union{VBC, GBC}, arch::AbstractArchitecture, grid::AbstractGrid, args...)
    @launch device(arch) config=launch_config(grid, :xy) _fill_bottom_halo!(c, bc, grid, args...)
    return
end

#####
##### General halo filling functions
#####

"Fill halo regions in x, y, and z for a given field."
function fill_halo_regions!(c::AbstractArray, fieldbcs, arch, grid, args...)

      fill_west_halo!(c, fieldbcs.x.left,  arch, grid, args...)
      fill_east_halo!(c, fieldbcs.x.right, arch, grid, args...)

     fill_south_halo!(c, fieldbcs.y.left,  arch, grid, args...)
     fill_north_halo!(c, fieldbcs.y.right, arch, grid, args...)

       fill_top_halo!(c, fieldbcs.z.left,  arch, grid, args...)
    fill_bottom_halo!(c, fieldbcs.z.right, arch, grid, args...)

    return
end

"""
    fill_halo_regions!(fields, bcs, arch, grid)

Fill halo regions for all fields in the tuple `fields` according
to the corresponding tuple of `bcs`.
"""
function fill_halo_regions!(fields::NamedTuple, bcs, arch, grid, args...)
    for (field, fieldbcs) in zip(fields, bcs)
        fill_halo_regions!(field, fieldbcs, arch, grid, args...)
    end
    return
end

"""
    fill_halo_regions!(fields, bcs, arch, grid)

Fill halo regions for each field in the tuple `fields` according
to the single instances of `FieldBoundaryConditions` in `bcs`.
"""
function fill_halo_regions!(fields::NamedTuple, bcs::FieldBoundaryConditions, arch, grid, args...)
    for field in fields
        fill_halo_regions!(field, bcs, arch, grid, args...)
    end
end

fill_halo_regions!(::Nothing, args...) = nothing

####
#### Halo zeroing functions
####

 zero_west_halo!(c, H, N) = @views @. c[1:H, :, :] = 0
zero_south_halo!(c, H, N) = @views @. c[:, 1:H, :] = 0
  zero_top_halo!(c, H, N) = @views @. c[:, :, 1:H] = 0

  zero_east_halo!(c, H, N) = @views @. c[N+H+1:N+2H, :, :] = 0
 zero_north_halo!(c, H, N) = @views @. c[:, N+H+1:N+2H, :] = 0
zero_bottom_halo!(c, H, N) = @views @. c[:, :, N+H+1:N+2H] = 0

function zero_halo_regions!(c::AbstractArray, grid)
      zero_west_halo!(c, grid.Hx, grid.Nx)
      zero_east_halo!(c, grid.Hx, grid.Nx)
     zero_south_halo!(c, grid.Hy, grid.Ny)
     zero_north_halo!(c, grid.Hy, grid.Ny)
       zero_top_halo!(c, grid.Hz, grid.Nz)
    zero_bottom_halo!(c, grid.Hz, grid.Nz)
    return
end

function zero_halo_regions!(fields::Tuple, grid)
    for field in fields
        zero_halo_regions!(field, grid)
    end
    return
end
