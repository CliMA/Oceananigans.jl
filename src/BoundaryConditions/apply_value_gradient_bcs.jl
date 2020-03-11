using GPUifyLoops: @launch, @loop, @unroll
using Oceananigans.Utils: @loop_xy, @loop_xz, launch_config

#####
##### Halo filling for value and gradient boundary conditions
#####

function fill_bottom_halo!(c, bc::Union{VBC, GBC}, arch, grid, args...)
    @launch device(arch) config=launch_config(grid, :xy) _fill_bottom_halo!(c, bc, grid, args...)
    return nothing
end

function fill_top_halo!(c, bc::Union{VBC, GBC}, arch, grid, args...)
    @launch device(arch) config=launch_config(grid, :xy) _fill_top_halo!(c, bc, grid, args...)
    return nothing
end

function fill_south_halo!(c, bc::Union{VBC, GBC}, arch, grid, args...)
    @launch device(arch) config=launch_config(grid, :xz) _fill_south_halo!(c, bc, grid, args...)
    return nothing
end

function fill_north_halo!(c, bc::Union{VBC, GBC}, arch, grid, args...)
    @launch device(arch) config=launch_config(grid, :xz) _fill_north_halo!(c, bc, grid, args...)
    return nothing
end

@inline linearly_extrapolate(c₀, ∇c, Δ) = c₀ + ∇c * Δ

@inline  left_gradient(bc::GBC, c¹, Δ, i, j, args...) = getbc(bc, i, j, args...)
@inline right_gradient(bc::GBC, cᴺ, Δ, i, j, args...) = getbc(bc, i, j, args...)

@inline  left_gradient(bc::VBC, c¹, Δ, i, j, args...) = ( c¹ - getbc(bc, i, j, args...) ) / (Δ/2)
@inline right_gradient(bc::VBC, cᴺ, Δ, i, j, args...) = ( getbc(bc, i, j, args...) - cᴺ ) / (Δ/2)

function _fill_bottom_halo!(c, bc::Union{VBC, GBC}, grid, args...)
    @loop_xy i j grid begin
        @inbounds ∇c = left_gradient(bc, c[i, j, 1], grid.Δz, i, j, grid, args...)
        @unroll for k in (1 - grid.Hz):0
            Δ = (k - 1) * grid.Δz  # separation between bottom grid cell and halo is negative
            @inbounds c[i, j, k] = linearly_extrapolate(c[i, j, 1], ∇c, Δ)
        end
    end
    return nothing
end

function _fill_top_halo!(c, bc::Union{VBC, GBC}, grid, args...)
    @loop_xy i j grid begin
        @inbounds ∇c = right_gradient(bc, c[i, j, grid.Nz], grid.Δz, i, j, grid, args...)
        @unroll for k in (grid.Nz + 1) : (grid.Nz + grid.Hz)
            Δ = (k - grid.Nz) * grid.Δz
            @inbounds c[i, j, k] = linearly_extrapolate(c[i, j, grid.Nz], ∇c, Δ)
        end
    end
    return nothing
end

function _fill_south_halo!(c, bc::Union{VBC, GBC}, grid, args...)
    @loop_xz i k grid begin
        @inbounds ∇c = left_gradient(bc, c[i, 1, k], grid.Δy, i, k, grid, args...)
        @unroll for j in (1 - grid.Hy):0
            Δ = (j - 1) * grid.Δy  # separation between southern-most grid cell and halo is negative
            @inbounds c[i, j, k] = linearly_extrapolate(c[i, 1, k], ∇c, Δ)
        end
    end
    return nothing
end

function _fill_north_halo!(c, bc::Union{VBC, GBC}, grid, args...)
    @loop_xz i k grid begin
        @inbounds ∇c = right_gradient(bc, c[i, grid.Ny, k], grid.Δy, i, k, grid, args...)
        @unroll for j in (grid.Ny + 1) : (grid.Ny + grid.Hy)
            Δ = (j - grid.Ny) * grid.Δy
            @inbounds c[i, j, k] = linearly_extrapolate(c[i, grid.Ny, k], ∇c, Δ)
        end
    end
    return nothing
end
