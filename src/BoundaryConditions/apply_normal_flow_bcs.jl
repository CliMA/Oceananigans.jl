using Oceananigans.Operators: Δx, Δy, ΔzC, div_xyᶜᶜᵃ, div_xzᶜᵃᶜ, div_yzᵃᶜᶜ

#####
##### Outer functions for setting velocity on boundary and filling halo beyond boundary.
#####

@inline function fill_west_halo!(u, bc::NFBC, arch, grid, clock, state)
    @launch(device(arch), config=launch_config(grid, :yz), set_east_west_u_velocity!(u, 1, bc, grid, clock, state))
    @launch(device(arch), config=launch_config(grid, :yz), _fill_west_halo!(u, bc, grid, state.velocities.v, state.velocities.w))
    return nothing
end

@inline function fill_east_halo!(u, bc::NFBC, arch, grid, clock, state)
    @launch(device(arch), config=launch_config(grid, :yz), set_east_west_u_velocity!(u, grid.Nx + 1, bc, grid, clock, state)) 
    @launch(device(arch), config=launch_config(grid, :yz), _fill_east_halo!(u, bc, grid, state.velocities.v, state.velocities.w))
    return nothing
end

@inline function fill_south_halo!(v, bc::NFBC, arch, grid, clock, state)
    @launch(device(arch), config=launch_config(grid, :xz), set_north_south_v_velocity!(v, 1, bc, grid, clock, state))
    @launch(device(arch), config=launch_config(grid, :xz), _fill_south_halo!(v, bc, grid, state.velocities.u, state.velocities.w))
    return nothing
end

@inline function fill_north_halo!(v, bc::NFBC, arch, grid, clock, state)
    @launch(device(arch), config=launch_config(grid, :xz), set_north_south_v_velocity!(v, grid.Ny + 1, bc, grid, clock, state))
    @launch(device(arch), config=launch_config(grid, :xz), _fill_north_halo!(v, bc, grid, state.velocities.u, state.velocities.v))
    return nothing
end


@inline function fill_bottom_halo!(w, bc::NFBC, arch, grid, clock, state)
    @launch(device(arch), config=launch_config(grid, :xy), set_top_bottom_w_velocity!(w, 1, bc, grid, clock, state))
    @launch(device(arch), config=launch_config(grid, :xy), _fill_bottom_halo!(w, bc, grid, state.velocities.u, state.velocities.v))
    return nothing
end

@inline function fill_top_halo!(w, bc::NFBC, arch, grid, clock, state)
    @launch(device(arch), config=launch_config(grid, :xy), set_top_bottom_w_velocity!(w, grid.Nz + 1, bc, grid, clock, state))
    @launch(device(arch), config=launch_config(grid, :xy), _fill_top_halo!(w, bc, grid, state.velocities.u, state.velocities.v))
    return nothing
end

#####
##### Kernels that use the continuity equation to fill interior points
#####

# No penetration for x-boundaries: fill in x, loop over yz

function _fill_west_halo!(u, ::NFBC, grid, v, w)
    @loop_yz j k grid begin
        @unroll for i in 0 : -1 : (1 - grid.Hx) # integrate westward
            @inbounds u[i, j, k] = u[i+1, j, k] + Δx(0, j, k, grid) * div_yzᵃᶜᶜ(i, j, k, grid, v, w)
        end
    end
    return nothing
end

function _fill_east_halo!(u, ::NFBC, grid, v, w)
    @loop_yz j k grid begin
        @unroll for i in (grid.Nx + 2) : (grid.Nx + 1 + grid.Hx) # integrate eastward, using face spacing 
            @inbounds u[i, j, k] = u[i-1, j, k] - Δx(grid.Nx+1, j, k, grid) * div_yzᵃᶜᶜ(i-1, j, k, grid, v, w)
        end
    end
    return nothing
end

# No penetration for y-boundaries: fill in y, loop over xz

function _fill_south_halo!(v, ::NFBC, grid, u, w)
    @loop_xz i k grid begin
        @unroll for j in 0 : -1 : (1 - grid.Hy) # integrate southward
            @inbounds v[i, j, k] = v[i, j+1, k] + Δy(i, 0, k, grid) * div_xzᶜᵃᶜ(i, j, k, grid, u, w)
        end
    end
    return nothing
end

function _fill_north_halo!(v, ::NFBC, grid, u, w)
    @loop_xz i k grid begin
        @unroll for j in (grid.Ny + 2) : (grid.Ny + 1 + grid.Hy) # integrate northward
            @inbounds v[i, j, k] = v[i, j-1, k] - Δy(i, grid.Ny+1, k, grid) * div_xzᶜᵃᶜ(i, j-1, k, grid, u, w)
        end
    end
    return nothing
end

# No penetration for z-boundaries: fill in z, loop over xy

function _fill_bottom_halo!(w, ::NFBC, grid, u, v)
    @loop_xy i j grid begin
        @unroll for k in 0 : -1 : (1 - grid.Hz) # integrate downwards
            @inbounds w[i, j, k] = w[i, j, k+1] + ΔzF(i, j, 0, grid) * div_xyᶜᶜᵃ(i, j, k, grid, u, v)
        end
    end
    return nothing
end

function _fill_top_halo!(w, ::NFBC, grid, u, v)
    @loop_xy i j grid begin
        @unroll for k in (grid.Nz + 2) : (grid.Nz + 1 + grid.Hz) # integrate upwards
            @inbounds w[i, j, k] = w[i, j, k-1] - ΔzF(i, j, grid.Nz+1, grid) * div_xyᶜᶜᵃ(i, j, k-1, grid, u, v)
        end
    end
    return nothing
end

function set_east_west_u_velocity!(u, i_boundary, bc, grid, args...)
    @loop_yz j k grid begin
        @inbounds u[i_boundary, j, k] = getbc(bc, j, k, grid, args...)
    end
    return nothing
end

function set_north_south_v_velocity!(v, j_boundary, bc, grid, args...)
    @loop_xz i k grid begin
        @inbounds v[i, j_boundary, k] = getbc(bc, i, k, grid, args...)
    end
    return nothing
end

function set_top_bottom_w_velocity!(w, k_boundary, bc, grid, args...)
    @loop_xy i j grid begin
        @inbounds w[i, j, k_boundary] = getbc(bc, i, j, grid, args...)
    end
    return nothing
end
