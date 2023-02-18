# Before the Shared memory craze
""" Calculate advection of prognostic quantities. """
function calculate_hydrostatic_free_surface_shared_advection_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid
    Nx, Ny, Nz = N = size(grid)

    barrier = device_event(model)

    Ix = gcd(16,   Nx)
    Iy = gcd(12,   Ny)

    workgroup = (min(Ix, Nx),  min(Iy, Ny),  1)
    worksize  = N

    halo = halo_size(grid)
    disp = min.(size(grid), halo)

    advection_contribution! = _calculate_hydrostatic_free_surface_XY_advection!(Architectures.device(arch), workgroup, worksize)
    advection_event         = advection_contribution!(model.timestepper.Gⁿ,
                                                      grid,
                                                      model.advection,
                                                      model.velocities,
                                                      Val(Int32(disp[1])), Val(Int32(disp[2])),
                                                      Val(Int32(halo[1])), Val(Int32(halo[2])),
                                                      Val(Int32(workgroup[1])), Val(Int32(workgroup[2]));
                                                      dependencies = barrier)
    
    wait(device(arch), advection_event)
    
    # workgroup = (1, 1, Nz)
    # worksize  = N
    
    # advection_contribution! = _calculate_hydrostatic_free_surface_Z_advection!(Architectures.device(arch), workgroup, worksize)
    # advection_event         = advection_contribution!(model.timestepper.Gⁿ,
    #                                                   grid,
    #                                                   model.advection,
    #                                                   model.velocities,
    #                                                   Val(disp[3]), Val(halo[3]), Val(workgroup[3]);
    #                                                   dependencies = barrier)

    wait(device(arch), advection_event)

    return nothing
end

import Base: getindex, setindex!
using Base: @propagate_inbounds

using Oceananigans.Advection: U_dot_∇u_h, U_dot_∇u_z, U_dot_∇v_h, U_dot_∇v_z
using Oceananigans.Advection: div_Uc_x, div_Uc_y, div_Uc_z

struct DisplacedXYSharedArray{V, I, J} 
    s_array :: V
    i :: I
    j :: J
end

@inline @propagate_inbounds Base.getindex(v::DisplacedXYSharedArray, i, j, k)       = @inbounds v.s_array[i + v.i, j + v.j]
@inline @propagate_inbounds Base.setindex!(v::DisplacedXYSharedArray, val, i, j, k) = setindex!(v.s_array, val, i + v.i, j + v.j)

struct DisplacedZSharedArray{V, K} 
    s_array :: V
    k :: K
end

@inline @propagate_inbounds Base.getindex(v::DisplacedZSharedArray, i, j, k)       = @inbounds v.s_array[k + v.k]
@inline @propagate_inbounds Base.setindex!(v::DisplacedZSharedArray, val, i, j, k) = setindex!(v.s_array, val, k + v.k)

@kernel function _calculate_hydrostatic_free_surface_Z_advection!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, 
                                                                   ::Val{H3}, ::Val{N3}, ::Val{O}) where {FT, H3, N3, O}
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local,  NTuple)
    ib, jb, kb = @index(Group,  NTuple)

    kg = - O * (kb - 0x1) + N3
    
    us_array = @localmem FT (O+2*N3)
    vs_array = @localmem FT (O+2*N3)
    
    us = @uniform DisplacedZSharedArray(us_array, kg)
    vs = @uniform DisplacedZSharedArray(vs_array, kg)

    @inbounds us[i, j, k] = velocities.u[i, j, k]
    @inbounds vs[i, j, k] = velocities.v[i, j, k]

    if ks <= H3
        @inbounds us[i, j, k - H3] = velocities.u[i, j, k - H3]
        @inbounds vs[i, j, k - H3] = velocities.v[i, j, k - H3]
    end
    if ks >= O - H3 + 1
        @inbounds us[i, j, k + H3] = velocities.u[i, j, k + H3]
        @inbounds vs[i, j, k + H3] = velocities.v[i, j, k + H3]
    end

    @synchronize

    @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u_z(i, j, k, grid, advection.momentum, (u = us, v = vs, w = velocities.w))
    @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v_z(i, j, k, grid, advection.momentum, (u = us, v = vs, w = velocities.w))
end

@kernel function _calculate_hydrostatic_free_surface_XY_advection!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, 
     ::Val{H1}, ::Val{H2}, ::Val{N1}, ::Val{N2}, ::Val{N}, ::Val{M}) where {FT, H1, H2, N1, N2, N, M}
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local, NTuple)
    ib, jb, kb = @index(Group,  NTuple)

    ig = - N * (ib - 0x1) + N1
    jg = - M * (jb - 0x1) + N2

    us_array = @localmem FT (N+2*N1, M+2*N2)
    vs_array = @localmem FT (N+2*N1, M+2*N2)

    us = @uniform DisplacedXYSharedArray(us_array, ig, jg)
    vs = @uniform DisplacedXYSharedArray(vs_array, ig, jg)

    fill_horizontal_velocities_shared_memory!(i, j, k, us, vs, velocities, is, js, H1, H2, N, M, advection.momentum)

    @synchronize

    @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u_h(i, j, k, grid, advection.momentum, (u = us, v = vs))
    @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v_h(i, j, k, grid, advection.momentum, (u = us, v = vs))
end

@inline function fill_horizontal_velocities_shared_memory!(i, j, k, us, vs, velocities, is, js, H1, H2, N, M, advection)
    @inbounds us[i, j, k] = velocities.u[i, j, k]
    @inbounds vs[i, j, k] = velocities.v[i, j, k]

    if is <= H1
        @inbounds us[i - H1, j, k] = velocities.u[i - H1, j, k]
        @inbounds vs[i - H1, j, k] = velocities.v[i - H1, j, k]
    end
    if is >= N - H1 + 0x1
        @inbounds us[i + H1, j, k] = velocities.u[i + H1, j, k]
        @inbounds vs[i + H1, j, k] = velocities.v[i + H1, j, k]
        # Fill the angles because of staggering!
        if js <= H2
            @inbounds us[i + H1, j - H2, k] = velocities.u[i + H1, j - H2, k]
        end
        if js >= M - H2 + 0x1
            @inbounds us[i + H1, j + H2, k] = velocities.u[i + H1, j + H2, k]
        end
    end

    if js <= H2
        @inbounds us[i, j - H2, k] = velocities.u[i, j - H2, k]
        @inbounds vs[i, j - H2, k] = velocities.v[i, j - H2, k]
    end
    if js >= M - H2 + 0x1
        @inbounds us[i, j + H2, k] = velocities.u[i, j + H2, k]
        @inbounds vs[i, j + H2, k] = velocities.v[i, j + H2, k]
        # Fill the angles because of staggering!
        if is <= H1
            @inbounds vs[i - H1, j + H2, k] = velocities.v[i - H1, j + H2, k]
        end
        if is >= N - H1 + 0x1
            @inbounds vs[i + H1, j + H2, k] = velocities.v[i + H1, j + H2, k]
        end
    end
end

@inline function fill_horizontal_velocities_shared_memory!(i, j, k, us, vs, velocities, is, js, H1, H2, N, M, ::VectorInvariant)
    @inbounds us[i, j, k] = velocities.u[i, j, k]
    @inbounds vs[i, j, k] = velocities.v[i, j, k]

    if is <= H1
        @inbounds us[i - H1, j, k] = velocities.u[i - H1, j, k]
        @inbounds vs[i - H1, j, k] = velocities.v[i - H1, j, k]
        # Fill the angles because of staggering!
        if js <= H2
            @inbounds us[i - H1, j - H2, k] = velocities.u[i - H1, j - H2, k]
            @inbounds vs[i - H1, j - H2, k] = velocities.v[i - H1, j - H2, k]
        end
        if js >= M - H2 + 0x1
            @inbounds us[i - H1, j + H2, k] = velocities.u[i - H1, j + H2, k]
            @inbounds vs[i - H1, j + H2, k] = velocities.v[i - H1, j + H2, k]
        end
    end
    if is >= N - H1 + 0x1
        @inbounds us[i + H1, j, k] = velocities.u[i + H1, j, k]
        @inbounds vs[i + H1, j, k] = velocities.v[i + H1, j, k]
        # Fill the angles because of staggering!
        if js <= H2
            @inbounds us[i + H1, j - H2, k] = velocities.u[i + H1, j - H2, k]
            @inbounds vs[i + H1, j - H2, k] = velocities.v[i + H1, j - H2, k]
        end
        if js >= M - H2 + 0x1
            @inbounds us[i + H1, j + H2, k] = velocities.u[i + H1, j + H2, k]
            @inbounds vs[i + H1, j + H2, k] = velocities.v[i + H1, j + H2, k]
        end
    end

    if js <= H2
        @inbounds us[i, j - H2, k] = velocities.u[i, j - H2, k]
        @inbounds vs[i, j - H2, k] = velocities.v[i, j - H2, k]
    end
    if js >= M - H2 + 0x1
        @inbounds us[i, j + H2, k] = velocities.u[i, j + H2, k]
        @inbounds vs[i, j + H2, k] = velocities.v[i, j + H2, k]
    end
end