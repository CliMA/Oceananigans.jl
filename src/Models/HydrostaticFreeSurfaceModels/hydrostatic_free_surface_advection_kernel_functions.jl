# Before the Shared memory craze

""" Calculate advection of prognostic quantities. """
function calculate_hydrostatic_free_surface_shared_advection_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid
    Nx, Ny, Nz = N = size(grid)

    barrier = device_event(model)

    Ix = gcd(16,   Nx)
    Iy = gcd(12,   Ny)
    Iz = gcd(1680, Nz)

    workgroup = (min(Ix, Nx),  min(Iy, Ny),  1)
    worksize  = N

    halo = halo_size(grid)
    disp = min.(size(grid), halo)

    advection_contribution! = _calculate_hydrostatic_free_surface_XY_advection!(Architectures.device(arch), workgroup, worksize)
    advection_event         = advection_contribution!(model.timestepper.Gⁿ,
                                                      grid,
                                                      model.advection,
                                                      model.velocities,
                                                      model.tracers,
                                                      Val(disp[1]), Val(disp[2]),
                                                      Val(halo[1]), Val(halo[2]);
                                                      dependencies = barrier)
    
    wait(device(arch), advection_event)
    
    workgroup = (1, 1, min(Iz, Nz))
    worksize  = N
    
    advection_contribution! = _calculate_hydrostatic_free_surface_Z_advection!(Architectures.device(arch), workgroup, worksize)
    advection_event         = advection_contribution!(model.timestepper.Gⁿ,
                                                      grid,
                                                      model.advection,
                                                      model.velocities,
                                                      model.tracers, 
                                                      Val(disp[3]), Val(halo[3]);
                                                      dependencies = barrier)

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

@inline @propagate_inbounds Base.getindex(v::DisplacedXYSharedArray, i, j, k)       = v.s_array[i + v.i, j + v.j]
@inline @propagate_inbounds Base.setindex!(v::DisplacedXYSharedArray, val, i, j, k) = setindex!(v.s_array, val, i + v.i, j + v.j)

struct DisplacedZSharedArray{V, K} 
    s_array :: V
    k :: K
end

@inline @propagate_inbounds Base.getindex(v::DisplacedZSharedArray, i, j, k)       = v.s_array[k + v.k]
@inline @propagate_inbounds Base.setindex!(v::DisplacedZSharedArray, val, i, j, k) = setindex!(v.s_array, val, k + v.k)

@kernel function _calculate_hydrostatic_free_surface_Z_advection!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, 
                                                                  tracers, ::Val{H3}, ::Val{N3}) where {FT, H3, N3}
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local,  NTuple)
    ib, jb, kb = @index(Group,  NTuple)

    O = @uniform @groupsize()[3]

    il = @localmem Int (1)
    jl = @localmem Int (1)
    kg = @localmem Int (1)
    
    if ks == 1
        il[1] = - i + 2
        jl[1] = - j + 2
        kg[1] = - O * (kb - 1) + N3
    end

    @synchronize

    # ws_array = @localmem FT (2, 2, O+2*N3)
    
    us_array = @localmem FT (O+2*N3)
    vs_array = @localmem FT (O+2*N3)
    cs_array = @localmem FT (O+2*N3)
    
    us = @uniform DisplacedZSharedArray(us_array, kg[1])
    vs = @uniform DisplacedZSharedArray(vs_array, kg[1])
    cs = @uniform DisplacedZSharedArray(cs_array, kg[1])

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

    ntuple(Val(length(tracers))) do n
        Base.@_inline_meta
        tracer = tracers[n]
        @inbounds cs[i, j, k] = tracer[i, j, k]

        # No corners needed for the tracer
        if ks <= H3
            @inbounds cs[i, j, k - H3] = tracer[i, j, k - H3]
        end
        if ks >= O - H3 + 1
            @inbounds cs[i, j, k + H3] = tracer[i, j, k + H3]
        end

        @synchronize

        @inbounds Gⁿ[n+3][i, j, k] -= div_Uc_z(i, j, k, grid, advection[n+1], (u = us, v = vs, w = velocities.w), cs)
    end
end

@kernel function _calculate_hydrostatic_free_surface_XY_advection!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, 
    tracers, ::Val{H1}, ::Val{H2}, ::Val{N1}, ::Val{N2}) where {FT, H1, H2, N1, N2}
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local,  NTuple)
    ib, jb, kb = @index(Group,  NTuple)

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]

    ig = @localmem Int (1)
    jg = @localmem Int (1)    

    if is == 1 && js == 1 && ks == 1
        ig[1] = - N * (ib - 1) + N1
        jg[1] = - M * (jb - 1) + N2
    end

    @synchronize

    us_array = @localmem FT (N+2*N1, M+2*N2)
    vs_array = @localmem FT (N+2*N1, M+2*N2)
    cs_array = @localmem FT (N+2*N1, M+2*N2)

    us = @uniform DisplacedXYSharedArray(us_array, ig[1], jg[1])
    vs = @uniform DisplacedXYSharedArray(vs_array, ig[1], jg[1])
    cs = @uniform DisplacedXYSharedArray(cs_array, ig[1], jg[1])

    fill_horizontal_velocities_shared_memory!(i, j, k, us, vs, velocities, is, js, H1, H2, N, M, advection.momentum)

    @synchronize

    @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u_h(i, j, k, grid, advection.momentum, (u = us, v = vs))
    @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v_h(i, j, k, grid, advection.momentum, (u = us, v = vs))

    ntuple(Val(length(tracers))) do n
        Base.@_inline_meta
        tracer = tracers[n]
        @inbounds cs[i, j, k] = tracer[i, j, k]

        # No corners needed for the tracer
        if is <= H1
            @inbounds cs[i - H1, j, k] = tracer[i - H1, j, k]
        end
        if is >= N - H1 + 1
            @inbounds cs[i + H1, j, k] = tracer[i + H1, j, k]
        end

        # No corners needed for the tracer
        if js <= H2
            @inbounds cs[i, j - H2, k] = tracer[i, j - H2, k]
        end
        if js >= M - H2 + 1
            @inbounds cs[i, j + H2, k] = tracer[i, j + H2, k]
        end

        @synchronize

        @inbounds Gⁿ[n+3][i, j, k] -= div_Uc_x(i, j, k, grid, advection[n+1], (u = us, v = vs), cs) +
                                      div_Uc_y(i, j, k, grid, advection[n+1], (u = us, v = vs), cs)
    end
end

@inline function fill_horizontal_velocities_shared_memory!(i, j, k, us, vs, velocities, is, js, H1, H2, N, M, advection)
    @inbounds us[i, j, k] = velocities.u[i, j, k]
    @inbounds vs[i, j, k] = velocities.v[i, j, k]

    if is <= H1
        @inbounds us[i - H1, j, k] = velocities.u[i - H1, j, k]
        @inbounds vs[i - H1, j, k] = velocities.v[i - H1, j, k]
    end
    if is >= N - H1 + 1
        @inbounds us[i + H1, j, k] = velocities.u[i + H1, j, k]
        @inbounds vs[i + H1, j, k] = velocities.v[i + H1, j, k]
        # Fill the angles because of staggering!
        if js <= H2
            @inbounds us[i + H1, j - H2, k] = velocities.u[i + H1, j - H2, k]
        end
        if js >= M - H2 + 1
            @inbounds us[i + H1, j + H2, k] = velocities.u[i + H1, j + H2, k]
        end
    end

    if js <= H2
        @inbounds us[i, j - H2, k] = velocities.u[i, j - H2, k]
        @inbounds vs[i, j - H2, k] = velocities.v[i, j - H2, k]
    end
    if js >= M - H2 + 1
        @inbounds us[i, j + H2, k] = velocities.u[i, j + H2, k]
        @inbounds vs[i, j + H2, k] = velocities.v[i, j + H2, k]
        # Fill the angles because of staggering!
        if is <= H1
            @inbounds vs[i - H1, j + H2, k] = velocities.v[i - H1, j + H2, k]
        end
        if is >= N - H1 + 1
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
        if js >= M - H2 + 1
            @inbounds us[i - H1, j + H2, k] = velocities.u[i - H1, j + H2, k]
            @inbounds vs[i - H1, j + H2, k] = velocities.v[i - H1, j + H2, k]
        end
    end
    if is >= N - H1 + 1
        @inbounds us[i + H1, j, k] = velocities.u[i + H1, j, k]
        @inbounds vs[i + H1, j, k] = velocities.v[i + H1, j, k]
        # Fill the angles because of staggering!
        if js <= H2
            @inbounds us[i + H1, j - H2, k] = velocities.u[i + H1, j - H2, k]
            @inbounds vs[i + H1, j - H2, k] = velocities.v[i + H1, j - H2, k]
        end
        if js >= M - H2 + 1
            @inbounds us[i + H1, j + H2, k] = velocities.u[i + H1, j + H2, k]
            @inbounds vs[i + H1, j + H2, k] = velocities.v[i + H1, j + H2, k]
        end
    end

    if js <= H2
        @inbounds us[i, j - H2, k] = velocities.u[i, j - H2, k]
        @inbounds vs[i, j - H2, k] = velocities.v[i, j - H2, k]
    end
    if js >= M - H2 + 1
        @inbounds us[i, j + H2, k] = velocities.u[i, j + H2, k]
        @inbounds vs[i, j + H2, k] = velocities.v[i, j + H2, k]
    end
end