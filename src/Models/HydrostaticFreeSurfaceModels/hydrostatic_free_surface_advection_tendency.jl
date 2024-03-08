using Oceananigans.Advection: div_Uc_x, div_Uc_y, div_Uc_z
using Oceananigans.Advection: U_dot_∇u_x, U_dot_∇u_y, U_dot_∇u_z
using Oceananigans.Advection: U_dot_∇v_x, U_dot_∇v_y, U_dot_∇v_z

""" Calculate advection of prognostic quantities. """
function calculate_hydrostatic_free_surface_advection_tendency_contributions!(model)

    arch = model.architecture
    grid = model.grid
    Nx, Ny, Nz = N = size(grid)

    workgroup_x = (min(Nx, 256), 1, 1)
    workgroup_y = (1, min(Ny, 256), 1)
    workgroup_z = (1, 1, min(Nz, 256))
    worksize  = N

    halo = min.(size(grid), halo_size(grid))

    advection_contribution_x! = _calculate_hydrostatic_free_surface_advection_x!(Architectures.device(arch), workgroup_x, worksize)
    advection_contribution_x!(model.timestepper.Gⁿ, grid, model.advection, model.velocities,
                              model.tracers, halo)
    
    advection_contribution_y! = _calculate_hydrostatic_free_surface_advection_y!(Architectures.device(arch), workgroup_y, worksize)
    advection_contribution_y!(model.timestepper.Gⁿ, grid, model.advection, model.velocities,
                              model.tracers, halo)

    advection_contribution_z! = _calculate_hydrostatic_free_surface_advection_z!(Architectures.device(arch), workgroup_z, worksize)
    advection_contribution_z!(model.timestepper.Gⁿ, grid, model.advection, model.velocities,
                              model.tracers, halo)

    return nothing
end

using Oceananigans.Grids: halo_size
using Base: @propagate_inbounds

struct OffsetSharedArray{S}
    s_array :: S
    i :: Int32
    j :: Int32
    k :: Int32
end

@inline @propagate_inbounds Base.getindex(v::OffsetSharedArray, i, j, k) = v.s_array[i + v.i, j + v.j, k + v.k]
@inline @propagate_inbounds Base.setindex!(v::OffsetSharedArray, val, i, j, k) = setindex!(v.s_array, val, i + v.i, j + v.j, k + v.k)

@inline @propagate_inbounds Base.lastindex(v::OffsetSharedArray)      = lastindex(v.s_array)
@inline @propagate_inbounds Base.lastindex(v::OffsetSharedArray, dim) = lastindex(v.s_array, dim)

@kernel function _calculate_hydrostatic_free_surface_advection_x!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, tracers, halo) where FT
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local,  NTuple)

    N = @uniform @groupsize()[1]

    u_shared = @localmem FT (N+2halo[1], 2, 1)
    v_shared = @localmem FT (N+2halo[1], 2, 1) # This shared memory will hold also the tracers

    us = OffsetSharedArray(u_shared, - i + halo[1], - j + 1, - k + 1)
    vs = OffsetSharedArray(v_shared, - i + halo[1], - j + 1, - k + 1)
    cs = OffsetSharedArray(c_shared, - i + halo[1], - j + 1, - k + 1)

    @inbounds u_shared[is+halo[1], 1, 1] = velocities.u[i, j, k]
    @inbounds v_shared[is+halo[1], 1, 1] = velocities.v[i, j, k]
    @inbounds u_shared[is+halo[1], 2, 1] = velocities.u[i, j+1, k]
    @inbounds v_shared[is+halo[1], 2, 1] = velocities.v[i, j+1, k]

    if is <= halo[1]
        @inbounds u_shared[is, 1, 1] = velocities.u[i - halo[1], j, k]
        @inbounds v_shared[is, 1, 1] = velocities.v[i - halo[1], j, k]
    end
    if is >= N - halo[1] + 1
        @inbounds u_shared[is+2halo[1], 1, 1] = velocities.u[i + halo[1], j, k]
        @inbounds v_shared[is+2halo[1], 1, 1] = velocities.v[i + halo[1], j, k]
    end

    @synchronize
    
    @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u_x(i, j, k, grid, advection.momentum, (u = us, v = vs))
    @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v_x(i, j, k, grid, advection.momentum, (u = us, v = vs))

    ntuple(Val(length(tracers))) do n
        Base.@_inline_meta
        @inbounds v_shared[is + halo[1], 1, 1] = tracer[i, j, k]

        if is <= halo
            @inbounds v_shared[is, 1, 1] = tracer[i - halo, j, k]
        end
        if is >= N - halo + 1
            @inbounds v_shared[is + 2halo[1], 1, 1] = tracer[i + halo, j, k]
        end

        @synchronize

        @inbounds Gⁿ[n+3] = div_Uc_x(i, j, k, grid, advection[n+1], (; u = us), vs)
    end
end

@kernel function _calculate_hydrostatic_free_surface_advection_y!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, tracers, halo) where FT
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local,  NTuple)

    N = @uniform @groupsize()[2]

    @synchronize

    u_shared = @localmem FT (2, N+2halo[2], 1) # This shared array will contain also tracers
    v_shared = @localmem FT (2, N+2halo[2], 1)

    us = OffsetSharedArray(u_shared, - i + 1, - j + halo[2], - k + 1)
    vs = OffsetSharedArray(v_shared, - i + 1, - j + halo[2], - k + 1)
    cs = OffsetSharedArray(c_shared, - i + 1, - j + halo[2], - k + 1)

    @inbounds u_shared[1, js+halo[2], 1] = velocities.u[i, j, k]
    @inbounds v_shared[1, js+halo[2], 1] = velocities.v[i, j, k]
    @inbounds u_shared[2, js+halo[2], 1] = velocities.u[i+1, j, k]
    @inbounds v_shared[2, js+halo[2], 1] = velocities.v[i+1, j, k]

    if is <= halo[1]
        @inbounds u_shared[1, js, 1] = velocities.u[i, j - halo[2], k]
        @inbounds v_shared[1, js, 1] = velocities.v[i, j - halo[2], k]
    end
    if is >= N - halo[1] + 1
        @inbounds u_shared[1, js+2halo[2], 1] = velocities.u[i, j + halo[2], k]
        @inbounds v_shared[1, js+2halo[2], 1] = velocities.v[i, j + halo[2], k]
    end

    @synchronize

    @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u_y(i, j, k, grid, advection.momentum, (u = us, v = vs))
    @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v_y(i, j, k, grid, advection.momentum, (u = us, v = vs))

    ntuple(Val(length(tracers))) do n
        Base.@_inline_meta
        @inbounds u_shared[1, js + halo[2], 1] = tracer[n][i, j, k]

        if is <= halo[2]
            @inbounds u_shared[1, js, 1] = tracer[n][i, j - halo[2], k]
        end
        if is >= N - halo[2] + 1
            @inbounds u_shared[1, js + 2halo[2], 1] = tracer[i, j + halo[2], k]
        end

        @synchronize

        @inbounds Gⁿ[n+3] = div_Uc_y(i, j, k, grid, advection[n+1], (; v = vs), us)
    end
end

@kernel function _calculate_hydrostatic_free_surface_advection_z!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, tracers, halo) where FT
    i,  j,  k  = @index(Global, NTuple)
    is, js, ks = @index(Local,  NTuple)

    N = @uniform @groupsize()[3]

    @synchronize

    w_shared = @localmem FT (2, 2, N+2halo[3])
    c_shared = @localmem FT (1, 1, N+2halo[3])

    ws = OffsetSharedArray(w_shared, - i + 1, - j + 1, - k + halo[3])
    cs = OffsetSharedArray(c_shared, - i + 1, - j + 1, - k + halo[3])

    @inbounds w_shared[1, 1, ks+halo[3]] = velocities.w[i-1, j-1, k]
    @inbounds w_shared[2, 1, ks+halo[3]] = velocities.w[i, j, k]
    @inbounds w_shared[1, 2, ks+halo[3]] = velocities.w[i, j, k]
    @inbounds c_shared[1, 1, ks+halo[3]] = velocities.u[i, j, k]

    if ks <= halo[3]
        @inbounds c_shared[1, 1, ks] = velocities.u[i, j, k - halo[3]]
    end
    if ks >= N - halo[3] + 1
        @inbounds c_shared[1, 1, ks+2halo[3]] = velocities.u[i, j, k + halo[3]]
    end

    @synchronize
    
    @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u_z(i, j, k, grid, advection.momentum, (u = cs, v = cs, w = ws))
    
    cs = velocities.v 

    @inbounds c_shared[1, 1, ks+halo[3]] = velocities.v[i, j, k]
    
    if ks <= halo[3]
        @inbounds cs[1, 1, ks] = velocities.v[i, j, k - halo[3]]
    end
    if ks >= N - halo[3] + 1
        @inbounds cs[1, 1, ks+2halo[3]] = velocities.v[i, j, k + halo[3]]
    end

    @synchronize

    @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v_z(i, j, k, grid, advection.momentum, (u = cs, v = cs, w = ws))

    ntuple(Val(length(tracers))) do n
        Base.@_inline_meta
        advect_tracer_z!(i, j, k, grid, Gⁿ[n+3], advection[n+1], (; w = ws), tracers[n], tracers[n], N, ks, halo[3])
    end
end

@inline advect_tracer_x!(i, j, k, grid, Gⁿ , ::Nothing, args...) = nothing
@inline advect_tracer_y!(i, j, k, grid, Gⁿ , ::Nothing, args...) = nothing
@inline advect_tracer_z!(i, j, k, grid, Gⁿ , ::Nothing, args...) = nothing

@inline function advect_tracer_x!(i, j, k, grid, Gⁿ , advection, velocities, cs, tracer, N, is, halo)


    @inbounds Gⁿ[i, j, k] -= div_Uc_y(i, j, k, grid, advection, velocities, cs)

    return nothing
end

@inline function advect_tracer_y!(i, j, k, grid, Gⁿ , advection, velocities, cs, tracer, N, js, halo)

    cs = tracer
    cs[1, js + halo, 1] = tracer[i, j, k]
    if js <= halo
        @inbounds cs[1, js, 1] = tracer[i, j - halo, k]
    end
    if js >= N - halo + 1
        @inbounds cs[1, js + 2halo, 1] = tracer[i, j + halo, k]
    end
    
    @inbounds Gⁿ[i, j, k] -= div_Uc_y(i, j, k, grid, advection, velocities, cs)

    return nothing
end

@inline function advect_tracer_z!(i, j, k, grid, Gⁿ , advection, velocities, cs, tracer, N, ks, halo)
    
    cs = tracer
    cs[1, 1, ks + halo] = tracer[i, j, k]
    if ks <= halo
        @inbounds cs[1, 1, ks] = tracer[i, j, k - halo]
    end
    if ks >= N - halo + 1
        @inbounds cs[1, 1, ks + 2halo] = tracer[i, j, k + halo]
    end
    
    @inbounds Gⁿ[i, j, k] -= div_Uc_z(i, j, k, grid, advection, velocities, cs)

    return nothing
end

# @kernel function _calculate_hydrostatic_free_surface_advection!(Gⁿ, grid::AbstractGrid{FT}, advection, velocities, tracers, halo) where FT
#     i,  j,  k  = @index(Global, NTuple)
#     is, js, ks = @index(Local,  NTuple)

#     N = @uniform @groupsize()[1]
#     M = @uniform @groupsize()[2]
#     O = @uniform @groupsize()[3]

#     @synchronize

#     us = OffsetSharedArray(@localmem FT (N+2halo[1], M+2halo[2], O+2halo[3]), 1, 1, 1)
#     vs = OffsetSharedArray(@localmem FT (N+2halo[1], M+2halo[2], O+2halo[3]), 1, 1, 1)
#     ws = OffsetSharedArray(@localmem FT (N+2halo[1], M+2halo[2], O+2halo[3]), 1, 1, 1)
#     cs = OffsetSharedArray(@localmem FT (N+2halo[1], M+2halo[2], O+2halo[3]), 1, 1, 1)

#     @inbounds us[is+halo[1], js+halo[2], ks+halo[3]] = velocities.u[i, j, k]
#     @inbounds vs[is+halo[1], js+halo[2], ks+halo[3]] = velocities.v[i, j, k]
#     @inbounds ws[is+halo[1], js+halo[2], ks+halo[3]] = velocities.w[i, j, k]

#     if is <= halo[1]
#         @inbounds us[is, js+halo[2], ks+halo[3]] = velocities.u[i - halo[1], j, k]
#         @inbounds vs[is, js+halo[2], ks+halo[3]] = velocities.v[i - halo[1], j, k]
#         @inbounds ws[is, js+halo[2], ks+halo[3]] = velocities.w[i - halo[1], j, k]
#     end
#     if is >= N - halo[1] + 1
#         @inbounds us[is+2halo[1], js+halo[2], ks+halo[3]] = velocities.u[i + halo[1], j, k]
#         @inbounds vs[is+2halo[1], js+halo[2], ks+halo[3]] = velocities.v[i + halo[1], j, k]
#         @inbounds ws[is+2halo[1], js+halo[2], ks+halo[3]] = velocities.w[i + halo[1], j, k]
#         # Fill the angles because of staggering!
#         if js <= halo[2]
#             @inbounds us[is+2halo[1], js, ks+halo[3]] = velocities.u[i + halo[1], j - halo[2], k]
#         end
#         if js >= M - halo[2] + 1
#             @inbounds us[is+2halo[1], js+2halo[2], ks+halo[3]] = velocities.u[i + halo[1], j + halo[2], k]
#         end
#         if ks <= halo[3]
#             @inbounds us[is+2halo[1], js+halo[2], ks] = velocities.u[i + halo[1], j, k - halo[3]]
#         end
#         if ks >= O - halo[3] + 1    
#             @inbounds us[is+2halo[1], js+halo[2], ks+2halo[3]] = velocities.u[i + halo[1], j, k + halo[3]]
#         end
#     end

#     if js <= halo[2]
#         @inbounds us[is+halo[1],js, ks+halo[3]] = velocities.u[i, j - halo[2], k]
#         @inbounds vs[is+halo[1],js, ks+halo[3]] = velocities.v[i, j - halo[2], k]
#         @inbounds ws[is+halo[1],js, ks+halo[3]] = velocities.w[i, j - halo[2], k]
#     end
#     if js >= M - halo[2] + 1
#         @inbounds us[is+halo[1], js+2halo[2], k+halo[3]] = velocities.u[i, j + halo[2], k]
#         @inbounds vs[is+halo[1], js+2halo[2], k+halo[3]] = velocities.v[i, j + halo[2], k]
#         @inbounds ws[is+halo[1], js+2halo[2], k+halo[3]] = velocities.w[i, j + halo[2], k]
#         # Fill the angles because of staggering!
#         if is <= halo[1]
#             @inbounds vs[is, js+2halo[2], ks+halo[3]] = velocities.v[i - halo[1], j + halo[2], k]
#         end
#         if is >= N - halo[1] + 1
#             @inbounds vs[is+2halo[1], js+2halo[2], k+halo[3]] = velocities.v[i + halo[1], j + halo[2], k]
#         end
#         if ks <= halo[3]
#             @inbounds vs[is+halo[1], js+2halo[2], ks] = velocities.v[i, j - halo[2], k - halo[3]]
#         end
#         if ks >= O - halo[3] + 1
#             @inbounds vs[is+halo[1], js+2halo[2], ks+2halo[3]] = velocities.v[i, j + halo[2], k + halo[3]]
#         end
#     end
    
#     if ks <= halo[3]
#         @inbounds us[is+halo[1], js+halo[2], ks] = velocities.u[i, j, k - halo[3]]
#         @inbounds vs[is+halo[1], js+halo[2], ks] = velocities.v[i, j, k - halo[3]]
#         @inbounds ws[is+halo[1], js+halo[2], ks] = velocities.w[i, j, k - halo[3]]
#     end
#     if ks >= O - halo[3] + 1
#         @inbounds us[is+halo[1], js+halo[2], ks+2halo[3]] = velocities.u[i, j, k + halo[3]]
#         @inbounds vs[is+halo[1], js+halo[2], ks+2halo[3]] = velocities.v[i, j, k + halo[3]]
#         @inbounds ws[is+halo[1], js+halo[2], ks+2halo[3]] = velocities.w[i, j, k + halo[3]]
#         # Fill the angles because of staggering!
#         if is <= halo[1]
#             @inbounds ws[is, js+halo[2], ks+2halo[3]] = velocities.w[i - halo[1], j, k + halo[3]]
#         end
#         if is >= N - halo[1] + 1
#             @inbounds ws[is+2halo[1], js+halo[2], ks+2halo[3]] = velocities.w[i + halo[1], j, k + halo[3]]
#         end
#         if js <= halo[2]
#             @inbounds ws[is+halo[1], js, ks+2halo[3]] = velocities.w[i, j - halo[2], k + halo[3]]
#         end
#         if js >= M - halo[2] + 1
#             @inbounds ws[is+halo[1], js+2halo[2], ks+2halo[3]] = velocities.w[i, j + halo[2], k + halo[3]]
#         end
#     end

#     @synchronize

#     @inbounds Gⁿ.u[i, j, k] -= U_dot_∇u(i, j, k, grid, advection.momentum, (u = us, v = vs, w = ws), is+halo[1], js+halo[2], ks+halo[3])
#     @inbounds Gⁿ.v[i, j, k] -= U_dot_∇v(i, j, k, grid, advection.momentum, (u = us, v = vs, w = ws), is+halo[1], js+halo[2], ks+halo[3])

#     ntuple(Val(length(tracers))) do n
#         Base.@_inline_meta
#         tracer = tracers[n]
#         @inbounds cs[is+halo[1], js+halo[2], ks+halo[3]] = tracer[i, j, k]
    
#         # No corners needed for the tracer
#         if is <= halo[1]
#             @inbounds cs[is, js+halo[2], ks+halo[3]] = tracer[i - halo[1], j, k]
#         end
#         if is >= N - halo[1] + 1
#             @inbounds cs[is+2halo[1], js+halo[2], ks+halo[3]] = tracer[i + halo[1], j, k]
#         end
    
#         if js <= halo[2]
#             @inbounds cs[is+halo[1], js, ks+halo[3]] = tracer[i, j - halo[2], k]
#         end
#         if js >= M - halo[2] + 1
#             @inbounds cs[is+halo[1], js+2halo[2], ks+halo[3]] = tracer[i, j + halo[2], k]
#         end
        
#         if ks <= halo[3]
#             @inbounds cs[is+halo[1], js+halo[2], ks] = tracer[i, j, k - halo[3]]
#         end
#         if ks >= O - halo[3] + 1
#             @inbounds cs[is+halo[1], j+halo[2], k+2halo[3]] = tracer[i, j, k + halo[3]]
#         end
    
#         @synchronize

#         @inbounds Gⁿ[n+3][i, j, k] -= div_Uc(i, j, k, grid, advection[n+1], (u = us, v = vs, w = ws), cs, is+halo[1], js+halo[2], ks+halo[3])
#     end
# end
