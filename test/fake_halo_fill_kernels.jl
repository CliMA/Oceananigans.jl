# Fake halo fill kernels for testing Reactant with raise=true
# These are independent of Oceananigans, using only KernelAbstractions and OffsetArrays

using KernelAbstractions
using OffsetArrays

#####
##### X-direction halo fill kernels
#####

# Bounded: left halo copies from first interior point
@kernel function _fake_x_left_halo_fill_bounded!(c, Nx)
    j, k = @index(Global, NTuple)
    @inbounds c[0, j, k] = c[1, j, k]
end

# Bounded: right halo copies from last interior point
@kernel function _fake_x_right_halo_fill_bounded!(c, Nx)
    j, k = @index(Global, NTuple)
    @inbounds c[Nx+1, j, k] = c[Nx, j, k]
end

# Periodic: left halo copies from right interior
@kernel function _fake_x_left_halo_fill_periodic!(c, Nx)
    j, k = @index(Global, NTuple)
    @inbounds c[0, j, k] = c[Nx, j, k]
end

# Periodic: right halo copies from left interior
@kernel function _fake_x_right_halo_fill_periodic!(c, Nx)
    j, k = @index(Global, NTuple)
    @inbounds c[Nx+1, j, k] = c[1, j, k]
end

#####
##### Y-direction halo fill kernels
#####

# Bounded: left halo copies from first interior point
@kernel function _fake_y_left_halo_fill_bounded!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, 0, k] = c[i, 1, k]
end

# Bounded: right halo copies from last interior point
@kernel function _fake_y_right_halo_fill_bounded!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, Ny+1, k] = c[i, Ny, k]
end

# Periodic: left halo copies from right interior
@kernel function _fake_y_left_halo_fill_periodic!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, 0, k] = c[i, Ny, k]
end

# Periodic: right halo copies from left interior
@kernel function _fake_y_right_halo_fill_periodic!(c, Ny)
    i, k = @index(Global, NTuple)
    @inbounds c[i, Ny+1, k] = c[i, 1, k]
end

#####
##### Z-direction halo fill kernels
#####

# Bounded: left halo copies from first interior point
@kernel function _fake_z_left_halo_fill_bounded!(c, Nz)
    i, j = @index(Global, NTuple)
    @inbounds c[i, j, 0] = c[i, j, 1]
end

# Bounded: right halo copies from last interior point
@kernel function _fake_z_right_halo_fill_bounded!(c, Nz)
    i, j = @index(Global, NTuple)
    @inbounds c[i, j, Nz+1] = c[i, j, Nz]
end

# Periodic: left halo copies from right interior
@kernel function _fake_z_left_halo_fill_periodic!(c, Nz)
    i, j = @index(Global, NTuple)
    @inbounds c[i, j, 0] = c[i, j, Nz]
end

# Periodic: right halo copies from left interior
@kernel function _fake_z_right_halo_fill_periodic!(c, Nz)
    i, j = @index(Global, NTuple)
    @inbounds c[i, j, Nz+1] = c[i, j, 1]
end

#####
##### Combined halo fill functions
#####

"""
    fake_fill_halo_x!(c, Nx, Ny, Nz, ::Val{:bounded}, backend)

Fill x-direction halos for bounded topology.
"""
function fake_fill_halo_x!(c, Nx, Ny, Nz, ::Val{:bounded}, backend)
    _fake_x_left_halo_fill_bounded!(backend, (Ny, Nz))(c, Nx, ndrange=(Ny, Nz))
    _fake_x_right_halo_fill_bounded!(backend, (Ny, Nz))(c, Nx, ndrange=(Ny, Nz))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    fake_fill_halo_x!(c, Nx, Ny, Nz, ::Val{:periodic}, backend)

Fill x-direction halos for periodic topology.
"""
function fake_fill_halo_x!(c, Nx, Ny, Nz, ::Val{:periodic}, backend)
    _fake_x_left_halo_fill_periodic!(backend, (Ny, Nz))(c, Nx, ndrange=(Ny, Nz))
    _fake_x_right_halo_fill_periodic!(backend, (Ny, Nz))(c, Nx, ndrange=(Ny, Nz))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    fake_fill_halo_y!(c, Nx, Ny, Nz, ::Val{:bounded}, backend)

Fill y-direction halos for bounded topology.
"""
function fake_fill_halo_y!(c, Nx, Ny, Nz, ::Val{:bounded}, backend)
    _fake_y_left_halo_fill_bounded!(backend, (Nx, Nz))(c, Ny, ndrange=(Nx, Nz))
    _fake_y_right_halo_fill_bounded!(backend, (Nx, Nz))(c, Ny, ndrange=(Nx, Nz))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    fake_fill_halo_y!(c, Nx, Ny, Nz, ::Val{:periodic}, backend)

Fill y-direction halos for periodic topology.
"""
function fake_fill_halo_y!(c, Nx, Ny, Nz, ::Val{:periodic}, backend)
    _fake_y_left_halo_fill_periodic!(backend, (Nx, Nz))(c, Ny, ndrange=(Nx, Nz))
    _fake_y_right_halo_fill_periodic!(backend, (Nx, Nz))(c, Ny, ndrange=(Nx, Nz))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    fake_fill_halo_z!(c, Nx, Ny, Nz, ::Val{:bounded}, backend)

Fill z-direction halos for bounded topology.
"""
function fake_fill_halo_z!(c, Nx, Ny, Nz, ::Val{:bounded}, backend)
    _fake_z_left_halo_fill_bounded!(backend, (Nx, Ny))(c, Nz, ndrange=(Nx, Ny))
    _fake_z_right_halo_fill_bounded!(backend, (Nx, Ny))(c, Nz, ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    fake_fill_halo_z!(c, Nx, Ny, Nz, ::Val{:periodic}, backend)

Fill z-direction halos for periodic topology.
"""
function fake_fill_halo_z!(c, Nx, Ny, Nz, ::Val{:periodic}, backend)
    _fake_z_left_halo_fill_periodic!(backend, (Nx, Ny))(c, Nz, ndrange=(Nx, Ny))
    _fake_z_right_halo_fill_periodic!(backend, (Nx, Ny))(c, Nz, ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    fake_fill_halo_regions!(c, Nx, Ny, Nz, topo_x, topo_y, topo_z, backend)

Fill all halo regions for an OffsetArray with the given topology.
`topo_x`, `topo_y`, `topo_z` should be either `:periodic` or `:bounded`.
"""
function fake_fill_halo_regions!(c, Nx, Ny, Nz, topo_x, topo_y, topo_z, backend)
    fake_fill_halo_x!(c, Nx, Ny, Nz, Val(topo_x), backend)
    fake_fill_halo_y!(c, Nx, Ny, Nz, Val(topo_y), backend)
    fake_fill_halo_z!(c, Nx, Ny, Nz, Val(topo_z), backend)
    return nothing
end

#####
##### Utilities for creating test arrays
#####

"""
    create_offset_array(Nx, Ny, Nz; halo=1)

Create an OffsetArray with halo regions (indices go from 1-halo to N+halo).
"""
function create_offset_array(Nx, Ny, Nz; halo=1)
    # Interior size + halos
    total_x = Nx + 2*halo
    total_y = Ny + 2*halo
    total_z = Nz + 2*halo
    
    # Create array and wrap with offset
    data = zeros(Float64, total_x, total_y, total_z)
    offset_data = OffsetArray(data, (1-halo):(Nx+halo), (1-halo):(Ny+halo), (1-halo):(Nz+halo))
    
    return offset_data
end

"""
    set_interior!(c, data, Nx, Ny, Nz)

Set the interior of an offset array to the given data.
"""
function set_interior!(c, data, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        c[i, j, k] = data[i, j, k]
    end
    return nothing
end

"""
    compare_arrays(name, a, b; rtol=1e-10, atol=1e-10)

Compare two arrays element-wise.
Works with OffsetArrays containing either regular Arrays or Reactant RArrays.
"""
function compare_arrays(name, a, b; rtol=1e-10, atol=1e-10)
    # Get underlying data, converting RArray to Array if needed
    pa = parent(a)
    pb = parent(b)
    
    # Convert Reactant RArray to regular Array if needed
    pa_arr = pa isa Array ? pa : Array(pa)
    pb_arr = pb isa Array ? pb : Array(pb)
    
    if size(pa_arr) != size(pb_arr)
        println("Size mismatch: $(size(pa_arr)) vs $(size(pb_arr))")
        return false
    end
    
    max_diff = 0.0
    for i in eachindex(pa_arr)
        diff = abs(pa_arr[i] - pb_arr[i])
        max_diff = max(max_diff, diff)
    end
    
    passed = max_diff < atol || max_diff < rtol * max(maximum(abs, pa_arr), maximum(abs, pb_arr))
    
    if !passed
        println("$name comparison failed: max_diff = $max_diff")
    end
    
    return passed
end

