#####
##### cuMemcpy2D-based halo pack/unpack for slab-x partitioning
#####
##### Replaces GPU broadcast kernels with DMA engine copies, freeing
##### GPU compute units for overlapping computation.
#####

"""
    strided_memcpy_2d!(dst, dst_offset, dst_pitch, src, src_offset, src_pitch, width, height)

2D strided copy using cuMemcpy2DAsync. Copies `width` bytes per row,
`height` rows, with independent src/dst pitches (strides).
Uses the DMA copy engine, not GPU compute cores.
"""
function strided_memcpy_2d!(dst::CuArray{T}, dst_offset, dst_pitch,
                            src::CuArray{T}, src_offset, src_pitch,
                            width, height) where T

    params = Ref(CUDA.CUDA_MEMCPY2D(
        UInt64(0),                     # srcXInBytes
        UInt64(0),                     # srcY
        CUDA.CU_MEMORYTYPE_DEVICE,     # srcMemoryType
        C_NULL,                        # srcHost
        pointer(src) + src_offset * sizeof(T),  # srcDevice
        CUDA.CuArrayPtr{Nothing}(0),                  # srcArray
        UInt64(src_pitch * sizeof(T)), # srcPitch
        UInt64(0),                     # dstXInBytes
        UInt64(0),                     # dstY
        CUDA.CU_MEMORYTYPE_DEVICE,     # dstMemoryType
        C_NULL,                        # dstHost
        pointer(dst) + dst_offset * sizeof(T),  # dstDevice
        CUDA.CuArrayPtr{Nothing}(0),                  # dstArray
        UInt64(dst_pitch * sizeof(T)), # dstPitch
        UInt64(width * sizeof(T)),     # WidthInBytes
        UInt64(height),                # Height
    ))
    CUDA.cuMemcpy2DAsync_v2(params, CUDA.stream())
    return nothing
end

"""
    nccl_fill_send_buffers_2d!(c_parent, buff, grid, ::WestAndEast)

Pack west/east halo send buffers using cuMemcpy2D instead of broadcast kernels.
For slab-x: copies Hx-wide strips from the field parent array into contiguous buffers.
"""
function nccl_fill_send_buffers_2d!(c_parent::CuArray{T}, buff, grid,
                                     ::Oceananigans.BoundaryConditions.WestAndEast) where T
    Hx, Hy, Hz = Oceananigans.Grids.halo_size(grid)
    Nx, Ny, Nz = size(grid)
    full_Nx = Nx + 2Hx
    n_rows = (Ny + 2Hy) * (Nz + 2Hz)

    # West send: parent[Hx+1 : 2Hx, :, :] → 0-based offset = Hx
    strided_memcpy_2d!(buff.west.send, 0, Hx,
                       c_parent, Hx, full_Nx,
                       Hx, n_rows)

    # East send: parent[Nx+1 : Nx+Hx, :, :] → 0-based offset = Nx
    strided_memcpy_2d!(buff.east.send, 0, Hx,
                       c_parent, Nx, full_Nx,
                       Hx, n_rows)
    return nothing
end

"""
    nccl_recv_from_buffers_2d!(c_parent, buff, grid, ::WestAndEast)

Unpack west/east halo recv buffers using cuMemcpy2D.
"""
function nccl_recv_from_buffers_2d!(c_parent::CuArray{T}, buff, grid,
                                     ::Oceananigans.BoundaryConditions.WestAndEast) where T
    Hx, Hy, Hz = Oceananigans.Grids.halo_size(grid)
    Nx, Ny, Nz = size(grid)
    full_Nx = Nx + 2Hx
    n_rows = (Ny + 2Hy) * (Nz + 2Hz)

    # West recv: buffer → parent[1:Hx, :, :] → 0-based offset = 0
    strided_memcpy_2d!(c_parent, 0, full_Nx,
                       buff.west.recv, 0, Hx,
                       Hx, n_rows)

    # East recv: buffer → parent[Nx+Hx+1 : Nx+2Hx, :, :] → 0-based offset = Nx+Hx
    strided_memcpy_2d!(c_parent, Nx + Hx, full_Nx,
                       buff.east.recv, 0, Hx,
                       Hx, n_rows)
    return nothing
end
