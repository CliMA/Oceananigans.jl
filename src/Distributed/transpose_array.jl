""" Utility to transpose fields and arrays from a X to a Y partitioned grid """
function transpose_x_to_y!(arry, arrx, ygrid::YPartitionedGrid, xgrid::XPartitionedGrid)

    px     = xgrid.partition
    region = Iterate(1:length(ygrid))

    _transpose_x_to_y!(arry, Reference(arrx), ygrid,  xgrid, px, region, xgrid.devices)

    return nothing
end

""" Utility to transpose fields and arrays from a Y to a X partitioned grid """
function transpose_y_to_x!(arrx, arry, xgrid::XPartitionedGrid, ygrid::YPartitionedGrid)

    py     = ygrid.partition
    region = Iterate(1:length(xgrid))

    _transpose_y_to_x!(arrx, Reference(arry), xgrid,  ygrid, py, region, ygrid.devices)

    return nothing
end

function transpose_x_to_y!(fieldy::DistributedField, fieldx::DistributedField) 

    ygrid  = fieldy.grid
    xgrid  = fieldx.grid
    px     = xgrid.partition
    
    region = Iterate(1:length(ygrid))

    _transpose_x_to_y!(fieldy, Reference(fieldx), ygrid,  xgrid, px, region)

    return nothing
end

function transpose_y_to_x!(fieldx::MultiRegionField, fieldy::MultiRegionField) 

    xgrid  = fieldx.grid
    ygrid  = fieldy.grid
    py     = ygrid.partition

    region = Iterate(1:length(ygrid))

    _transpose_y_to_x!(fieldx, Reference(fieldy), xgrid,  ygrid, py, region)

    return nothing
end

const XPartitionedField = Field{LX, LY, LZ, O, <:XPartitionedGrid} where {LX, LY, LZ, O}
const YPartitionedField = Field{LX, LY, LZ, O, <:YPartitionedGrid} where {LX, LY, LZ, O}

transpose_y_to_x!(fx::XPartitionedField, fy::YPartitionedField) = transpose_y_to_x!(fx, fy)
transpose_x_to_y!(fx::XPartitionedField, fy::YPartitionedField) = transpose_x_to_y!(fy, fx)

# - - - - - - - - - - - - - - #
# Transpose map goes this way
#
#   
#
#

function _transpose_x_to_y!(fieldy, full_fieldx, ygrid, xgrid, partition, rank, devices = nothing)
    
    region = isnothing(devices) ? Iterate(1:length(partition)) : MultiRegionObject(Tuple(1:length(partition)), devices)
    
    @apply_regionally _local_transpose_x_to_y!(fieldy, full_fieldx, ygrid, xgrid, region, rank)
end

function _local_transpose_x_to_y!(fieldy, fieldx, ygrid, xgrid, r, rank)
    xNx, _, xNz = size(xgrid)
    _, yNy, yNz = size(ygrid)

    xNz != yNz && throw(ArgumentError("Fields have to have same vertical dimensions!"))

    send_size = [1:xNx, (rank-1)*yNy+1:(rank*yNy), 1:xNz]
    recv_size = [(r-1)*xNx+1:(r*xNx), 1:xNx, 1:xNz]

    send_buff = viewof(fieldx, send_size...)
    
    switch_device!(getdevice(fieldy))

    recv_buff = viewof(fieldy, recv_size...)

    copyto!(recv_buff, send_buff)
end

function _transpose_y_to_x!(fieldx, full_fieldy, xgrid, ygrid, partition, rank, devices = nothing)
    
    region = isnothing(devices) ? Iterate(1:length(partition)) : MultiRegionObject(Tuple(1:length(partition)), devices)
    
    @apply_regionally _local_transpose_y_to_x!(fieldx, full_fieldy, xgrid, ygrid, region, rank)
end

function _local_transpose_y_to_x!(fieldx, fieldy, xgrid, ygrid, r, rank)
    xNx, _, xNz = size(xgrid)
    _, yNy, yNz = size(ygrid)

    xNz != yNz && throw(ArgumentError("Fields have to have same vertical dimensions!"))

    send_size = [(rank-1)*xNx+1:(rank*xNx), 1:yNy, 1:xNz]
    recv_size = [1:xNx, (r-1)*yNy+1:(r*yNy), 1:xNz]

    send_buff = viewof(fieldy, send_size...)
    
    switch_device!(getdevice(fieldx))

    recv_buff = viewof(fieldx, recv_size...)

    copyto!(recv_buff, send_buff)
end

@inline viewof(a::AbstractArray, idxs...) = view(a, idxs...)
@inline viewof(a::AbstractField, idxs...) = interior(a, idxs...)

####
#### Twin transposed grid
####

struct ParallelFields{FX, FY, FZ, ZY, YZ, YX, XY}
    xfield :: FX # X-direction is free, if `nothing` slab decomposition with `Rx == 1`
    yfield :: FY # Y-direction is free, if `nothing` slab decomposition with `Ry == 1`
    zfield :: FZ # Z-direction is free
    zybuffer :: ZY
    yzbuffer :: YZ
    yxbuffer :: YX
    xybuffer :: XY
end

const SlabXFields = ParallelFields{<:Any,     <:Nothing}
const SlabYFields = ParallelFields{<:Nothing, <:Any}

function ParallelFields(field_in)
    Rx, Ry, _ = architecture(field_in).ranks

    zgrid = field_in.grid
    ygrid = TwinGrid(zgrid; free_dim = :y)
    xgrid = TwinGrid(zgrid; free_dim = :x)

    loc = location(field_in)
    zfield = Field(Complex{FT}, loc, zgrid)
    yfield = Rx == 1 ? nothing : Field(Complex{FT}, loc, ygrid)
    xfield = Ry == 1 ? nothing : Field(Complex{FT}, loc, xgrid)
    
    Nx = size(xgrid)
    Ny = size(ygrid)
    Nz = size(zgrid)

    zybuffer = (send = arch_array(arch, zeros(Complex{FT}, Nz...)), 
                recv = arch_array(arch, zeros(Complex{FT}, Ny...)))
    xybuffer = (send = arch_array(arch, zeros(Complex{FT}, Nx...)), 
                recv = arch_array(arch, zeros(Complex{FT}, Ny...)))
    yzbuffer = (send = arch_array(arch, zeros(Complex{FT}, Ny...)), 
                recv = arch_array(arch, zeros(Complex{FT}, Nz...)))
    yxbuffer = (send = arch_array(arch, zeros(Complex{FT}, Ny...)), 
                recv = arch_array(arch, zeros(Complex{FT}, Nx...)))
    
    return ParallelFields(xfield, yfield, zfield, zybuffer, yzbuffer, yxbuffer, xybuffer)
end

# Fallbacks for slab decompositions
transpose_z_to_y!(::SlabXFields) = nothing
transpose_y_to_z!(::SlabXFields) = nothing
transpose_x_to_y!(::SlabYFields) = nothing
transpose_y_to_x!(::SlabYFields) = nothing

# Frees up the y direction `transpose_fields.yfield`
function transpose_z_to_y!(pf::ParallelFields)
    fill_transpose_buffer!(pf.yzbuffer, pf.fieldz)

    # Actually transpose!
    MPI.Alltoallv!(pf.zybuffer.send, pf.zybuffer.recv, size(zybuffer.send), size(pf.zybuffer.recv), MPI.COMM_WORLD)

    recv_transpose_buffer!(pf.fieldy, pf.yzbuffer)

    fill_halo_regions!(pf.fieldy; only_local_halos = true)
    return nothing
end

# Frees up the x direction `transpose_fields.xfield`
function transpose_y_to_x!(pf::ParallelFields)
    fill_transpose_buffer!(pf.xybuffer, pf.fieldy)

    # Actually transpose!
    _transpose_y_to_z!(pf.xybuffer)

    recv_transpose_buffer!(pf.fieldx, pf.xybuffer)

    fill_halo_regions!(pf.fieldx; only_local_halos = true)
    return nothing
end

# Frees up the y direction `transpose_fields.yfield`
function transpose_x_to_y!(pf::ParallelFields)
    fill_transpose_buffer!(pf.buffer_x_y.send, pf.fieldx)

    # Actually transpose!
    _transpose_y_to_z!(pf.xybuffer)

    recv_transpose_buffer!(pf.fieldy, pf.xybuffer)

    fill_halo_regions!(transpose_fields.fieldy; only_local_halos = true)
    return nothing
end

# Frees up the z direction `transpose_fields.zfield`
function transpose_y_to_z!(pf::ParallelFields)
    fill_transpose_buffer!(pf.yzbuffer, pf.fieldy)

    # Actually transpose!
    _transpose_y_to_z!(pf.yzbuffer)

    recv_transpose_buffer!(pf.fieldz, pf.yzbuffer)

    fill_halo_regions!(transpose_fields.fieldy; only_local_halos = true)
    return nothing
end

function TwinGrid(grid::DistributedGrid; free_dim = :y)

    arch = grid.architecture
    ri, rj, rk = arch.local_index

    R = arch.ranks

    nx, ny, nz = n = size(grid)
    Nx, Ny, Nz = map(sum, concatenate_local_sizes(n, arch))

    TX, TY, TZ = topology(grid)

    TX = reconstruct_global_topology(TX, Rx, ri, rj, rk, arch.communicator)
    TY = reconstruct_global_topology(TY, Ry, rj, ri, rk, arch.communicator)
    TZ = reconstruct_global_topology(TZ, Rz, rk, ri, rj, arch.communicator)

    x = cpu_face_constructor_x(grid)
    y = cpu_face_constructor_y(grid)
    z = cpu_face_constructor_z(grid)

    ## This will not work with 3D parallelizations!!
    xG = Rx == 1 ? x : assemble(x, nx, Rx, ri, rj, rk, arch.communicator)
    yG = Ry == 1 ? y : assemble(y, ny, Ry, rj, ri, rk, arch.communicator)
    zG = Rz == 1 ? z : assemble(z, nz, Rz, rk, ri, rj, arch.communicator)

    child_arch = child_architecture(arch)

    FT = eltype(grid)

    if free_dim == :y
        ranks = R[1], 1, R[2]

        nnx, nny, nnz = nx, Ny, nz ÷ ranks[3]

        if (nnz * ranks[3] < Nz) && (rj == ranks[3])
            nnz = Nz - nnz * (ranks[3] - 1)
        end
    elseif free_dim == :x
        ranks = 1, R[1], R[2]

        nnx, nny, nnz = Nx, Ny ÷ ranks[2], nz

        if (nny * ranks[2] < Ny) && (ri == ranks[2])
            nny = Ny - nny * (ranks[2] - 1)
        end
    elseif free_dims = :z
        @warn "That is the standard grid!!!"
        return grid
    end

    new_arch = DistributedArch(child_arch; 
                               ranks,
                               topology = (TX, TY, TZ))

    return construct_grid(grid, new_arch, FT; 
                          size = (nnx, nny, nnz), 
                        extent = (xG, yG, zG),
                      topology = (TX, TY, TZ))
end

construct_grid(::RectilinearGrid, arch, FT; size, extent, topology) = 
        RectilinearGrid(arch, FT; size, 
                        x = extent[1],
                        y = extent[2], 
                        z = extent[3],
                        topology)

construct_grid(::LatitudeLongitudeGrid, arch, FT; size, extent, topology) = 
        LatitudeLongitudeGrid(arch, FT; size, 
                        longitude = extent[1],
                         latitude = extent[2], 
                                z = extent[3],
                                topology)

