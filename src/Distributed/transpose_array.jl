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

# Frees up the y direction
function transpose_y_to_z!(fieldy, fieldz)
    archy = architecture(fieldy)
    archz = architecture(fieldz)

    ygrid = fieldy.grid
    zgrid = fieldz.grid
end

function TwinGrid(grid::DistributedGrid; free_dims = :y)

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

    if free_dims == :y
        ranks = R[1], 1, R[2]

        nnx, nny, nnz = nx, Ny, nz ÷ ranks[3]

        if (nnz * ranks[3] < Nz) && (rj == ranks[3])
            nnz = Nz - nnz * (ranks[3] - 1)
        end
    elseif free_dims == :x
        ranks = 1, R[1], R[3]

        nnx, nny, nnz = Nx, Ny ÷ ranks[2], nz

        if (nny * ranks[2] < Ny) && (ri == ranks[2])
            nny = Ny - nny * (ranks[2] - 1)
        end
    elseif free_dims = :z
        @warn "That is the standard grid!!!"
    end

    new_arch = DistributedArch(child_arch; 
                               ranks,
                               topology = (TX, TY, TZ))

    return construct_grid(grid, new_arch, FT; size = (nnx, nny, nnz), extent = (xG, yG, zG), topology = (TX, TY, TZ))
end

construct_grid(::RectilinearGrid, arch, FT; size, extent, topology) = RectilinearGrid(arch, FT; size, 
                                                                                      x = extent[1],
                                                                                      y = extent[2], 
                                                                                      z = extent[3],
                                                                                      topology)

construct_grid(::LatitudeLongitudeGrid, arch, FT; size, extent, topology) = LatitudeLongitudeGrid(arch, FT; size, 
                                                                                                  longitude = extent[1],
                                                                                                  latitude = extent[2], 
                                                                                                  z = extent[3],
                                                                                                  topology)

