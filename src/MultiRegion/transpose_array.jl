## Utility to transpose fields from a X to a Y partition and viceversa

function transpose_x_to_y!(arry, arrx, ygrid::YPartitionedGrid, xgrid::XPartitionedGrid)

    px     = xgrid.partition
    region = Iterate(1:length(ygrid))

    @apply_regionally _transpose_x_to_y!(arry, Reference(arrx), ygrid,  xgrid, px, region, xgrid.devices)

    return nothing
end

function transpose_y_to_x!(arrx, arry, xgrid::XPartitionedGrid, ygrid::YPartitionedGrid)

    py     = ygrid.partition
    region = Iterate(1:length(xgrid))

    @apply_regionally _transpose_y_to_x!(arrx, Reference(arry), xgrid,  ygrid, py, region, ygrid.devices)

    return nothing
end

function transpose_x_to_y!(fieldy::MultiRegionField, fieldx::MultiRegionField) 

    ygrid  = fieldy.grid
    xgrid  = fieldx.grid
    px     = xgrid.partition
    
    region = Iterate(1:length(ygrid))

    @apply_regionally _transpose_x_to_y!(fieldy, Reference(fieldx), ygrid,  xgrid, px, region)

    return nothing
end

function transpose_y_to_x!(fieldx::MultiRegionField, fieldy::MultiRegionField) 

    xgrid  = fieldx.grid
    ygrid  = fieldy.grid
    py     = ygrid.partition

    region = Iterate(1:length(ygrid))

    @apply_regionally _transpose_y_to_x!(fieldx, Reference(fieldy), xgrid,  ygrid, py, region)

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

function transposed_grid(grid::MultiRegionGrid) 
    global_grid = reconstruct_global_grid(grid)

    p       = grid.partition
    devices = grid.devices
    
    if p isa XPartition
        return MultiRegionGrid(global_grid; partition = YPartition(length(p)), devices, validate = false)
    else
        return MultiRegionGrid(global_grid; partition = XPartition(length(p)), devices, validate = false)
    end
end
