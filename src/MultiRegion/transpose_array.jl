## Utility to transpose fields from a X to a Y partition and viceversa

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
const YPartitionedField = Field{LX, LY, LZ, O, <:XPartitionedGrid} where {LX, LY, LZ, O}

transpose_y_to_x!(fx::XPartitionedField, fy::YPartitionedField) = transpose_y_to_x!(fx, fy)
transpose_x_to_y!(fx::XPartitionedField, fy::YPartitionedField) = transpose_x_to_y!(fy, fx)

# - - - - - - - - - - - - - - #
# Transpose map goes this way
#
#   
#
#

function _transpose_x_to_y!(fieldy, full_fieldx, ygrid, xgrid, partition, rank)
    region = Iterate(1:length(partition))
    @apply_regionally _local_transpose_x_to_y!(fieldy, full_fieldx, ygrid, xgrid, region, rank)
end

function _local_transpose_x_to_y!(fieldy, fieldx, ygrid, xgrid, r, rank)
    xNx, _, xNz = size(xgrid)
    _, yNy, yNz = size(ygrid)

    xNz != yNz && throw(ArgumentError("Fields have to have same vertical dimensions!"))

    send_size = [1:xNx, (rank-1)*yNy+1:(rank*yNy), 1:xNz]
    recv_size = [(r-1)*xNx+1:(r*xNx), 1:xNx, 1:xNz]

    send_buff  = arch_array(architecture(ygrid), zeros(length.(send_size)...))
    send_buff .= interior(fieldx, send_size...)
    
    switch_device!(fieldy.data)

    recv_buff  = arch_array(architecture(ygrid), zeros(length.(recv_size)...))

    device_copy_to!(recv_buff, send_buff)
    
    interior(fieldy, recv_size...) .= recv_buff
end

function _transpose_y_to_x!(fieldx, full_fieldy, xgrid, ygrid, partition, rank)
    region = Iterate(1:length(partition))
    @apply_regionally _local_transpose_y_to_x!(fieldx, full_fieldy, xgrid, ygrid, region, rank)
end

function _local_transpose_y_to_x!(fieldx, fieldy, xgrid, ygrid, r, rank)
    xNx, _, xNz = size(xgrid)
    _, yNy, yNz = size(ygrid)

    xNz != yNz && throw(ArgumentError("Fields have to have same vertical dimensions!"))

    send_size = [(rank-1)*xNx+1:(rank*xNx), 1:yNy, 1:xNz]
    recv_size = [1:xNx, (r-1)*yNy+1:(r*yNy), 1:xNz]

    send_buff  = arch_array(architecture(ygrid), zeros(length.(send_size)...))
    send_buff .= interior(fieldy, send_size...)
    
    switch_device!(fieldx.data)

    recv_buff  = arch_array(architecture(ygrid), zeros(length.(recv_size)...))

    device_copy_to!(recv_buff, send_buff)
    
    interior(fieldx, recv_size...) .= recv_buff
end

####
#### Twin transposed grid
####

function transposed_grid(grid::MultiRegionGrid) 
    global_grid = reconstruct_global_grid(grid)

    p       = grid.partition
    devices = grid.devices
    
    if p isa XPartition
        return MultiRegionGrid(global_grid; partition = YPartition(length(p)), devices)
    else
        return MultiRegionGrid(global_grid; partition = XPartition(length(p)), devices)
    end
end
