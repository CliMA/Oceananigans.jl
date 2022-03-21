using Oceananigans.BoundaryConditions: HBC
using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: halo_size

struct FieldBoundaryBuffers{W, E, S, N}
    west :: W
    east :: E
   south :: S
   north :: N
end

FieldBoundaryBuffers() = FieldBoundaryBuffers(nothing, nothing, nothing, nothing)

function FieldBoundaryBuffers(grid, data, boundary_conditions)

    Hx, Hy, Hz = halo_size(grid)

    west  = create_buffer_x(architecture(grid), data, Hx, boundary_conditions.west)
    east  = create_buffer_x(architecture(grid), data, Hx, boundary_conditions.east)
    south = create_buffer_y(architecture(grid), data, Hy, boundary_conditions.south)
    north = create_buffer_y(architecture(grid), data, Hy, boundary_conditions.north)

    return FieldBoundaryBuffers(west, east, south, north)
end

create_buffer_x(arch, data, H, bc)    = nothing
create_buffer_y(arch, data, H, bc)    = nothing
create_buffer_x(arch, data, H, ::HBC) = arch_array(arch, zeros(H, size(parent(data), 2), size(parent(data), 3)))
create_buffer_y(arch, data, H, ::HBC) = arch_array(arch, zeros(size(parent(data), 1), H, size(parent(data), 3)))

Adapt.adapt_structure(to, buff::FieldBoundaryBuffers) =
    FieldBoundaryBuffers(Adapt.adapt(to, buff.west), 
                         Adapt.adapt(to, buff.east),    
                         Adapt.adapt(to, buff.north), 
                         Adapt.adapt(to, buff.south))