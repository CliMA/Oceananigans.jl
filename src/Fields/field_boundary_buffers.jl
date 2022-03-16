using Oceananigans.BoundaryConditions: CBC
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

    west  = create_buffer(architecture(grid), data, Hx, boundary_conditions.west)
    east  = create_buffer(architecture(grid), data, Hx, boundary_conditions.east)
    south = create_buffer(architecture(grid), data, Hy, boundary_conditions.south)
    north = create_buffer(architecture(grid), data, Hy, boundary_conditions.north)

    return FieldBoundaryBuffers(west, east, south, north)
end

create_buffer(arch, data, H, bc)    = nothing
create_buffer(arch, data, H, ::CBC) = arch_array(arch, zeros(H, size(parent(data), 2), size(parent(data), 3)))

Adapt.adapt_structure(to, buff::FieldBoundaryBuffers) =
    FieldBoundaryBuffers(Adapt.adapt(to, buff.west), 
                         Adapt.adapt(to, buff.east),    
                         Adapt.adapt(to, buff.north), 
                         Adapt.adapt(to, buff.south))