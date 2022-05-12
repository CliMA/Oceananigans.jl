using Oceananigans.BoundaryConditions: CBC
using Oceananigans.Architectures: arch_array
using Oceananigans.Grids: halo_size

struct FieldBoundaryBuffers{W, E, S, N}
    west :: W
    east :: E
   south :: S
   north :: N
end

function FieldBoundaryBuffers(fields)

    grid = first(fields).grid
    boundary_conditions = first(fields).boundary_conditions

    Hx, Hy, Hz = halo_size(grid)
    arch = architecture(grid)
    
    Nx = minimum([size(parent(field.data), 1) for field in fields])
    Ny = minimum([size(parent(field.data), 2) for field in fields])
    Nz = minimum([size(parent(field.data), 3) for field in fields])

    N = (Nx, Ny, Nz)
    west  = create_buffer_x(arch, N, Hx, length(fields), boundary_conditions.west)
    east  = create_buffer_x(arch, N, Hx, length(fields), boundary_conditions.east)
    south = create_buffer_y(arch, N, Hy, length(fields), boundary_conditions.south)
    north = create_buffer_y(arch, N, Hy, length(fields), boundary_conditions.north)

    return FieldBoundaryBuffers(west, east, south, north)
end

create_buffer_x(arch, N, H, Nvar, bc)    = nothing
create_buffer_y(arch, N, H, Nvar, bc)    = nothing

create_buffer_x(arch, N, H, Nvar, ::CBC) = (send = arch_array(arch, zeros(H, N[2], N[3], Nvar)), 
                                            recv = arch_array(arch, zeros(H, N[2], N[3], Nvar)))    
create_buffer_y(arch, N, H, Nvar, ::CBC) = (send = arch_array(arch, zeros(N[1], H, N[3], Nvar)), 
                                            recv = arch_array(arch, zeros(N[1], H, N[3], Nvar)))

Adapt.adapt_structure(to, buff::FieldBoundaryBuffers) =
    FieldBoundaryBuffers(Adapt.adapt(to, buff.west), 
                         Adapt.adapt(to, buff.east),    
                         Adapt.adapt(to, buff.north), 
                         Adapt.adapt(to, buff.south))