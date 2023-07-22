using Oceananigans

# Boundaries of host region
struct East end
struct West end
struct North end
struct South end

struct Uvelocity end
struct Vvelocity end
struct North end 
struct South end 

# Rotations of adjacent regions with respect to host region.
# Transforming host indices into the adjacent region coordinate system
# _inverts_ this rotation.
struct ↺ end 
struct ↻ end 

# Shift host index into adjacent region coordinate system
@inline shift_index(i, j, Nx, Ny, ::East)  = i - Nx, j
@inline shift_index(i, j, Nx, Ny, ::West)  = i + Nx, j
@inline shift_index(i, j, Nx, Ny, ::North) =      i, j - Ny
@inline shift_index(i, j, Nx, Ny, ::South) =      i, j + Ny

# Rotate host region index into adjacent region coordinate system
@inline rotate_index(i, j, Nx, Ny, ::Nothing) = i, j
@inline rotate_index(i, j, Nx, Ny, ::↻) = Ny + 1 - j, i # Rotate host index _counterclockwise_
@inline rotate_index(i, j, Nx, Ny, ::↺) = j, Nx + 1 - i # Rotate host index _clockwise_

is_interior(i, j, k, grid) = i in 1:grid.Nx && j in 1:grid.Ny && k in 1:grid.Nz ? "hi" : "no"


""" 
Returns a scalar cell-centered value in an adjacent region
corresponding to host region indices `i, j, k` and host `grid`.
The adjacent region joins the host region at `host_boundary`
and is rotated by `adjacent_rotation` with respect to the host region.
"""
@inline function adjacent_scalar(i, j, k, grid, host_boundary, adjacent_rotation, adjacent_c)
    i′,  j′  =  shift_index(i,  j,  grid.Nx, grid.Ny, host_boundary)
    i′′, j′′ = rotate_index(i′, j′, grid.Nx, grid.Ny, adjacent_rotation)
    @show i′′, j′′
    return @inbounds adjacent_c[i′, j′, k]
end


# Vector adjacencies: easy case with no rotation.
@inline function adjacent_vector(i, j, k, grid, host_boundary, ::Nothing, ua, va)
    @show "(+u, +v)"
    return (adjacent_scalar(i, j, k, grid, host_boundary, nothing, ua),
            adjacent_scalar(i, j, k, grid, host_boundary, nothing, va))
end

# host          adjacent
# -------       ---u---
# |     |       |  ↓  |
# u→    |   ↻   v→    |  
# |  ↑  |       |     |
# ---v---       -------
#                ↓↓ increasing i_adjacent
#
@inline function adjacent_vector(i, j, k, grid, host_boundary, ::↻, ua, va)
    @show "(+v, -u)"
    return ( + adjacent_scalar(i, j,     k, grid, host_boundary, ↻(), va),
             - adjacent_scalar(i, j - 1, k, grid, host_boundary, ↻(), ua))
end

# host          adjacent
# -------       -------
# |     |       |     |
# u→    |   ↺   |    ←v   →→ decreasing j_adjacent
# |  ↑  |       |  ↑  |
# ---v---       ---u---
@inline function adjacent_vector(i, j, k, grid, host_boundary, ::↺, ua, va)
    @show "(-v, +u)"
    return ( - adjacent_scalar(i - 1, j, k, grid, host_boundary, ↺(), va),
             + adjacent_scalar(i,     j, k, grid, host_boundary, ↺(), ua))
end

Nx, Ny, Nz = 3, 3, 1
ca, ua, va = zeros(Nx, Ny), zeros(Nx, Ny), zeros(Nx, Ny)
grid = ConformalCubedSphereGrid(panel_size=(Nx, Ny, Nz), z=(-1, 0))


[adjacent_scalar(Nx+1, j, 1, grid, East(), nothing, ca) for j in 1:Ny];
[adjacent_scalar(i, Ny+1, 1, grid, North(), ↺(), ca) for i in 1:Nx];


[adjacent_vector(Nx+1, j, 1, grid, East(), nothing, ua, va) for j in 1:Ny];
[adjacent_vector(i, Ny+1, 1, grid, North(), ↺(), ua, va) for i in 1:Nx];