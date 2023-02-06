using Oceananigans.Architectures: arch_array

# Partitioning (localization of global objects) and assembly (global assembly of local objects)
# Used for grid constructors (cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z)
# which means that we need to repeat the value at the right boundary

partition(c::Colon, loc, n, R, ri) = Colon()

""" Partition the vector. """
partition(c::AbstractVector, ::Face,   n, R, ri) = c[1 + (ri-1) * n : n * ri + 1]
partition(c::AbstractVector, ::Center, n, R, ri) = c[1 + (ri-1) * n : n * ri]

""" Partition the bounds of the interval c. """
partition((left, right)::Tuple, ::Face, n, R, ri) = (left + (ri - 1) * (right - left) / R,
                                                     left +       ri * (right - left) / R)

function partition(c::UnitRange, ::Face, n, R, ri)
    bounds = (first(c), last(r))
    local_bounds = partition(bounds, n, R, ri)
    return UnitRange(local_bounds[1], local_bounds[2])
end

"""
    assemble(c::AbstractVector, Nc, Nr, r, arch) 

Build a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `Nc`, number of ranks `Nr`, rank `r`,
and `arch`itecture. Since we use a global reduction, only ranks at positions
1 in the other two directions `r1 == 1` and `r2 == 1` fill the 1D array.
"""
function assemble(c_local::AbstractVector, Nc, Nr, r, r1, r2, comm) 
    c_global = zeros(eltype(c_local), Nc*Nr+1)

    if r1 == 1 && r2 == 1
        c_global[1 + (r-1) * Nc : Nc * r] .= c_local[1:end-1]
        r == Nr && (c_global[end] = c_local[end])
    end

    MPI.Allreduce!(c_global, +, comm)

    return c_global
end

assemble(c::Tuple, Nc, Nr, r, r1, r2, comm) = (c[2] - r * (c[2] - c[1]), c[2] - (r - Nr) * (c[2] - c[1]))

"""
    partition_global_array(arch, c_global, (Nx, Ny, Nz))

partition a global array (2D of size Nx, Ny or 3D of size Nx, Ny, Nz) in local arrays.
Usefull for boundary arrays, forcings and initial conditions
"""
partition_global_array(arch, c_global::Function, N) = c_global 

function partition_global_array(arch, c_global::AbstractArray, N) 
    c_global = arch_array(CPU(), c_global)
    Rx, Ry, Rz = R = arch.ranks
    ri, rj, rk = r = arch.local_index

    dims = length(size(c_global))

    if dims == 2 
        nx, ny = n = Int.(N[1:2] ./ R[1:2])

        c_local = zeros(eltype(c_global), nx, ny)

        c_local .= c_global[1 + (ri-1) * nx : nx * ri, 
                            1 + (rj-1) * ny : ny * rj]
    
        return arch_array(child_architecture(arch), c_local)
    else
        nx, ny, nz = n = Int.(N ./ R)

        c_local = zeros(eltype(c_global), nx, ny, nz)

        c_local .= c_global[1 + (ri-1) * nx : nx * ri, 
                            1 + (rj-1) * ny : ny * rj, 
                            1 + (rk-1) * nz : nz * rk]

        return arch_array(child_architecture(arch), c_local)
    end
end

"""
    reconstruct_global_array(arch, c_global, (nx, ny, nz))

reconstruct local arrays (2D of size nx, ny or 3D of size nx, ny, nz) in local arrays.
Usefull for boundary arrays, forcings and initial conditions
"""
reconstruct_global_array(arch, c_local::Function, N) = c_local

# TODO: This does not work for 2D parallelizations!!!
function reconstruct_global_array(arch, c_local::AbstractArray, n) 
    c_local = arch_array(CPU(), c_local)
    Rx, Ry, Rz = R = arch.ranks
    ri, rj, rk = r = arch.local_index

    dims = length(size(c_local))

    if dims == 2 
        nx, ny = n[1:2]
        Nx, Ny = N = Int.(n[1:2] .* R[1:2])
    
        c_global = zeros(eltype(c_local), Nx, Ny)
    
        c_global[1 + (ri-1) * nx : nx * ri, 
                 1 + (rj-1) * ny : ny * rj] .= c_local[1:nx, 1:ny]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
        
        return arch_array(child_architecture(arch), c_global)
    else
        nx, ny, nz = n
        Nx, Ny, Nz = N = Int.(n .* R)

        c_global = zeros(eltype(c_local), Nx, Ny, Nz)

        c_global[1 + (ri-1) * nx : nx * ri, 
                 1 + (rj-1) * ny : ny * rj, 
                 1 + (rk-1) * nz : nz * rk] .= c_local[1:nx, 1:ny, 1:nz]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
        
        return arch_array(child_architecture(arch), c_global)
    end
end
