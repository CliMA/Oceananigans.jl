using Oceananigans.Architectures: arch_array

"""
    concatenate_local_size(n, arch::DistributedArch) 

returns a 3-Tuple containing a vector of `size(grid, idx)` for each rank in 
all 3 directions
"""
concatenate_local_size(n, arch::DistributedArch) = (concatenate_local_size(n, arch, 1),
                                                    concatenate_local_size(n, arch, 2),
                                                    concatenate_local_size(n, arch, 3))

function concatenate_local_size(n, arch::DistributedArch, idx)
    R = arch.ranks[idx]
    r = arch.local_index[idx]
    n = n[idx]
    l = zeros(Int, R)

    r1, r2 = arch.local_index[[1, 2, 3] .!= idx]
    
    if r1 == 1 && r2 == 1
        l[r] = n
    end

    MPI.Allreduce!(l, +, arch.communicator)
    
    return l
end

function concatenate_local_size(n, R, r) 
    l = zeros(Int, R)
    l[r] = n
    MPI.Allreduce!(l, +, MPI.COMM_WORLD)

    return l
end

# Partitioning (localization of global objects) and assembly (global assembly of local objects)
# Used for grid constructors (cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z)
# which means that we need to repeat the value at the right boundary

# Have to fix this! This won't work for face constructors??
function partition(c::AbstractVector, n, R, r)
    nl = concatenate_local_size(n, R, r)
    return c[1 + sum(nl[1:r-1]) : 1 + sum(nl[1:r])]
end

function partition(c::Tuple, n, R, r)
    nl = concatenate_local_size(n, R, r)
    N  = sum(nl)

    Δl = (c[2] - c[1]) / N  

    l = Tuple{Float64, Float64}[(c[1], c[1] + Δl * nl[1])]
    for i in 2:R
        lp = l[i-1][2]
        push!(l, (lp, lp + Δl * nl[i]))
    end

    return l[r]
end

"""
    assemble(c::AbstractVector, n, R, r, r1, r2, comm) 

Builds a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `Nc`, number of ranks `Nr`, rank `r`,
and `arch`itecture. Since we use a global reduction, only ranks at positions
1 in the other two directions `r1 == 1` and `r2 == 1` fill the 1D array.
"""
function assemble(c_local::AbstractVector, n, R, r, r1, r2, comm) 
    nl = concatenate_local_size(n, R, r)

    c_global = zeros(eltype(c_local), sum(nl)+1)

    if r1 == 1 && r2 == 1
        c_global[1 + sum(nl[1:r-1]) : sum(nl[1:r])] .= c_local[1:end-1]
        r == Nr && (c_global[end] = c_local[end])
    end

    MPI.Allreduce!(c_global, +, comm)

    return c_global
end

# Simple case, just take the first and the last core
function assemble(c::Tuple, n, R, r, r1, r2, comm) 
    c_global = zeros(Float64, 2)

    if r == 1 && r1 == 1 && r2 == 1
        c_global[1] = c[1]
    elseif r == R && r1 == 1 && r2 == 1
        c_global[2] = c[2]
    end

    MPI.Allreduce!(c_global, +, comm)

    return tuple(c_global...)
end 

# TODO: partition_global_array and construct_global_array
# do not currently work for 3D parallelizations
# (They are not used anywhere in the code at the moment exept for immersed boundaries)
"""
    partition_global_array(arch, c_global, (nx, ny, nz))

Partition a global array in local arrays of size `(nx, ny)` if 2D or `(nx, ny, nz)` is 3D.
Usefull for boundary arrays, forcings and initial conditions.
"""
partition_global_array(arch, c_global::Function, n) = c_global 

# Here we just assume we cannot partition in z (we should remove support for that!!)
function partition_global_array(arch, c_global::AbstractArray, n) 
    c_global = arch_array(CPU(), c_global)

    ri, rj, rk = r = arch.local_index

    dims = length(size(c_global))
    nx, ny, nz = concatenate_local_size(n, arch)

    nz = nz[1]

    if dims == 2 
        c_local = zeros(eltype(c_global), nx[ri], ny[rj])

        c_local .= c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                            1 + sum(ny[1:rj-1]) : sum(ny[1:rj])]
    else
        c_local = zeros(eltype(c_global), nx[ri], ny[rj], nz)

        c_local .= c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                            1 + sum(ny[1:rj-1]) : sum(ny[1:rj]), 
                            1:nz]
    end
    return arch_array(child_architecture(arch), c_local)
end

"""
    construct_global_array(arch, c_local, (nx, ny, nz))

Construct global array from local arrays (2D of size `(nx, ny)` or 3D of size (`nx, ny, nz`)).
Usefull for boundary arrays, forcings and initial conditions.
"""
construct_global_array(arch, c_local::Function, N) = c_local

# TODO: This does not work for 3D parallelizations!!!
function construct_global_array(arch, c_local::AbstractArray, n) 
    c_local = arch_array(CPU(), c_local)

    ri, rj, rk = arch.local_index

    dims = length(size(c_local))

    nx, ny, nz = concatenate_local_size(n, arch)

    Nx = sum(nx)
    Ny = sum(ny)
    Nz = nz[1]

    if dims == 2 
        c_global = zeros(eltype(c_local), Nx, Ny)
    
        c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                 1 + sum(ny[1:rj-1]) : sum(ny[1:rj])] .= c_local[1:nx[ri], 1:ny[rj]]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
    else
        c_global = zeros(eltype(c_local), Nx, Ny, Nz)

        c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                 1 + sum(ny[1:rj-1]) : sum(ny[1:rj]),
                 1:Nz] .= c_local[1:nx[ri], 1:ny[rj], 1:Nz]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
    end
    return arch_array(child_architecture(arch), c_global)
end
