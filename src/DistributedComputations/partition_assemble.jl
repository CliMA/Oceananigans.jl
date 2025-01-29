using Oceananigans.Fields: Field

import Oceananigans.Architectures: on_architecture

all_reduce(op, val, arch::Distributed) = MPI.Allreduce(val, op, arch.communicator)
all_reduce(op, val, arch) = val

# MPI Barrier
barrier!(arch) = nothing
barrier!(arch::Distributed) = MPI.Barrier(arch.communicator)

"""
    concatenate_local_sizes(local_size, arch::Distributed) 

Return a 3-Tuple containing a vector of `size(grid, dim)` for each rank in 
all 3 directions.
"""
concatenate_local_sizes(local_size, arch::Distributed) = 
    Tuple(concatenate_local_sizes(local_size, arch, d) for d in 1:length(local_size))

concatenate_local_sizes(sz, arch, dim) = concatenate_local_sizes(sz[dim], arch, dim)

function concatenate_local_sizes(n::Number, arch::Distributed, dim)
    R = arch.ranks[dim]
    r = arch.local_index[dim]
    N = zeros(Int, R)

    r1, r2 = arch.local_index[[1, 2, 3] .!= dim]
    
    if r1 == 1 && r2 == 1
        N[r] = n
    end

    MPI.Allreduce!(N, +, arch.communicator)
    
    return N
end

"""
    partition_coordinate(coordinate, n, arch, dim)

Return the local component of the global `coordinate`, which has
local length `n` and is distributed on `arch`itecture
in the x-, y-, or z- `dim`ension.
"""
function partition_coordinate(c::AbstractVector, n, arch, dim)
    nl = concatenate_local_sizes(n, arch, dim)
    r  = arch.local_index[dim]

    start_idx = sum(nl[1:r-1]) + 1 # sum of all previous rank's dimension + 1
    end_idx   = if r == ranks(arch)[dim]
        length(c)
    else
        sum(nl[1:r]) + 1 
    end

    return c[start_idx:end_idx]
end

function partition_coordinate(c::Tuple, n, arch, dim)
    nl = concatenate_local_sizes(n, arch, dim)
    N  = sum(nl)
    R  = arch.ranks[dim]
    Δl = (c[2] - c[1]) / N  

    l = Tuple{Float64, Float64}[(c[1], c[1] + Δl * nl[1])]
    for i in 2:R
        lp = l[i-1][2]
        push!(l, (lp, lp + Δl * nl[i]))
    end

    return l[arch.local_index[dim]]
end

"""
    assemble_coordinate(c::AbstractVector, n, R, r, r1, r2, comm) 

Builds a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `Nc`, number of ranks `Nr`, rank `r`,
and `arch`itecture. Since we use a global reduction, only ranks at positions
1 in the other two directions `r1 == 1` and `r2 == 1` fill the 1D array.
"""
function assemble_coordinate(c_local::AbstractVector, n, arch, dim) 
    nl = concatenate_local_sizes(n, arch, dim)
    R  = arch.ranks[dim]
    r  = arch.local_index[dim]
    r2 = [arch.local_index[i] for i in filter(x -> x != dim, (1, 2, 3))]

    c_global = zeros(eltype(c_local), sum(nl)+1)

    if r2[1] == 1 && r2[2] == 1
        c_global[1 + sum(nl[1:r-1]) : sum(nl[1:r])] .= c_local[1:end-1]
        r == R && (c_global[end] = c_local[end])
    end

    MPI.Allreduce!(c_global, +, arch.communicator)

    return c_global
end

# Simple case, just take the first and the last core
function assemble_coordinate(c_local::Tuple, n, arch, dim) 
    c_global = zeros(Float64, 2)
    
    rank = arch.local_index
    R    = arch.ranks[dim]
    r    = rank[dim]
    r2   = [rank[i] for i in filter(x -> x != dim, (1, 2, 3))]

    if rank[1] == 1 && rank[2] == 1 && rank[3] == 1
        c_global[1] = c_local[1]
    elseif r == R && r2[1] == 1 && r2[1] == 1
        c_global[2] = c_local[2]
    end

    MPI.Allreduce!(c_global, +, arch.communicator)

    return tuple(c_global...)
end 

# TODO: make partition and construct_global_array work for 3D distribution.

"""
    partition(A, b)

Partition the globally-sized `A` into local arrays with the same size as `b`.
"""
partition(A, b::Field) = partition(A, architecture(b), size(b))
partition(F::Field, b::Field) = partition(interior(F), b)
partition(f::Function, arch, n) = f
partition(A::AbstractArray, arch::AbstractSerialArchitecture, local_size) = A

"""
    partition(A, arch, local_size)

Partition the globally-sized `A` into local arrays with `local_size` on `arch`itecture.
"""
function partition(A::AbstractArray, arch::Distributed, local_size) 
    A = on_architecture(CPU(), A)

    ri, rj, rk = arch.local_index
    dims = length(size(A))

    # Vectors with the local size for every rank
    nxs, nys, nzs = concatenate_local_sizes(local_size, arch)

    # The local size
    nx = nxs[ri]
    ny = nys[rj]
    nz = nzs[1]
    # @assert (nx, ny, nz) == local_size

    up_to = nxs[1:ri-1]
    including = nxs[1:ri]
    i₁ = sum(up_to) + 1
    i₂ = sum(including)

    up_to = nys[1:rj-1]
    including = nys[1:rj]
    j₁ = sum(up_to) + 1
    j₂ = sum(including)

    ii = UnitRange(i₁, i₂)
    jj = UnitRange(j₁, j₂)
    kk = 1:nz # no partitioning in z

    # TODO: undo this toxic assumption that all 2D arrays span x, y.
    if dims == 2 
        a = zeros(eltype(A), nx, ny)
        a .= A[ii, jj]
    else
        a = zeros(eltype(A), nx, ny, nz)
        a .= A[ii, jj, 1:nz]
    end

    return on_architecture(child_architecture(arch), a)
end

"""
    construct_global_array(arch, c_local, (nx, ny, nz))

Construct global array from local arrays (2D of size `(nx, ny)` or 3D of size (`nx, ny, nz`)).
Usefull for boundary arrays, forcings and initial conditions.
"""
construct_global_array(arch, c_local::AbstractArray, n) = c_local
construct_global_array(arch, c_local::Function, N)      = c_local

# TODO: This does not work for 3D parallelizations
function construct_global_array(arch::Distributed, c_local::AbstractArray, n) 
    c_local = on_architecture(CPU(), c_local)

    ri, rj, rk = arch.local_index

    dims = length(size(c_local))

    nx, ny, nz = concatenate_local_sizes(n, arch)

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

    return on_architecture(child_architecture(arch), c_global)
end
