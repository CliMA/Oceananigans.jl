using Oceananigans.Fields: Field

import Oceananigans.Architectures: on_architecture

all_reduce(op, val, arch::Distributed) = MPI.Allreduce(val, op, arch.communicator)
all_reduce(op, val, arch) = val

all_reduce!(op, val, arch::Distributed) = MPI.Allreduce!(val, op, arch.communicator)
all_reduce!(op, val, arch) = val

# MPI Barrier
barrier(arch) = nothing
barrier(arch::Distributed) = barrier(arch.communicator)

"""
$(TYPEDSIGNATURES)

Return a 3-Tuple of vectors, where each vector contains the value of
`local_size[d]` for every rank in dimension `d` of `arch`.
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
$(TYPEDSIGNATURES)

Return the local component of the global `coordinate`, which has
local length `n` and is distributed on `arch`itecture
in the x-, y-, or z- `dim`ension.
"""
function partition_coordinate(c::AbstractVector, n, arch, dim)
    range = local_ranges(arch)[dim]
    return c[first(range):last(range)+1]
end

function partition_coordinate(c::Tuple, n, arch, dim)
    sizes = line_sizes(arch, dim)
    Δl = (c[2] - c[1]) / sum(sizes)

    l = Tuple{Float64, Float64}[(c[1], c[1] + Δl * sizes[1])]
    for i in 2:length(sizes)
        lp = l[i-1][2]
        push!(l, (lp, lp + Δl * sizes[i]))
    end

    return l[arch.local_index[dim]]
end

# Sizes along `dim` of the ranks that share the other two indices of the local rank
function line_sizes(arch, dim)
    Rx, Ry, Rz = arch.ranks
    i, j, k = arch.local_index
    ranges = arch.connectivity.ranges
    dim == 1 && return [length(ranges[index2rank(r, j, k, Rx, Ry, Rz) + 1][1]) for r in 1:Rx]
    dim == 2 && return [length(ranges[index2rank(i, r, k, Rx, Ry, Rz) + 1][2]) for r in 1:Ry]
    return [length(ranges[index2rank(i, j, r, Rx, Ry, Rz) + 1][3]) for r in 1:Rz]
end

"""
$(TYPEDSIGNATURES)

Builds a linear global coordinate vector along dimension `dim` given a local coordinate
vector `c_local`, the local sizes `n`, and the architecture `arch`. Since we use a global
reduction, only ranks at position 1 in the other two directions fill the 1D array.
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
$(TYPEDSIGNATURES)

Partition the globally-sized `A` into local arrays with the same size as `b`.
"""
partition(A, b::Field) = partition(A, architecture(b), size(b))
partition(F::Field, b::Field) = partition(interior(F), b)
partition(f::Function, arch, n) = f
partition(A::AbstractArray, arch::AbstractSerialArchitecture, local_size) = A

"""
$(TYPEDSIGNATURES)

Partition the globally-sized `A` into local arrays with `local_size` on `arch`itecture.
"""
function partition(A::AbstractArray, arch::Distributed, local_size)
    A = on_architecture(CPU(), A)

    i, j, _ = first.(local_ranges(arch))
    nx, ny, nz = local_size

    # TODO: undo this toxic assumption that all 2D arrays span x, y.
    a = ndims(A) == 2 ? A[i:i+nx-1, j:j+ny-1] : A[i:i+nx-1, j:j+ny-1, 1:nz]

    return on_architecture(child_architecture(arch), a)
end

"""
$(TYPEDSIGNATURES)

Construct global array from local arrays (2D of size `(nx, ny)` or 3D of size (`nx, ny, nz`)).
Useful for boundary arrays, forcings, and initial conditions.
"""
construct_global_array(c_local::AbstractArray, arch, n) = c_local
construct_global_array(c_local::Function, arch,  N)     = c_local

# TODO: This does not work for 3D parallelizations
function construct_global_array(c_local::AbstractArray, arch::Distributed, n)
    c_local = on_architecture(CPU(), c_local)

    Nx, Ny, Nz = global_size(arch, n)
    i, j, _ = first.(local_ranges(arch))
    nx, ny, _ = n

    if ndims(c_local) == 2
        c_global = zeros(eltype(c_local), Nx, Ny)
        c_global[i:i+nx-1, j:j+ny-1] .= c_local[1:nx, 1:ny]
    else
        c_global = zeros(eltype(c_local), Nx, Ny, Nz)
        c_global[i:i+nx-1, j:j+ny-1, 1:Nz] .= c_local[1:nx, 1:ny, 1:Nz]
    end

    MPI.Allreduce!(c_global, +, arch.communicator)

    return on_architecture(child_architecture(arch), c_global)
end
