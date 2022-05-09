# Partitioning (localization of global objects) and assembly (global assembly of local objects)

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

assemble(c::Tuple, Nc, Nr, r, arch) = (c[2] - r * (c[2] - c[1]), c[2] - (r - Nr) * (c[2] - c[1]))

"""
    assemble(c::AbstractVector, Nc, Nr, r, arch) 

Build a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `Nc`, number of ranks `Nr`, rank `r`,
and `arch`itecture.
"""
function assemble(c_local::AbstractVector, Nc, Nr, r, arch) 
    c_global = zeros(eltype(c_local), Nc * Nr + 1)
    c_global[1 + (r-1) * Nc : Nc * r] .= c_local[1:end-1]
    r == Nr && (c_global[end] = c_local[end])

    MPI.Allreduce!(c_global, +, arch.communicator)

    return c_global
end

