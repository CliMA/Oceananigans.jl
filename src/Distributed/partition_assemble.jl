# Partitioning (localization of global objects) and assembly (global assembly of local objects)

partition(c::AbstractVector, Nc, Nr, r) = c[1 + (r-1) * Nc : 1 + Nc * r]
partition(c::Colon,          Nc, Nr, r) = Colon()
partition(c::Tuple,          Nc, Nr, r) = (c[1] + (r-1) * (c[2] - c[1]) / Nr,    c[1] + r * (c[2] - c[1]) / Nr)

function partition(c::UnitRange, Nc, Nr, r)
    g = (first(c), last(r))
    ℓ = partition(g, Nc, Nr, r)
    return UnitRange(ℓ[1], ℓ[2])
end

assemble(c::Tuple, Nc, Nr, r, arch) = (c[2] - r * (c[2] - c[1]), c[2] - (r - Nr) * (c[2] - c[1]))

"""
    assemble(c::AbstractVector, Nc, Nr, r, arch) 

Build a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `Nc`, number of ranks `Nr`, rank `r`,
and `arch`itecture.
"""
function assemble(c_local::AbstractVector, Nc, Nr, r, arch) 
    c_global = zeros(eltype(c_local), Nc*Nr+1)
    c_global[1 + (r-1) * Nc : Nc * r] .= c[1:end-1]
    r == Nr && (c_global[end] = c[end])

    MPI.Allreduce!(c_global, +, arch.communicator)

    return c_global
end

