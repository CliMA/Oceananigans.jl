using Oceananigans.Architectures: arch_array

# Partitioning (localization of global objects) and assembly (global assembly of local objects)
# Used for grid constructors (cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z)
# which means that we need to repeat the value at the right boundary

partition(c::AbstractVector, Nc, Nr, r) = c[1 + (r-1) * Nc : 1 + Nc * r]
partition(c::Colon,          Nc, Nr, r) = Colon()
partition(c::Tuple,          Nc, Nr, r) = (c[1] + (r-1) * (c[2] - c[1]) / Nr,    c[1] + r * (c[2] - c[1]) / Nr)

function partition(c::UnitRange, Nc, Nr, r)
    g = (first(c), last(r))
    ℓ = partition(g, Nc, Nr, r)
    return UnitRange(ℓ[1], ℓ[2])
end

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

assemble(c::Tuple, Nc, Nr, r, arch) = (c[2] - r * (c[2] - c[1]), c[2] - (r - Nr) * (c[2] - c[1]))

"""
    partition_global_array(arch, c_global, (Nx, Ny, Nz))

partition a global array (2D of size Nx, Ny or 3D of size Nx, Ny, Nz) in local arrays.
Usefull for boundary arrays, forcings and initial conditions
"""
partition_global_array(arch, c_global::Function) = c_global 

function partition_global_array(arch, c_global::AbstractArray, N) 
    c_global = arch_array(CPU(), c_global)
    Rx, Ry, Rz = R = arch.ranks
    ri, rj, rk = r = arch.local_index

    dims = length(size(c_global))

    if length(dims) == 2 
        nx, ny = n = Int.(N ./ R[1:2])

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
reconstruct_global_array(arch, c_local::Function) = c_local

function reconstruct_global_array(arch, c_local::AbstractArray, n) 
    c_local = arch_array(CPU(), c_local)
    Rx, Ry, Rz = R = arch.ranks
    ri, rj, rk = r = arch.local_index

    dims = length(size(c_local))

    if length(dims) == 2 
        nx, ny = n
        Nx, Ny = N = Int.(n .* R[1:2])
    
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