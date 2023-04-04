using Oceananigans.Architectures: arch_array

# Partitioning (localization of global objects) and assembly (global assembly of local objects)
# Used for grid constructors (cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z)
# which means that we need to repeat the value at the right boundary

partition(c::Colon, Nc, Nr, r) = Colon()
partition(c::Tuple, Nc, Nr, r) = (c[1] + (r-1) * (c[2] - c[1]) / Nr,  c[1] + r * (c[2] - c[1]) / Nr)

# Have to fix this! This won't work for face constructors
partition(c::AbstractVector, Nc, Nr, r) = c[1 + (r-1) * Nc : 1 + Nc * r]

function partition(c::UnitRange, Nc, Nr, r)
    g = (first(c), last(r))
    ℓ = partition(g, Nc, Nr, r)
    return UnitRange(ℓ[1], ℓ[2])
end

# Have to fix this! This won't work for face constructors
partition(c::AbstractVector, Nc::AbstractVector, Nr, r) = c[1 + sum(Nc[1:r-1]) : 1 + sum(Nc[1:r])]

function partition(c::Tuple, Nc::AbstractVector, Nr, r)
    Nt = sum(Nc)
    Δl = (c[2] - c[1]) / Nt      

    l = Tuple{Float64, Float64}[(c[1], c[1] + Δl * Nc[1])]
    for i in 2:length(Nc)
        lp = l[i-1][2]
        push!(l, (lp, lp + Δl * Nc[i]))
    end

    return l[r]
end

"""
    assemble(c::AbstractVector, Nc, Nr, r, arch) 

Build a linear global coordinate vector given a local coordinate vector `c_local`
a local number of elements `Nc`, number of ranks `Nr`, rank `r`,
and `arch`itecture. Since we use a global reduction, only ranks at positions
1 in the other two directions `r1 == 1` and `r2 == 1` fill the 1D array.
"""
function assemble(c_local::AbstractVector, Nc, Nr, r, r1, r2, comm) 

    Nl = zeros(Int, Nr)
    Nl[r] = Nc
    MPI.Allreduce!(Nl, +, comm)

    c_global = zeros(eltype(c_local), sum(Nl)+1)

    if r1 == 1 && r2 == 1
        c_global[1 + sum(Nl[1:r-1]) : sum(Nl[1:r])] .= c_local[1:end-1]
        r == Nr && (c_global[end] = c_local[end])
    end

    MPI.Allreduce!(c_global, +, comm)

    return c_global
end

# Simple case, just take the first and the last core
function assemble(c::Tuple, Nc, Nr, r, r1, r2, comm) 
    c_global = zeros(Int, 2)

    if r == 1
        c_global[1] = c[1]
    elseif r == Nr
        c_global[2] = c[2]
    end

    MPI.Allreduce!(c_global, +, comm)

    return tuple(c_global...)
end 

# TODO: partition_global_array and construct_global_array
# do not currently work for 2D or 3D parallelizations
# (They are not used anywhere in the code at the moment)
"""
    partition_global_array(arch, c_global, (nx, ny, nz))

Partition a global array in local arrays of size `(nx, ny)` if 2D or `(nx, ny, nz)` is 3D.
Usefull for boundary arrays, forcings and initial conditions.
"""
partition_global_array(arch, c_global::Function, Nl) = c_global 

# Here we just assume we cannot partition in z (we should remove support for that!!)
function partition_global_array(arch, c_global::AbstractArray, nl) 
    c_global = arch_array(CPU(), c_global)
    Rx, Ry, Rz = R = arch.ranks
    ri, rj, rk = r = arch.local_index

    dims = length(size(c_global))

    nx = zeros(Int, Rx)
    nx[r] = nl[1]
    MPI.Allreduce!(nx, +, comm)

    ny = zeros(Int, Ry)
    ny[r] = nl[2]
    MPI.Allreduce!(ny, +, comm)

    nz = nl[3]

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

# TODO: This does not work for 2D parallelizations!!!
function construct_global_array(arch, c_local::AbstractArray, nl) 
    c_local = arch_array(CPU(), c_local)
    Rx, Ry, Rz = R = arch.ranks
    ri, rj, rk = r = arch.local_index

    dims = length(size(c_local))

    nx = zeros(Int, Rx)
    nx[r] = nl[1]
    MPI.Allreduce!(nx, +, comm)

    ny = zeros(Int, Ry)
    ny[r] = nl[2]
    MPI.Allreduce!(ny, +, comm)
    
    Nx = sum(nxl)
    Ny = sum(nyl)
    Nz = n[3]

    if dims == 2 
        c_global = zeros(eltype(c_local), Nx, Ny)
    
        c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                 1 + sum(ny[1:rj-1]) : sum(ny[1:rj])] .= c_local[1:nx[ri], 1:ny[rj]]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
    else
        c_global = zeros(eltype(c_local), Nx, Ny, Nz)

        c_global[1 + sum(nx[1:ri-1]) : sum(nx[1:ri]), 
                 1 + sum(ny[1:rj-1]) : sum(ny[1:rj]),
                 1:Nz] .= c_local[1:nxl[ri], 1:nyl[rj], 1:Nz]
        
        MPI.Allreduce!(c_global, +, arch.communicator)
    end
    return arch_array(child_architecture(arch), c_global)
end
