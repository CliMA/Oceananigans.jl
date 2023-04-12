using Oceananigans
using Oceananigans.Distributed
using Oceananigans.Distributed: partition_global_array
using Oceananigans.Grids: architecture
using Oceananigans.Units
using MPI

MPI.Init()

comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

topo = (Bounded, Periodic, Bounded)
arch = DistributedArch(CPU(); topology = topo, 
                 ranks=(Nranks, 1, 1),
                 use_buffers = true)

Lh = 100kilometers
Lz = 400meters

Nx = [10, 13, 18, 39]

grid = RectilinearGrid(arch,
                       size = (Nx[rank+1], 2, 1),
                       x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
                       topology = topo,
                       )


array_full = zeros(prod(Nx), 2)
for element in 1:prod(Nx)
    array_full[element, :] .= element
end

arr = partition_global_array(architecture(grid), array_full, size(grid))

@info "on rank $rank" size(grid) arr
for r in 0:Nranks-1
    if r == rank
        @show rank arr
    end
    MPI.Barrier(MPI.COMM_WORLD)
end