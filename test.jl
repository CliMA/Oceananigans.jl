using Revise
using Oceananigans
using Oceananigans.Utils

using KernelAbstractions: @index, @kernel

arch = CPU()

@kernel function _test_indices(a)
    i, j, k = @index(Global, NTuple)
    a[i, j, k] = i + j + k
end

grid  = RectilinearGrid(arch, size = (3, 3, 3), extent = (1, 1, 1))
array = zeros(arch, 3, 3, 3)
imap  = on_architecture(arch, [(i, j, k) for i in 1:3, j in 1:3, k in 1:2])

launch!(arch, grid_cpu, :xyz, _test_indices, array; active_cells_map = imap)