struct CPU <: Architecture end

# A slightly complex (but fully-featured?) implementation:

struct ThreadBlockLayout{NT, NB}
    threads :: NTuple{NT, Int}
     blocks :: NTuple{NB, Int}
end

XYZThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads, 
    (floor(Int, threads[1]/grid.Nx), floor(Int, threads[2]/grid.Ny), grid.Nz) )

XYThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads,
    (floor(Int, threads[1]/grid.Nx), floor(Int, threads[2]/grid.Ny)) )

XZThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads,
    (floor(Int, threads[1]/grid.Nx), grid.Nz) )

YZThreadBlockLayout(threads, grid) = ThreadBlockLayout(threads, 
    (floor(Int, threads[2]/grid.Ny), grid.Nz) )

struct GPU{XYZ, XY, XZ, YZ} <: Architecture
    xyz :: XYZ
     xy :: XY
     xz :: XZ
     yz :: YZ
end

GPU(grid; threads=(16, 16)) = GPU(
    XYZThreadBlockLayout(threads, grid), XYThreadBlockLayout(threads, grid),
     XZThreadBlockLayout(threads, grid), YZThreadBlockLayout(threads, grid) )

GPU() = GPU(nothing, nothing, nothing, nothing) # stopgap while code is unchanged.

# Functions permitting generalization:
threads(geom, arch) = nothing
 blocks(geom, arch) = nothing

threads(geom, arch::GPU) = getproperty(getproperty(arch, geom), :threads)
 blocks(geom, arch::GPU) = getproperty(getproperty(arch, geom), :blocks)

# @launch looks like xyz_kernel(args..., threads=threads(:xyz, arch), blocks=blocks(:xyz, arch))
# etc.

device(::CPU) = GPUifyLoops.CPU()
device(::GPU) = GPUifyLoops.CUDA()

