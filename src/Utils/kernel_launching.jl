#####
##### Dynamic launch configuration
#####

using CUDA

function launch_config(grid, dims)
    return function (kernel)
        fun = kernel.fun
        config = launch_configuration(fun)

        # Adapt the suggested config from 1D to the requested grid dimensions
        if dims == :xyz
            t = floor(Int, cbrt(config.threads))
            threads = [t, t, t]
            blocks  = ceil.(Int, [grid.Nx, grid.Ny, grid.Nz] ./ t)
        elseif dims == :xy
            t = floor(Int, sqrt(config.threads))
            threads = [t, t]
            blocks  = ceil.(Int, [grid.Nx, grid.Ny] ./ t)
        elseif dims == :xz
            t = floor(Int, sqrt(config.threads))
            threads = [t, t]
            blocks  = ceil.(Int, [grid.Nx, grid.Nz] ./ t)
        elseif dims == :yz
            t = floor(Int, sqrt(config.threads))
            threads = [t, t]
            blocks  = ceil.(Int, [grid.Ny, grid.Nz] ./ t)
        else
            error("Unsupported launch configuration: $dims")
        end

        return (threads=Tuple(threads), blocks=Tuple(blocks))
    end
end

# See: https://github.com/climate-machine/Oceananigans.jl/pull/308
function work_layout(grid, dims)

    workgroup = grid.Nx == 1 ? (1, min(256, grid.Ny)) :
                grid.Ny == 1 ? (1, min(256, grid.Nx)) : 
                               (16, 16)
    
    worksize = dims == :xyz ? (grid.Nx, grid.Ny, grid.Nz) :
               dims == :xy  ? (grid.Nx, grid.Ny) :
               dims == :xz  ? (grid.Nx, grid.Nz) :
               dims == :yz  ? (grid.Ny, grid.Nz) : error("Unsupported launch configuration: $dims")

    return workgroup, worksize
end

using Oceananigans.Architectures: device

function launch!(arch, grid, layout, kernel!, args...; sync=true, kwargs...)
    workgroup, worksize = work_layout(grid, layout)
    loop! = kernel!(device(arch), workgroup, worksize)
    event = loop!(args...; ndrange=worksize, kwargs..., dependencies=Event(device(arch)()
     wait(device(arch), event)
    return nothing
end
