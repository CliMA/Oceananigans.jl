#####
##### Dynamic launch configuration
#####

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
        else
            error("Unsupported launch configuration: $dims")
        end

        return (threads=Tuple(threads), blocks=Tuple(blocks))
    end
end
