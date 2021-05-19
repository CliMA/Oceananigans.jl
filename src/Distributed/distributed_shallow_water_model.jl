using Oceananigans.Grids
using Oceananigans.Grids: halo_size

#####
##### Distributed shallow water model constructor
#####

function DistributedShallowWaterModel(; architecture, grid, boundary_conditions=nothing, model_kwargs...)
    i, j, k = architecture.local_index
    Rx, Ry, Rz = architecture.ranks
    my_connectivity = architecture.connectivity

    Nx, Ny = size(grid)
    Lx, Ly = length(grid)

    # Pull out endpoints for full grid.
    xL, xR = grid.xF[1], grid.xF[Nx+1]
    yL, yR = grid.yF[1], grid.yF[Ny+1]

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)

    nx, ny = Nx÷Rx, Ny÷Ry
    lx, ly = Lx/Rx, Ly/Ry

    x₁, x₂ = xL + (i-1)*lx, xL + i*lx
    y₁, y₂ = yL + (j-1)*ly, yL + j*ly

    # FIXME: local grid might have different topology!
    my_grid = RegularRectilinearGrid(topology=topology(grid), size=(nx, ny), x=(x₁, x₂), y=(y₁, y₂), halo=(3,3))

    ## Construct local model

    my_model = ShallowWaterModel(;
               architecture = architecture,
                       grid = my_grid,
    #    boundary_conditions = communicative_bcs,
                       model_kwargs...
    )

    return my_model
end
