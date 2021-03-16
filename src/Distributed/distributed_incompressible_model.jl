using Oceananigans.Grids
using Oceananigans.Fields

#####
##### Distributed incompressible model constructor
#####

function DistributedIncompressibleModel(; architecture, grid, model_kwargs...)
    i, j, k = architecture.local_index
    Rx, Ry, Rz = architecture.ranks
    my_connectivity = architecture.connectivity

    Nx, Ny, Nz = size(grid)
    Lx, Ly, Lz = length(grid)

    # Pull out endpoints for full grid.
    xL, xR = grid.xF[1], grid.xF[Nx+1]
    yL, yR = grid.yF[1], grid.yF[Ny+1]
    zL, zR = grid.zF[1], grid.zF[Nz+1]

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    # TODO: Check that we have enough grid points on each rank to fit the halos!
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)

    nx, ny, nz = Nx÷Rx, Ny÷Ry, Nz÷Rz
    lx, ly, lz = Lx/Rx, Ly/Ry, Lz/Rz

    x₁, x₂ = xL + (i-1)*lx, xL + i*lx
    y₁, y₂ = yL + (j-1)*ly, yL + j*ly
    z₁, z₂ = zL + (k-1)*lz, zL + k*lz

    # FIXME? local grid might have different topology!
    my_grid = RegularRectilinearGrid(topology=topology(grid), size=(nx, ny, nz), x=(x₁, x₂), y=(y₁, y₂), z=(z₁, z₂), halo=halo_size(grid))

    ## Construct local model

    pressure_solver = haskey(model_kwargs, :pressure_solver) ? Dict(model_kwargs)[:pressure_solver] :
                                                               DistributedFFTBasedPoissonSolver(architecture, grid, my_grid)

    my_model = IncompressibleModel(;
           architecture = architecture,
                   grid = my_grid,
        pressure_solver = pressure_solver,
                   model_kwargs...
    )

    return my_model
end
