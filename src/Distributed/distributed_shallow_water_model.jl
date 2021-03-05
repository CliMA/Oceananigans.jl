using Oceananigans.Grids
using Oceananigans.Grids: halo_size

#####
##### Distributed shallow water model constructor
#####

function DistributedShallowWaterModel(; architecture, grid, boundary_conditions=nothing, model_kwargs...)
    my_rank = architecture.my_rank
    i, j, k = architecture.my_index
    Rx, Ry, Rz = architecture.ranks
    my_connectivity = architecture.connectivity

    Nx, Ny, Nz = size(grid)
    Lx, Ly, Lz = length(grid)

    @assert Nz == 1
    @assert Rz == 1

    # Pull out endpoints for full grid.
    xL, xR = grid.xF[1], grid.xF[Nx+1]
    yL, yR = grid.yF[1], grid.yF[Ny+1]
    zL, zR = grid.zF[1], grid.zF[Nz+1]

    # Make sure we can put an integer number of grid points in each rank.
    # Will generalize in the future.
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)

    nx, ny = Nx÷Rx, Ny÷Ry
    lx, ly = Lx/Rx, Ly/Ry

    x₁, x₂ = xL + (i-1)*lx, xL + i*lx
    y₁, y₂ = yL + (j-1)*ly, yL + j*ly

    # FIXME: local grid might have different topology!
    my_grid = RegularRectilinearGrid(topology=topology(grid), size=(nx, ny, Nz), x=(x₁, x₂), y=(y₁, y₂), z=(zL, zR), halo=halo_size(grid))

    ## Change appropriate boundary conditions to halo communication BCs

    # FIXME: Stop assuming (uh, vh, h).

    bcs = isnothing(boundary_conditions) ? NamedTuple() : boundary_conditions

    bcs = (
        uh = haskey(bcs, :uh) ? bcs.uh : UVelocityBoundaryConditions(my_grid),
        vh = haskey(bcs, :vh) ? bcs.vh : VVelocityBoundaryConditions(my_grid),
         h = haskey(bcs, :h)  ? bcs.h  : TracerBoundaryConditions(my_grid)
    )

    communicative_bcs = (
        uh = inject_halo_communication_boundary_conditions(bcs.uh, my_rank, my_connectivity),
        vh = inject_halo_communication_boundary_conditions(bcs.vh, my_rank, my_connectivity),
         h = inject_halo_communication_boundary_conditions(bcs.h,  my_rank, my_connectivity)
    )

    ## Construct local model

    my_model = ShallowWaterModel(;
               architecture = architecture,
                       grid = my_grid,
        boundary_conditions = communicative_bcs,
                       model_kwargs...
    )

    return my_model
end
