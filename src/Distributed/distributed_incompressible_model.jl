using Oceananigans.Grids
using Oceananigans.Fields

#####
##### Distributed incompressible model constructor
#####

function DistributedIncompressibleModel(; architecture, grid, boundary_conditions=nothing, model_kwargs...)
    my_rank = architecture.my_rank
    i, j, k = architecture.my_index
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
    @assert isinteger(Nx / Rx)
    @assert isinteger(Ny / Ry)
    @assert isinteger(Nz / Rz)

    nx, ny, nz = Nx÷Rx, Ny÷Ry, Nz÷Rz
    lx, ly, lz = Lx/Rx, Ly/Ry, Lz/Rz

    x₁, x₂ = xL + (i-1)*lx, xL + i*lx
    y₁, y₂ = yL + (j-1)*ly, yL + j*ly
    z₁, z₂ = zL + (k-1)*lz, zL + k*lz

    # FIXME: local grid might have different topology!
    my_grid = RegularRectilinearGrid(topology=topology(grid), size=(nx, ny, nz), x=(x₁, x₂), y=(y₁, y₂), z=(z₁, z₂), halo=halo_size(grid))

    ## Change appropriate boundary conditions to halo communication BCs

    # FIXME: Stop assuming (u, v, w, T, S).

    bcs = isnothing(boundary_conditions) ? NamedTuple() : boundary_conditions

    bcs = (
        u = haskey(bcs, :u) ? bcs.u : UVelocityBoundaryConditions(my_grid),
        v = haskey(bcs, :v) ? bcs.v : VVelocityBoundaryConditions(my_grid),
        w = haskey(bcs, :w) ? bcs.w : WVelocityBoundaryConditions(my_grid),
        T = haskey(bcs, :T) ? bcs.T : TracerBoundaryConditions(my_grid),
        S = haskey(bcs, :S) ? bcs.S : TracerBoundaryConditions(my_grid)
    )

    communicative_bcs = (
        u = inject_halo_communication_boundary_conditions(bcs.u, my_rank, my_connectivity),
        v = inject_halo_communication_boundary_conditions(bcs.v, my_rank, my_connectivity),
        w = inject_halo_communication_boundary_conditions(bcs.w, my_rank, my_connectivity),
        T = inject_halo_communication_boundary_conditions(bcs.T, my_rank, my_connectivity),
        S = inject_halo_communication_boundary_conditions(bcs.S, my_rank, my_connectivity)
    )

    ## Construct local model

    pressure_solver = haskey(model_kwargs, :pressure_solver) ? Dict(model_kwargs)[:pressure_solver] :
                                                               DistributedFFTBasedPoissonSolver(architecture, grid, my_grid)

    p_bcs = PressureBoundaryConditions(my_grid)
    p_bcs = inject_halo_communication_boundary_conditions(p_bcs, my_rank, my_connectivity)

    pHY′ = CenterField(architecture, my_grid, p_bcs)
    pNHS = CenterField(architecture, my_grid, p_bcs)
    pressures = (pHY′=pHY′, pNHS=pNHS)

    my_model = IncompressibleModel(;
               architecture = architecture,
                       grid = my_grid,
        boundary_conditions = communicative_bcs,
            pressure_solver = pressure_solver,
                  pressures = pressures,
                       model_kwargs...
    )

    return my_model
end
