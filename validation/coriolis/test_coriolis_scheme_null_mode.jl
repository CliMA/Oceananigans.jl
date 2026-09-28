using Oceananigans
using Oceananigans.Units

include("oriented_coriolis.jl")

function run_coriolis(scheme)

    grid = RectilinearGrid(size = (16, 16, 1), extent = (10kilometers, 10kilometers, 100))
    coriolis = FPlane(; f = 1e-4, scheme)
    model = HydrostaticFreeSurfaceModel(grid; coriolis, momentum_advection = nothing, free_surface = SplitExplicitFreeSurface(grid, substeps = 20), timestepper = :SplitRungeKutta3)

    vi = [(-1)^i for i in 1:16, j in 1:16]
    ui = [(-1)^j for i in 1:16, j in 1:16]
    set!(model, u = ui)

    ur = []
    vr = []
    er = []

    for _ in 1:1000
        time_step!(model, 2.5minutes)
        push!(ur, deepcopy(interior(model.velocities.u, :, :, 1)))
        push!(vr, deepcopy(interior(model.velocities.v, :, :, 1)))
        push!(er, deepcopy(interior(model.free_surface.displacement, :, :, 1)))
    end

    return ur, vr, er
end

