using JLD2

using Oceananigans.Fields: FunctionField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, fᶠᶠᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, VectorInvariant
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.AbstractOperations: KernelFunctionOperation, volume

function run_hydrostatic_free_turbulence_regression_test(grid, free_surface; regenerate_data=false)

    #####
    ##### Constructing Grid and model
    #####
    
    model = HydrostaticFreeSurfaceModel(grid = grid,
                          momentum_advection = VectorInvariant(),
                                free_surface = free_surface,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                     closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=1e+5, κh=1e+4))
    
    #####
    ##### Imposing initial conditions:
    #####    u = function of latitude
    #####    v = function of longitude
    #####    vertical shear for u-velocity
    #####

    step_function(x, d, c) = 1/2 * (1 + tanh((x - c) / d))
    polar_mask(y)          = step_function(y, -5, 40) * step_function(y, 5, -40)
    shear_func(x, y, z, p) = p.U * (0.5 + z / p.Lz) * polar_mask(y)
    
    set!(model, u = (λ, φ, z) -> polar_mask(φ) * exp(-φ^2 / 200),
                v = (λ, φ, z) -> polar_mask(φ) * sind(2λ))

    u, v, w = model.velocities
    U       = 0.1 * maximum(abs, u)
    shear   = FunctionField{Face, Center, Center}(shear_func, grid, parameters=(U=U, Lz=grid.Lz))
    u      .= u + shear

    # Time-scale for gravity wave propagation across the smallest grid cell
    # wave_speed is the hydrostatic (shallow water) gravity wave speed
    gravity    = model.free_surface.gravitational_acceleration
    wave_speed = sqrt(gravity * grid.Lz)                                 
    
    CUDA.allowscalar(true)
    minimum_Δx = grid.radius * cosd(maximum(abs, view(grid.φᵃᶜᵃ, 1:grid.Ny))) * deg2rad(minimum(grid.Δλᶜᵃᵃ))
    minimum_Δy = grid.radius * deg2rad(minimum(grid.Δφᵃᶜᵃ))
    CUDA.allowscalar(false)

    wave_time_scale = min(minimum_Δx, minimum_Δy) / wave_speed

    # Δt based on wave propagation time scale
    Δt = 0.2 * wave_time_scale
    
    #####
    ##### Simulation setup
    #####

    stop_iteration = 20

    simulation = Simulation(model,
                            Δt = Δt,
                            stop_iteration = stop_iteration)

    η = model.free_surface.η

    free_surface_str = string(typeof(model.free_surface).name.wrapper)
    x_topology_str = string(topology(grid, 1))
    output_prefix = "hydrostatic_free_turbulence_regression_$(x_topology_str)_$free_surface_str"

    if regenerate_data
        @warn "Generating new data for the Hydrostatic regression test."
        
        directory =  joinpath(dirname(@__FILE__), "data")
        outputs   = (; u, v, w, η)
        simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                              dir = directory,
                                                         schedule = IterationInterval(stop_iteration),
                                                           prefix = output_prefix,
                                                            force = true)
    end
   

    # Let's gooooooo!
    run!(simulation)

    # Test results
    test_fields = (
        u = Array(interior(u)),
        v = Array(interior(v)),
        w = Array(interior(w)),
        η = Array(interior(η))
    )

    datadep_path = "regression_test_data/" * output_prefix * ".jld2"
    regression_data_path = @datadep_str datadep_path
    file = jldopen(regression_data_path)

    truth_fields = (
        u = file["timeseries/u/$stop_iteration"][:, :, :],
        v = file["timeseries/v/$stop_iteration"][:, :, :],
        w = file["timeseries/w/$stop_iteration"][:, :, :],
        η = file["timeseries/η/$stop_iteration"][:, :, :]
    )

    close(file)

    summarize_regression_test(test_fields, truth_fields)

    @test all(test_fields.u .≈ truth_fields.u)
    @test all(test_fields.v .≈ truth_fields.v)
    @test all(test_fields.w .≈ truth_fields.w)
    @test all(test_fields.η .≈ truth_fields.η)

    return nothing
end
