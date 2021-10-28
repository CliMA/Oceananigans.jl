
using Oceananigans.Fields: FunctionField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, fᶠᶠᵃ
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, VectorInvariant
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.AbstractOperations: KernelFunctionOperation, volume

function run_lat_lon_free_turbulence_regression_test(grid_λ, grid_φ, grid_z, arch, precompute; regenerate_data=false)

    #####
    ##### Constructing Grid and model
    #####
    
    N = (180, 60, 3)
    H = (  2,  2, 2)

    if grid_λ == :regular
        lon = (-180, 180)
    else
        lon =  collect(-180:2:180)
    end
   
    if grid_φ == :regular
        lat = (-60, 60)
    else
        lat =  collect(-60:2:60)
    end
    
    if grid_z == :regular
        z = (-90, 0)
    else
        z =  collect(-90:30:0)
    end

    free_surface = ExplicitFreeSurface(gravitational_acceleration=1.0)

    grid = LatitudeLongitudeGrid(size = N, longitude = lon, latitude = lat, z = z, halo = H, architecture = arch,
                                 precompute_metrics = precompute)
                                 
    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        architecture = arch,
                                        momentum_advection = VectorInvariant(),
                                        free_surface = free_surface,
                                        coriolis = HydrostaticSphericalCoriolis(),
                                        tracers = :c,
                                        buoyancy = nothing,
                                        closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=1e+5, κh=1e+4))  
    
    #####
    ##### Imposing initial conditions:
    #####    u = function of latitude
    #####    v = function of longitude
    #####    vertical shear for u-velocity
    #####

    step_function(x, d, c) = 1/2 * (1 + tanh((x - c) / d))
    polar_mask(y)          = step_function(y, -5, 40) * step_function(y, 5, -40)
    
    set!(model, c = (λ, φ, z) -> cosd(2λ) * cosd(4φ) * z,
                u = (λ, φ, z) -> polar_mask(φ) * exp(-φ^2 / 200),
                v = (λ, φ, z) -> polar_mask(φ) * sind(2λ)) 

    u, v, w = model.velocities

    U = 0.1 * maximum(abs, u)
    shear_func(x, y, z, p) = p.U * (0.5 + z / p.Lz) * polar_mask(y)
    shear = FunctionField{Face, Center, Center}(shear_func, grid, parameters=(U=U, Lz=grid.Lz))
    u .= u + shear

    #####
    ##### Shenanigans for rescaling the velocity field to have a magnitude (ish) that's a fixed fraction of the surface gravity wave speed;
    ##### 

    CUDA.allowscalar(true)

    # Time-scale for gravity wave propagation across the smallest grid cell
    g = model.free_surface.gravitational_acceleration
    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

    minimum_Δx = grid.radius * cosd(maximum(abs, view(grid.φᵃᶜᵃ, 1:grid.Ny))) * deg2rad(minimum(grid.Δλᶜᵃᵃ))
    minimum_Δy = grid.radius * deg2rad(minimum(grid.Δφᵃᶜᵃ))
    wave_propagation_time_scale = min(minimum_Δx, minimum_Δy) / gravity_wave_speed

    Δt = 0.2 * wave_propagation_time_scale

    max_u = maximum(abs, u)
    max_v = maximum(abs, v)

    max_speed_ish = sqrt(max_u^2 + max_v^2)
    target_speed = 0.5 * gravity_wave_speed

    u .*= target_speed / max_speed_ish
    v .*= target_speed / max_speed_ish

    u_cpu = XFaceField(CPU(), grid)
    v_cpu = YFaceField(CPU(), grid)
    set!(u_cpu, u)
    set!(v_cpu, v)

    max = maximum(u)
    max_v = maximum(v)

    u_dV = u_cpu * volume
    u_reduced = AveragedField(u_dV, dims=(1, 2, 3))
    mean!(u_reduced, u_dV)
    integrated_u = u_reduced[1, 1, 1]

    v_dV = v_cpu * volume
    v_reduced = AveragedField(v_dV, dims=(1, 2, 3))
    mean!(v_reduced, v_dV)
    integrated_v = v_reduced[1, 1, 1]

    # Calculate total volume
    u_cpu .= 1
    v_cpu .= 1
    compute!(u_reduced)
    compute!(v_reduced)

    u_volume = u_reduced[1, 1, 1]
    v_volume = v_reduced[1, 1, 1]

    u .-= integrated_u / u_volume
    v .-= integrated_v / v_volume
    
    CUDA.allowscalar(false) 

    #####
    ##### Simulation setup
    #####

    start_time = [time_ns()]
    function progress(sim)
        wall_time = (time_ns() - start_time[1]) * 1e-9

        @info @sprintf("Time: %s, iteration: %d, max(|η|): %.2e s⁻¹, wall time: %s",
                    prettytime(sim.model.clock.time),
                    sim.model.clock.iteration,
                    maximum(abs, sim.model.free_surface.η),
                    prettytime(wall_time))

        start_time[1] = time_ns()

        return nothing
    end

    stop_iteration = 20

    simulation = Simulation(model,
                            Δt = Δt,
                            stop_iteration = stop_iteration,
                            iteration_interval = 10,
                            progress = progress)

    c = model.tracers.c
    η = model.free_surface.η

    # Uncomment to regenerate regression test data
    #=
    if regenerate_data
        @warn "Generating new data for the ocean LES regression test."
    
        outputs = (; u, v, w, c, η)
        simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                              schedule = IterationInterval(stop_iteration),
                                                              prefix = output_prefix,
                                                              force = true)
    end
    =#

    # Let's gooooooo!
    run!(simulation)

    test_fields = (
        u = Array(interior(u)),
        v = Array(interior(v)),
        w = Array(interior(w)),
        c = Array(interior(c)),
        η = Array(interior(η))
    )

    free_surface_str = string(typeof(model.free_surface).name.wrapper)
    output_prefix = "lat_lon_free_turbulence_regression_$free_surface_str"
    regression_data_path = joinpath(dirname(@__FILE__), "data", output_prefix * ".jld2")
    file = jldopen(regression_data_path)

    truth_fields = (
        u = file["timeseries/u/$stop_iteration"][:, :, :],
        v = file["timeseries/v/$stop_iteration"][:, :, :],
        w = file["timeseries/w/$stop_iteration"][:, :, :],
        c = file["timeseries/c/$stop_iteration"][:, :, :],
        η = file["timeseries/η/$stop_iteration"][:, :, :]
    )

    close(file)

    summarize_regression_test(test_fields, truth_fields)

    @test all(test_fields.u .≈ truth_fields.u)
    @test all(test_fields.v .≈ truth_fields.v)
    @test all(test_fields.w .≈ truth_fields.w)
    @test all(test_fields.c .≈ truth_fields.c)
    @test all(test_fields.η .≈ truth_fields.η)

    return nothing
end
