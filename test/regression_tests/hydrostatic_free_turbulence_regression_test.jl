using Oceananigans.Fields: FunctionField
using Oceananigans.Grids: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, VectorInvariant
using Oceananigans.TurbulenceClosures: HorizontalScalarDiffusivity

using Oceananigans.DistributedComputations: Distributed, DistributedGrid, DistributedComputations, all_reduce
using Oceananigans.DistributedComputations: reconstruct_global_topology, partition_global_array, cpu_architecture

using JLD2

ordered_indices(r, i) = i == 1 ? r : i == 2 ? (r[2], r[1], r[3]) : (r[3], r[2], r[1])

global_topology(grid, i) = string(topology(grid, i))

function global_topology(grid::DistributedGrid, i) 
    arch = architecture(grid)
    R = arch.ranks[i]
    r = ordered_indices(arch.local_index, i)
    T = reconstruct_global_topology(topology(grid, i), R, r..., arch.communicator)
    return string(T)
end

function run_hydrostatic_free_turbulence_regression_test(grid, free_surface; regenerate_data=false)

    #####
    ##### Constructing Grid and model
    #####
    
    model = HydrostaticFreeSurfaceModel(grid = grid,
                          momentum_advection = VectorInvariant(),
                                free_surface = free_surface,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                     closure = HorizontalScalarDiffusivity(ν=1e+5, κ=1e+4))
    
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
    U       = all_reduce(max, U, architecture(grid))
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
    Δt = all_reduce(min, Δt, architecture(grid))

    #####
    ##### Simulation setup
    #####

    stop_iteration = 20

    simulation = Simulation(model,
                            Δt = Δt,
                            stop_iteration = stop_iteration)

    η = model.free_surface.η

    free_surface_str = string(typeof(model.free_surface).name.wrapper)
    x_topology_str = global_topology(grid, 1)
    output_filename = "hydrostatic_free_turbulence_regression_$(x_topology_str)_$(free_surface_str).jld2"

    if regenerate_data && !(grid isa DistributedGrid) # never regenerate on Distributed
        @warn "Generating new data for the Hydrostatic regression test."
        
        directory =  joinpath(dirname(@__FILE__), "data")
        outputs   = (; u, v, w, η)
        simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                              dir = directory,
                                                              schedule = IterationInterval(stop_iteration),
                                                              filename = output_filename,
                                                              with_halos = true,
                                                              overwrite_existing = true)
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

    if !regenerate_data
        datadep_path = "regression_test_data/" * output_filename
        regression_data_path = @datadep_str datadep_path
        file = jldopen(regression_data_path)

        cpu_arch = cpu_architecture(architecture(grid))

        # Data was saved with 2 halos per direction (see issue #3260)
        H = 2
        truth_fields = (
            u = partition_global_array(cpu_arch, file["timeseries/u/$stop_iteration"][H+1:end-H, H+1:end-H, H+1:end-H], size(u)),
            v = partition_global_array(cpu_arch, file["timeseries/v/$stop_iteration"][H+1:end-H, H+1:end-H, H+1:end-H], size(v)),
            w = partition_global_array(cpu_arch, file["timeseries/w/$stop_iteration"][H+1:end-H, H+1:end-H, H+1:end-H], size(w)),
            η = partition_global_array(cpu_arch, file["timeseries/η/$stop_iteration"][H+1:end-H, H+1:end-H, :], size(η))
        )

        close(file)

        summarize_regression_test(test_fields, truth_fields)

        test_fields_equality(cpu_arch, test_fields, truth_fields)
    end
    
    return nothing
end

function test_fields_equality(arch, test_fields, truth_fields)
    @test all(test_fields.u .≈ truth_fields.u)
    @test all(test_fields.v .≈ truth_fields.v)
    @test all(test_fields.w .≈ truth_fields.w)
    @test all(test_fields.η .≈ truth_fields.η)

    return nothing
end

# function test_fields_equality(::Distributed, test_fields, truth_fields)
#     rtol = 10 * sqrt(eps(eltype(truth_fields.u)))

#     @test all(isapprox.(test_fields.u, truth_fields.u; rtol))
#     @test all(isapprox.(test_fields.v, truth_fields.v; rtol))
#     @test all(isapprox.(test_fields.w, truth_fields.w; rtol))
#     @test all(isapprox.(test_fields.η, truth_fields.η; rtol))

#     return nothing
# end

