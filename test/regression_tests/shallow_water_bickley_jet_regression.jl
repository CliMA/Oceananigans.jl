using Oceananigans
using Oceananigans.Advection: VelocityStencil
using Oceananigans.Models.ShallowWaterModels: VectorInvariantFormulation, ConservativeFormulation

Lx, Ly, Lz = 2π, 20, 10
Nx, Ny = 128, 128

advection(formulation::ConservativeFormulation)    = WENO()
advection(formulation::VectorInvariantFormulation) = VectorInvariant(vorticity_scheme = WENO(), divergence_scheme = nothing, vertical_scheme = EnergyConservingScheme())

function run_shallow_water_regression(arch, formulation; regenerate_data = false)

    grid = RectilinearGrid(arch, size = (Nx, Ny),
                           x = (0, Lx), y = (-Ly/2, Ly/2),
                           topology = (Periodic, Bounded, Flat),
                           halo = (4, 4))

    gravitational_acceleration = 1
    coriolis = FPlane(f=1)

    model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration, formulation,
                              timestepper = :RungeKutta3,
                              momentum_advection = advection(formulation))

    U = 1 # Maximum jet velocity
    f = coriolis.f
    g = gravitational_acceleration
    Δη = f * U / g  # Maximum free-surface deformation as dictated by geostrophy

    h̄(x, y, z) = Lz - Δη * tanh(y)
    ū(x, y, z) = U * sech(y)^2

    small_amplitude = 1e-4
    
    uⁱ(x, y, z) = ū(x, y, z) + small_amplitude * exp(-y^2) * cos(8π * x / Lx)
    uhⁱ(x, y, z) = uⁱ(x, y, z) * h̄(x, y, z)

    if formulation isa VectorInvariantFormulation
        set!(model, u = uⁱ, h = h̄)
    else
        set!(model, uh = uhⁱ, h = h̄)
    end

    stop_iteration = 20
    Δt = 5e-3

    simulation = Simulation(model; stop_iteration, Δt)
    
    ## Build velocities
    u, v, _ = model.velocities
    h = model.solution.h

    output_filename = "shallow_water_bickley_jet_regression_$(typeof(formulation)).jld2"

    if regenerate_data
        @warn "Generating new data for the ShallowWater Bickley Jet regression test."
        
        directory =  joinpath(dirname(@__FILE__), "data")
        outputs   = (; u, v, h)
        simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                              dir = directory,
                                                              array_type = Array{Float32},
                                                              schedule = IterationInterval(stop_iteration),
                                                              filename = output_filename,
                                                              overwrite_existing = true)
    end
   
    run!(simulation)

    u, v, _ = model.velocities
    h = model.solution.h

    # Test results
    test_fields = (
        u = Array(interior(u)),
        v = Array(interior(v)),
        h = Array(interior(h))
    )

    if !regenerate_data
        datadep_path = "regression_test_data/" * output_filename
        regression_data_path = @datadep_str datadep_path
        file = jldopen(regression_data_path)

        truth_fields = (
            u = file["timeseries/u/$stop_iteration"][:, :, 1],
            v = file["timeseries/v/$stop_iteration"][:, :, 1],
            h = file["timeseries/h/$stop_iteration"][:, :, 1],
        )

        close(file)

        summarize_regression_test(test_fields, truth_fields)

        @test all(test_fields.u .≈ truth_fields.u)
        @test all(test_fields.v .≈ truth_fields.v)
        @test all(test_fields.h .≈ truth_fields.h)
    end
end
