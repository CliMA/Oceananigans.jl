include("dependencies_for_runtests.jl")


function initialise_simulation(model)

    # Initial condition that excites a baroclinic wave
  Tᵢ(λ, φ, z) = 30 * (1 - tanh((abs(φ) - 45) / 8)) / 2 + rand()
  Sᵢ(λ, φ, z) = 28 - 5e-3 * z + rand()
  set!(model, T=Tᵢ, S=Sᵢ)

  return Simulation(model, Δt = 1, stop_iteration=100)
end

@testset "Test HydrostaticFreeSurfaceModel with split cells maps" begin
    @info "Testing for numerical divergence when using split cells map..."

    for grid in grids
        reference_model = HydrostaticFreeSurfaceModel(grid;
                                                      condition_momentum_advection=false,
                                                      condition_tracer_advection=false,
        )

        reference_simulation = initialise_simulation(reference_model)

        run!(reference_simulation)

        test_model = HydrostaticFreeSurfaceModel(grid;
                                                 condition_momentum_advection=true,
                                                 condition_tracer_advection=true,
        )

        test_simulation = initialise_simulation(test_model)

        run!(test_simulation)

        ur, vr = reference_model.velocities
        ur = interior(ur)
        vr = interior(vr)

        Tr, Sr = reference_model.tracers
        Tr = interior(Tr)
        Sr = interior(Sr)

        ut, vt = test_model.velocities
        ut = interior(ut)
        vt = interior(vt)

        Tt, St = test_model.tracers
        Tt = interior(Tt)
        St = interior(St)

        @test ut ≈ ur
        @test vt ≈ vr
        @test Tt ≈ Tr
        @test St ≈ Sr

    end
end
