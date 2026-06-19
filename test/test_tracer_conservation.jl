include("dependencies_for_runtests.jl")

@testset "Tracer conservation" begin
    for arch in archs
        size = (22, 20, 10)
        H = 5000                   # domain depth [m]
        latitude = (-85, 85)       # latitude range (avoiding poles for lat-lon grid)
        longitude = (0, 360)       # longitude range
        z = (-H, 0)                # vertical extent
        radius = Oceananigans.defaults.planet_radius

        rectilinear_grid = RectilinearGrid(arch; size, x=(-10, 10), y=(-4, 4), z, topology = (Bounded, Bounded, Bounded))
        latitude_longitude_grid = LatitudeLongitudeGrid(arch; size, longitude, latitude, z, radius)

        halo = (7, 7, 7)
        underlying_tripolar_grid = TripolarGrid(arch; size, halo, z, radius)

        σφ, σλ = 4, 8       # mountain extent in latitude and longitude (degrees)
        λ₀, φ₀ = 70, 55     # first pole location
        h = H + 1000        # mountain height above the bottom (m)

        gaussian(λ, φ) = exp(-((λ - λ₀)^2 / 2σλ^2 + (φ - φ₀)^2 / 2σφ^2))
        gaussian_mountains(λ, φ) = -H + h * (gaussian(λ, φ) + gaussian(λ - 180, φ) + gaussian(λ - 360, φ))

        tripolar_grid = ImmersedBoundaryGrid(underlying_tripolar_grid, GridFittedBottom(gaussian_mountains))

        for grid in (rectilinear_grid, latitude_longitude_grid, tripolar_grid)
            @info "Testing tracer conservation on $(summary(grid))"

            # We initialize a tracer field and check that the total tracer content changes only due
            # to the flux through the surface boundary.

            Jᶜ(λ, φ, t) = -2.2e-4 + 6.1e-5 * exp(-λ^2 - φ^2)
            c_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᶜ))

            models = Oceananigans.AbstractModel[]

            hydrostatic_model = HydrostaticFreeSurfaceModel(grid,
                                                            timestepper=:SplitRungeKutta3,
                                                            tracers=:c,
                                                            boundary_conditions=(c=c_bcs,),
                                                            free_surface=SplitExplicitFreeSurface(substeps=15))

            push!(models, hydrostatic_model)

            if grid !== tripolar_grid
                nonhydrostatic_model = NonhydrostaticModel(grid,
                                                           tracers=:c,
                                                           boundary_conditions=(c=c_bcs,))

                push!(models, nonhydrostatic_model)
            end

            for model in models
                @info "... on $(summary(model))"
                set!(model, c = (λ, φ, z) -> -z / H * cosd(λ)^2 * cosd(φ))
                c = model.tracers.c

                total_c = Integral(c, dims=(1, 2, 3))
                surface_c_flux = Integral(Oceananigans.Models.BoundaryConditionField(c, :top, model), dims=(1, 2))

                Δt = 2.3
                simulation = Simulation(model; Δt, stop_iteration=4, verbose=false)

                for _ = 1:simulation.stop_iteration
                    previous_total_c = first(Field(total_c))
                    previous_surface_c_flux = first(Field(surface_c_flux))

                    time_step!(simulation, Δt)

                    current_total_c = first(Field(total_c))
                    last_Δt = simulation.model.clock.last_Δt

                    actual_Δc = current_total_c - previous_total_c
                    expected_Δc = - previous_surface_c_flux * last_Δt
                    predicted_total_c = previous_total_c + expected_Δc

                    @test actual_Δc ≈ expected_Δc
                    @test current_total_c ≈ predicted_total_c
                end
            end
        end
    end
end
