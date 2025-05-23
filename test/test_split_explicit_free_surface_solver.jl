include("dependencies_for_runtests.jl")

using Oceananigans.Fields: VelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: calculate_substeps,
                                                                                  calculate_adaptive_settings,
                                                                                  constant_averaging_kernel,
                                                                                  materialize_free_surface,
                                                                                  SplitExplicitFreeSurface,
                                                                                  iterate_split_explicit!

@testset "Split-Explicit Dynamics" begin

    for FT in float_types
        for arch in archs
            topology = (Periodic, Periodic, Bounded)

            Nx, Ny, Nz = 128, 64, 1
            Lx = Ly = 2π
            Lz = 1 / Oceananigans.BuoyancyFormulations.g_Earth

            grid = RectilinearGrid(arch, FT;
                                   topology, size = (Nx, Ny, Nz),
                                   x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                                   halo = (1, 1, 1))

            velocities = VelocityFields(grid)

            sefs = SplitExplicitFreeSurface(substeps = 200, averaging_kernel = constant_averaging_kernel)
            sefs = materialize_free_surface(sefs, velocities, grid)

            sefs.η .= 0
            GU = Field{Face, Center, Nothing}(grid)
            GV = Field{Center, Face, Nothing}(grid)

            @testset " One timestep test " begin
                state = sefs.filtered_state
                U, V  = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η, state.U, state.V

                η = sefs.η
                Δτ = 1.0

                η₀(x, y, z) = sin(x)
                set!(η, η₀)

                Nsubsteps = calculate_substeps(sefs.substepping, 1)
                fractional_Δt, weights = calculate_adaptive_settings(sefs.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                iterate_split_explicit!(sefs, grid, GU, GV, Δτ, weights, Val(1))

                U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
                U_exact = (reshape(-cos.(grid.xᶠᵃᵃ), (length(grid.xᶜᵃᵃ), 1)).+reshape(0 * grid.yᵃᶜᵃ, (1, length(grid.yᵃᶜᵃ))))[2:Nx+1, 2:Ny+1]

                @test maximum(abs.(U_exact - U_computed)) < 1e-3
            end

            @testset "Multi-timestep test " begin
                state = sefs.filtered_state
                U, V = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η, state.U, state.V
                η = sefs.η

                T  = 2π
                Δτ = 2π / maximum([Nx, Ny]) * 5e-2 # the last factor is essentially the order of accuracy
                Nt = floor(Int, T / Δτ)
                Δτ_end = T - Nt * Δτ

                sefs = SplitExplicitFreeSurface(substeps = Nt, averaging_kernel = constant_averaging_kernel)
                sefs = materialize_free_surface(sefs, velocities, grid)

                # set!(η, f(x, y))
                η₀(x, y, z) = sin(x)
                set!(η, η₀)
                set!(U, 0)
                set!(V, 0)

                η̅  .= 0
                U̅  .= 0
                V̅  .= 0
                GU .= 0
                GV .= 0

                weights = sefs.substepping.averaging_weights

                for _ in 1:Nt
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, weights, Val(1))
                end
                iterate_split_explicit!(sefs, grid, GU, GV, Δτ_end, weights, Val(1))

                U_computed = Array(deepcopy(interior(U)))
                η_computed = Array(deepcopy(interior(η)))
                set!(η, η₀)
                set!(U, 0)
                U_exact = Array(deepcopy(interior(U)))
                η_exact = Array(deepcopy(interior(η)))

                @test maximum(abs.(U_computed - U_exact)) < 1e-3
                @test maximum(abs.(η_computed - η_exact)) < max(100eps(FT), 1e-6)
            end

            sefs = SplitExplicitFreeSurface(substeps = 200, averaging_kernel = constant_averaging_kernel)
            sefs = materialize_free_surface(sefs, velocities, grid)

            sefs.η .= 0

            @testset "Averaging / Do Nothing test " begin
                state = sefs.filtered_state
                U, V  = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η, state.U, state.V
                η = sefs.η
                g = sefs.gravitational_acceleration

                Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # the last factor is essentially the order of accuracy

                # set!(η, f(x, y))
                η_avg = 1
                U_avg = 2
                V_avg = 3
                fill!(η, η_avg)
                fill!(U, U_avg)
                fill!(V, V_avg)

                fill!(η̅ , 0)
                fill!(U̅ , 0)
                fill!(V̅ , 0)
                fill!(GU, 0)
                fill!(GV, 0)


                Nsubsteps  = calculate_substeps(sefs.substepping, 1)
                fractional_Δt, weights = calculate_adaptive_settings(sefs.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                for step in 1:Nsubsteps
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, weights, Val(1))
                end

                U_computed = Array(deepcopy(interior(U)))
                V_computed = Array(deepcopy(interior(V)))
                η_computed = Array(deepcopy(interior(η)))

                U̅_computed = Array(deepcopy(interior(U̅)))
                V̅_computed = Array(deepcopy(interior(V̅)))
                η̅_computed = Array(deepcopy(interior(η̅)))

                tolerance = 100eps(FT)

                @test maximum(abs.(U_computed .- U_avg)) < tolerance
                @test maximum(abs.(η_computed .- η_avg)) < tolerance
                @test maximum(abs.(V_computed .- V_avg)) < tolerance

                @test maximum(abs.(U̅_computed .- U_avg)) < tolerance
                @test maximum(abs.(η̅_computed .- η_avg)) < tolerance
                @test maximum(abs.(V̅_computed .- V_avg)) < tolerance
            end

            @testset "Complex Multi-Timestep " begin
                # Test 3: Testing analytic solution to
                # ∂ₜη + ∇⋅U̅ = 0
                # ∂ₜU̅ + ∇η  = G̅
                kx = 2
                ky = 3
                ω = sqrt(kx^2 + ky^2)
                T = 2π / ω / 3 * 2
                Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # error mostly spatially dependent, except in the averaging
                Nt = floor(Int, T / Δτ)
                Δτ_end = T - Nt * Δτ

                sefs = SplitExplicitFreeSurface(grid; substeps = Nt + 1, averaging_kernel = constant_averaging_kernel)
                sefs = materialize_free_surface(sefs, velocities, grid)

                state = sefs.filtered_state
                U, V = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η, state.U, state.V
                η = sefs.η
                g = sefs.gravitational_acceleration

                # set!(η, f(x, y)) k² = ω²
                gu_c = 1
                gv_c = 2
                η₀(x, y, z) = sin(kx * x) * sin(ky * y) + 1
                set!(η, η₀)

                η_mean_before = mean(Array(interior(η)))

                U .= 0 # so that ∂ₜη(t=0) = 0
                V .= 0 # so that ∂ₜη(t=0) = 0
                η̅ .= 0
                U̅ .= 0
                V̅ .= 0
                GU .= gu_c
                GV .= gv_c

                weights = sefs.substepping.averaging_weights
                for i in 1:Nt
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, weights, Val(1))
                end
                iterate_split_explicit!(sefs, grid, GU, GV, Δτ_end, weights, Val(1))

                η_mean_after = mean(Array(interior(η)))

                tolerance = 10eps(FT)
                @test abs(η_mean_after - η_mean_before) < tolerance

                η_computed = Array(deepcopy(interior(η, :, 1, 1)))
                U_computed = Array(deepcopy(interior(U, :, 1, 1)))
                V_computed = Array(deepcopy(interior(V, :, 1, 1)))

                η̅_computed = Array(deepcopy(interior(η̅, :, 1, 1)))
                U̅_computed = Array(deepcopy(interior(U̅, :, 1, 1)))
                V̅_computed = Array(deepcopy(interior(V̅, :, 1, 1)))

                set!(η, η₀)

                # ∂ₜₜ(η) = Δη
                η_exact = cos(ω * T) * (Array(interior(η, :, 1, 1)) .- 1) .+ 1

                U₀(x, y) = kx * cos(kx * x) * sin(ky * y) # ∂ₜU = - ∂x(η), since we know η
                set!(U, U₀)
                U_exact = -(sin(ω * T) * 1 / ω) .* Array(interior(U, :, 1, 1)) .+ gu_c * T

                V₀(x, y) = ky * sin(kx * x) * cos(ky * y) # ∂ₜV = - ∂y(η), since we know η
                set!(V, V₀)
                V_exact = -(sin(ω * T) * 1 / ω) .* Array(interior(V, :, 1, 1)) .+ gv_c * T

                η̅_exact = (sin(ω * T) / ω - sin(ω * 0) / ω) / T * (Array(interior(η, :, 1, 1)) .- 1) .+ 1
                U̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(interior(U, :, 1, 1)) .+ gu_c * T / 2
                V̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(interior(V, :, 1, 1)) .+ gv_c * T / 2

                tolerance = 1e-2

                @test maximum(abs.(U_computed - U_exact)) / maximum(abs.(U_exact)) < tolerance
                @test maximum(abs.(V_computed - V_exact)) / maximum(abs.(V_exact)) < tolerance
                @test maximum(abs.(η_computed - η_exact)) / maximum(abs.(η_exact)) < tolerance

                @test maximum(abs.(U̅_computed - U̅_exact)) < tolerance
                @test maximum(abs.(V̅_computed - V̅_exact)) < tolerance
                @test maximum(abs.(η̅_computed - η̅_exact)) < tolerance
            end
        end # end of architecture loop
    end # end of float type loop
end # end of testset loop
