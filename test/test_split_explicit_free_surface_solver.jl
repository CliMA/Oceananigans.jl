include("dependencies_for_runtests.jl")
using Oceananigans.Models.HydrostaticFreeSurfaceModels
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitFreeSurface
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitAuxiliaryFields, SplitExplicitSettings, iterate_split_explicit!

using Oceananigans.Models.HydrostaticFreeSurfaceModels: constant_averaging_kernel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: calculate_substeps, calculate_adaptive_settings

@testset "Split-Explicit Dynamics" begin

    for FT in float_types
        for arch in archs
            topology = (Periodic, Periodic, Bounded)

            Nx, Ny, Nz = 128, 64, 1
            Lx = Ly = 2π
            Lz = 1 / Oceananigans.BuoyancyModels.g_Earth

            grid = RectilinearGrid(arch, FT;
                                   topology, size = (Nx, Ny, Nz),
                                   x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                                   halo=(1, 1, 1))

            settings = SplitExplicitSettings(eltype(grid); substeps = 200, averaging_kernel = constant_averaging_kernel)
            sefs = SplitExplicitFreeSurface(grid; settings)

            sefs.η .= 0

            @testset " One timestep test " begin
                state = sefs.state
                U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅

                η = sefs.η
                Δτ = 1.0

                η₀(x, y, z) = sin(x)
                set!(η, η₀)
            
                Nsubsteps = calculate_substeps(settings.substepping, 1)
                fractional_Δt, weights = calculate_adaptive_settings(settings.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                iterate_split_explicit!(sefs, grid, Δτ, weights, Val(1)) 

                U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
                U_exact = (reshape(-cos.(grid.xᶠᵃᵃ), (length(grid.xᶜᵃᵃ), 1)).+reshape(0 * grid.yᵃᶜᵃ, (1, length(grid.yᵃᶜᵃ))))[2:Nx+1, 2:Ny+1]

                @test maximum(abs.(U_exact - U_computed)) < 1e-3
            end

            @testset "Multi-timestep test " begin
                state = sefs.state
                auxiliary = sefs.auxiliary
                U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅
                Gᵁ, Gⱽ = auxiliary.Gᵁ, auxiliary.Gⱽ
                η = sefs.η

                T  = 2π
                Δτ = 2π / maximum([Nx, Ny]) * 5e-1 # the last factor is essentially the order of accuracy
                Nt = floor(Int, T / Δτ)
                Δτ_end = T - Nt * Δτ

                settings = SplitExplicitSettings(eltype(grid); substeps = Nt, averaging_kernel = constant_averaging_kernel)

                # set!(η, f(x,y))
                η₀(x, y, z) = sin(x)
                set!(η, η₀)
                U₀(x, y, z) = 0
                set!(U, U₀)
                V₀(x, y, z) = 0
                set!(V, V₀)

                η̅  .= 0
                U̅  .= 0
                V̅  .= 0
                Gᵁ .= 0
                Gⱽ .= 0

                Nsubsteps  = calculate_substeps(settings.substepping, 1)
                fractional_Δt, weights = calculate_adaptive_settings(settings.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                iterate_split_explicit!(sefs, grid, Δτ, weights, Val(Nsubsteps)) 
    
                U_computed = Array(parent(U))[2:Nx+1, 2:Ny+1]
                η_computed = Array(parent(η))[2:Nx+1, 2:Ny+1]
                set!(η, η₀)
                set!(U, U₀)
                U_exact = Array(parent(U))[2:Nx+1, 2:Ny+1]
                η_exact = Array(parent(η))[2:Nx+1, 2:Ny+1]

                @test maximum(abs.(U_computed - U_exact)) < 1e-3
                @show maximum(abs.(η_computed))
                @show maximum(abs.(η_exact))
                @test maximum(abs.(η_computed - η_exact)) < max(100eps(FT), 1e-6)
            end

            @testset "Averaging / Do Nothing test " begin
                state = sefs.state
                auxiliary = sefs.auxiliary
                U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅
                Gᵁ, Gⱽ = auxiliary.Gᵁ, auxiliary.Gⱽ

                g = sefs.gravitational_acceleration
                η = sefs.η

                Δτ = 2π / maximum([Nx, Ny]) * 5e-1 # the last factor is essentially the order of accuracy

                # set!(η, f(x, y))
                η_avg = 1
                U_avg = 2
                V_avg = 3
                η₀(x, y, z) = η_avg
                set!(η, η₀)
                U₀(x, y, z) = U_avg
                set!(U, U₀)
                V₀(x, y, z) = V_avg
                set!(V, V₀)

                η̅ .= 0
                U̅ .= 0
                V̅ .= 0
                Gᵁ .= 0
                Gⱽ .= 0
                settings = sefs.settings

                Nsubsteps  = calculate_substeps(settings.substepping, 1)
                fractional_Δt, weights = calculate_adaptive_settings(settings.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights
    
                iterate_split_explicit!(sefs, grid, Δτ, weights, Val(Nsubsteps)) 

                U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
                V_computed = Array(V.data.parent)[2:Nx+1, 2:Ny+1]
                η_computed = Array(η.data.parent)[2:Nx+1, 2:Ny+1]

                U̅_computed = Array(U̅.data.parent)[2:Nx+1, 2:Ny+1]
                V̅_computed = Array(V̅.data.parent)[2:Nx+1, 2:Ny+1]
                η̅_computed = Array(η̅.data.parent)[2:Nx+1, 2:Ny+1]

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

                sefs = SplitExplicitFreeSurface(grid)
                state = sefs.state
                auxiliary = sefs.auxiliary
                U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅
                Gᵁ, Gⱽ = auxiliary.Gᵁ, auxiliary.Gⱽ
                η = sefs.η
                g = sefs.gravitational_acceleration

                # set!(η, f(x,y)) k² = ω²
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
                Gᵁ .= gu_c
                Gⱽ .= gv_c

                settings = SplitExplicitSettings(eltype(grid); substeps = Nt + 1, averaging_kernel = constant_averaging_kernel)
                sefs = sefs(settings)

                weights = settings.substepping.averaging_weights
                for i in 1:Nt
                    iterate_split_explicit!(sefs, grid, Δτ, weights, Val(1)) 
                end
                iterate_split_explicit!(sefs, grid, Δτ_end, weights, Val(1)) 

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

                U₀(x, y, z) = kx * cos(kx * x) * sin(ky * y) # ∂ₜU = - ∂x(η), since we know η
                set!(U, U₀)
                U_exact = -(sin(ω * T) * 1 / ω) .* Array(interior(U, :, 1, 1)) .+ gu_c * T

                V₀(x, y, z) = ky * sin(kx * x) * cos(ky * y) # ∂ₜV = - ∂y(η), since we know η
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