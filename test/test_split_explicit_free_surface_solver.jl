include("dependencies_for_runtests.jl")
using Oceananigans.Models.HydrostaticFreeSurfaceModels
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitFreeSurface
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitAuxiliaryFields, SplitExplicitSettings, split_explicit_free_surface_substep!

using Oceananigans.Models.HydrostaticFreeSurfaceModels: constant_averaging_kernel

@testset "Split-Explicit Dynamics" begin

    for FT in float_types
        for arch in archs
            topology = (Periodic, Periodic, Bounded)

            Nx, Ny, Nz = 128, 64, 16
            Lx = Ly = Lz = 2π

            grid = RectilinearGrid(arch, FT;
                                   topology, size = (Nx, Ny, Nz),
                                   x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                                   halo=(1, 1, 1))

            settings = SplitExplicitSettings(; substeps = 200, barotropic_averaging_kernel = constant_averaging_kernel)
            sefs = SplitExplicitFreeSurface(grid; settings)

            sefs.η .= 0

            @testset " One timestep test " begin
                state = sefs.state
                auxiliary = sefs.auxiliary
                U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅
                Gᵁ, Gⱽ = auxiliary.Gᵁ, auxiliary.Gⱽ
                Hᶠᶜ, Hᶜᶠ = sefs.auxiliary.Hᶠᶜ, sefs.auxiliary.Hᶜᶠ
                g = sefs.gravitational_acceleration

                Hᶠᶜ .= 1 / g
                Hᶜᶠ .= 1 / g
                η = sefs.η
                velocity_weight = 0.0
                free_surface_weight = 0.0
                Δτ = 1.0

                η₀(x, y, z) = sin(x)
                set!(η, η₀)
                U₀(x, y) = 0
                set!(U, U₀)
                V₀(x, y) = 0
                set!(V, V₀)

                η̅ .= 0
                U̅ .= 0
                V̅ .= 0
                Gᵁ .= 0
                Gⱽ .= 0

                split_explicit_free_surface_substep!(η, sefs.state, sefs.auxiliary, sefs.settings, arch, grid, g, Δτ, 1)
                U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
                U_exact = (reshape(-cos.(grid.xᶠᵃᵃ), (length(grid.xᶜᵃᵃ), 1)).+reshape(0 * grid.yᵃᶜᵃ, (1, length(grid.yᵃᶜᵃ))))[2:Nx+1, 2:Ny+1]

                @test maximum(abs.(U_exact - U_computed)) < 1e-3
            end

            @testset "Multi-timestep test " begin
                state = sefs.state
                auxiliary = sefs.auxiliary
                U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅
                Gᵁ, Gⱽ = auxiliary.Gᵁ, auxiliary.Gⱽ
                g = sefs.gravitational_acceleration
                sefs.auxiliary.Hᶠᶜ .= 1 / g
                sefs.auxiliary.Hᶜᶠ .= 1 / g
                η = sefs.η
                velocity_weight = 0.0
                free_surface_weight = 0.0

                T = 2π
                Δτ = 2π / maximum([Nx, Ny]) * 5e-2 # the last factor is essentially the order of accuracy
                Nt = floor(Int, T / Δτ)
                Δτ_end = T - Nt * Δτ

                settings = SplitExplicitSettings(; substeps = Nt, barotropic_averaging_kernel = constant_averaging_kernel)

                # set!(η, f(x,y))
                η₀(x, y, z) = sin(x)
                set!(η, η₀)
                U₀(x, y) = 0
                set!(U, U₀)
                V₀(x, y) = 0
                set!(V, V₀)

                η̅ .= 0
                U̅ .= 0
                V̅ .= 0
                Gᵁ .= 0
                Gⱽ .= 0

                for i in 1:Nt
                    split_explicit_free_surface_substep!(η, sefs.state, sefs.auxiliary, settings, arch, grid, g, Δτ, i)
                end

                # + correction for exact time
                split_explicit_free_surface_substep!(η, sefs.state, sefs.auxiliary, settings, arch, grid, g, Δτ_end, 1)

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
                sefs.auxiliary.Hᶠᶜ .= 1 / g
                sefs.auxiliary.Hᶜᶠ .= 1 / g
                η = sefs.η
                velocity_weight = 0.0
                free_surface_weight = 0.0

                Δτ = 2π / maximum([Nx, Ny]) * 5e-2 # the last factor is essentially the order of accuracy

                # set!(η, f(x, y))
                η_avg = 1
                U_avg = 2
                V_avg = 3
                η₀(x, y, z) = η_avg
                set!(η, η₀)
                U₀(x, y) = U_avg
                set!(U, U₀)
                V₀(x, y) = V_avg
                set!(V, V₀)

                η̅ .= 0
                U̅ .= 0
                V̅ .= 0
                Gᵁ .= 0
                Gⱽ .= 0
                settings = sefs.settings

                for i in 1:settings.substeps
                    split_explicit_free_surface_substep!(η, sefs.state, sefs.auxiliary, sefs.settings, arch, grid, g, Δτ, i)
                end

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
                sefs.auxiliary.Hᶠᶜ .= 1 / g # to make life easy
                sefs.auxiliary.Hᶜᶠ .= 1 / g # to make life easy

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

                settings = SplitExplicitSettings(substeps = Nt + 1, barotropic_averaging_kernel = constant_averaging_kernel)
                sefs = sefs(settings)

                for i in 1:Nt
                    split_explicit_free_surface_substep!(η, sefs.state, sefs.auxiliary, sefs.settings, arch, grid, g, Δτ, i)
                end
                # + correction for exact time
                split_explicit_free_surface_substep!(η, sefs.state, sefs.auxiliary, sefs.settings, arch, grid, g, Δτ_end, Nt + 1)

                η_mean_after = mean(Array(interior(η)))

                tolerance = 10eps(FT)
                @test abs(η_mean_after - η_mean_before) < tolerance

                η_computed = Array(η.data.parent)[2:Nx+1, 2:Ny+1]
                U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
                V_computed = Array(V.data.parent)[2:Nx+1, 2:Ny+1]

                η̅_computed = Array(η̅.data.parent)[2:Nx+1, 2:Ny+1]
                U̅_computed = Array(U̅.data.parent)[2:Nx+1, 2:Ny+1]
                V̅_computed = Array(V̅.data.parent)[2:Nx+1, 2:Ny+1]

                set!(η, η₀)

                # ∂ₜₜ(η) = Δη
                η_exact = cos(ω * T) * (Array(η.data.parent)[2:Nx+1, 2:Ny+1] .- 1) .+ 1

                U₀(x, y) = kx * cos(kx * x) * sin(ky * y) # ∂ₜU = - ∂x(η), since we know η
                set!(U, U₀)
                U_exact = -(sin(ω * T) * 1 / ω) .* Array(U.data.parent)[2:Nx+1, 2:Ny+1] .+ gu_c * T

                V₀(x, y) = ky * sin(kx * x) * cos(ky * y) # ∂ₜV = - ∂y(η), since we know η
                set!(V, V₀)
                V_exact = -(sin(ω * T) * 1 / ω) .* Array(V.data.parent)[2:Nx+1, 2:Ny+1] .+ gv_c * T

                η̅_exact = (sin(ω * T) / ω - sin(ω * 0) / ω) / T * (Array(η.data.parent)[2:Nx+1, 2:Ny+1] .- 1) .+ 1
                U̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(U.data.parent)[2:Nx+1, 2:Ny+1] .+ gu_c * T / 2
                V̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(V.data.parent)[2:Nx+1, 2:Ny+1] .+ gv_c * T / 2

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