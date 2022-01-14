include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface.jl") # CHANGE TO USING MODULE EVENTUALLY
include(pwd() * "/src/Models/HydrostaticFreeSurfaceModels/split_explicit_free_surface_kernels.jl")
# TODO: clean up test, change to use interior
# TODO: clean up substep function so that it only takes in SplitExplicitFreeSurface

using Test
using Revise
using Oceananigans
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using KernelAbstractions
using Oceananigans.Models.HydrostaticFreeSurfaceModels
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitFreeSurface
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitAuxiliary, SplitExplicitSettings, split_explicit_free_surface_substep!

@testset "Split-Explicit Dynamics" begin
    # @testset "Split Explicit Free Surface " begin
    arch = Oceananigans.CPU()
    FT = Float64
    topology = (Periodic, Periodic, Bounded)
    Nx = 128
    Ny = 64
    Nz = 16
    Lx = Ly = Lz = 2π
    grid = RectilinearGrid(topology = topology, size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (-Lz, 0))

    tmp = SplitExplicitFreeSurface()
    sefs = SplitExplicitState(grid, arch)
    sefs = SplitExplicitAuxiliary(grid, arch)
    sefs = SplitExplicitFreeSurface(grid, arch)

    sefs.Gᵁ
    sefs.η .= 0.0
    @test sefs.state.η === sefs.η
    @test sefs.auxiliary.Gᵁ === sefs.Gᵁ

    @testset " One timestep test " begin

        U, V, η̅, U̅, V̅, Gᵁ, Gⱽ = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
        Hᶠᶜ, Hᶜᶠ = sefs.Hᶠᶜ, sefs.Hᶜᶠ
        g = sefs.gravitational_acceleration
        Hᶠᶜ .= 1 / g
        Hᶜᶠ .= 1 / g
        η = sefs.η
        velocity_weight = 0.0
        free_surface_weight = 0.0
        Δτ = 1.0

        η₀(x, y) = sin(x)
        set!(η, η₀)
        U₀(x, y) = 0.0
        set!(U, U₀)
        V₀(x, y) = 0.0
        set!(V, V₀)

        η̅ .= 0.0
        U̅ .= 0.0
        V̅ .= 0.0
        Gᵁ .= 0.0
        Gⱽ .= 0.0

        # split_explicit_free_surface_step!(arch, grid, Δτ, sefs, 1)
        split_explicit_free_surface_substep!(sefs.state, sefs.auxiliary, sefs.settings, arch, grid, g, Δτ, 1)
        U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
        U_exact = (reshape(-cos.(grid.xᶠᵃᵃ), (length(grid.xᶜᵃᵃ), 1)).+reshape(0 * grid.yᵃᶜᵃ, (1, length(grid.yᵃᶜᵃ))))[2:Nx+1, 2:Ny+1]

        @test maximum(abs.(U_exact - U_computed)) < 1e-3

    end

    @testset "Multi-timestep test " begin
        U, V, η̅, U̅, V̅, Gᵁ, Gⱽ = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
        sefs.Hᶠᶜ .= 1 / sefs.parameters.g
        sefs.Hᶜᶠ .= 1 / sefs.parameters.g
        η = sefs.η
        velocity_weight = 0.0
        free_surface_weight = 0.0

        T = 2π
        Δτ = 2π / maximum([Nx, Ny]) * 5e-2 # the last factor is essentially the order of accuracy
        Nt = floor(Int, T / Δτ)
        Δτ_end = T - Nt * Δτ

        # set!(η, f(x,y))
        η₀(x, y) = sin(x)
        set!(η, η₀)
        U₀(x, y) = 0.0
        set!(U, U₀)
        V₀(x, y) = 0.0
        set!(V, V₀)

        η̅ .= 0.0
        U̅ .= 0.0
        V̅ .= 0.0
        Gᵁ .= 0.0
        Gⱽ .= 0.0

        for i in 1:Nt
            # free_surface_substep!(arch, grid, Δτ, sefs, 1)
            split_explicit_free_surface_substep!(sefs.state, sefs.auxiliary, sefs.settings, arch, grid, g, Δτ, 1)
        end
        # + correction for exact time
        free_surface_substep!(arch, grid, Δτ_end, sefs, 1)

        U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
        η_computed = Array(η.data.parent)[2:Nx+1, 2:Ny+1]
        set!(η, η₀)
        set!(U, U₀)
        U_exact = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
        η_exact = Array(η.data.parent)[2:Nx+1, 2:Ny+1]

        err1 = maximum(abs.(U_computed - U_exact))
        err2 = maximum(abs.(η_computed - η_exact))

        @test err1 < 1e-3
        @test err2 < 1e-6

    end

    @testset "Complex Multi-Timestep " begin
        # Test 3: Testing analytic solution to 
        # ∂ₜη + ∇⋅U⃗ = 0
        # ∂ₜU⃗ + ∇η  = G⃗
        kx = 2
        ky = 3
        ω = sqrt(kx^2 + ky^2)
        T = 2π / ω / 3 * 2
        Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # error mostly spatially dependent, except in the averaging
        Nt = floor(Int, T / Δτ)
        Δτ_end = T - Nt * Δτ


        sefs = SplitExplicitFreeSurface(grid, arch, settings = SplitExplicitSettings(Nt + 1))
        U, V, η̅, U̅, V̅, Gᵁ, Gⱽ = sefs.U, sefs.V, sefs.η̅, sefs.U̅, sefs.V̅, sefs.Gᵁ, sefs.Gⱽ
        η = sefs.η
        sefs.Hᶠᶜ .= 1 / sefs.parameters.g # to make life easy
        sefs.Hᶜᶠ .= 1 / sefs.parameters.g # to make life easy

        # set!(η, f(x,y)) k^2 = ω^2
        gu_c = 1.0
        gv_c = 2.0
        η₀(x, y) = sin(kx * x) * sin(ky * y)
        set!(η, η₀)

        U .= 0.0 # so that ∂ᵗη(t=0) = 0.0 
        V .= 0.0 # so that ∂ᵗη(t=0) = 0.0
        η̅ .= 0.0
        U̅ .= 0.0
        V̅ .= 0.0
        Gᵁ .= gu_c
        Gⱽ .= gv_c
        # overwrite weights
        sefs.velocity_weights .= ones(Nt + 1) ./ Nt        # since taking Nt+1 timesteps
        sefs.free_surface_weights .= ones(Nt + 1) ./ Nt  # since taking Nt+1 timesteps
        sefs.velocity_weights[Nt+1] = Δτ_end / T         # since last timestep is different
        sefs.free_surface_weights[Nt+1] = Δτ_end / T     # since last timestep is different

        for i in 1:Nt
            free_surface_substep!(arch, grid, Δτ, sefs, i)
        end
        # + correction for exact time
        free_surface_substep!(arch, grid, Δτ_end, sefs, Nt + 1)


        η_computed = Array(η.data.parent)[2:Nx+1, 2:Ny+1]
        U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
        V_computed = Array(V.data.parent)[2:Nx+1, 2:Ny+1]

        η̅_computed = Array(η̅.data.parent)[2:Nx+1, 2:Ny+1]
        U̅_computed = Array(U̅.data.parent)[2:Nx+1, 2:Ny+1]
        V̅_computed = Array(V̅.data.parent)[2:Nx+1, 2:Ny+1]

        set!(η, η₀)
        # ∂ₜₜ(η) = Δη
        η_exact = cos(ω * T) * Array(η.data.parent)[2:Nx+1, 2:Ny+1]

        U₀(x, y) = kx * cos(kx * x) * sin(ky * y) # ∂ₜU = - ∂x(η), since we know η
        set!(U, U₀)
        U_exact = -(sin(ω * T) * 1 / ω) .* Array(U.data.parent)[2:Nx+1, 2:Ny+1] .+ gu_c * T

        V₀(x, y) = ky * sin(kx * x) * cos(ky * y) # ∂ₜV = - ∂y(η), since we know η
        set!(V, V₀)
        V_exact = -(sin(ω * T) * 1 / ω) .* Array(V.data.parent)[2:Nx+1, 2:Ny+1] .+ gv_c * T

        η̅_exact = (sin(ω * T) / ω - sin(ω * 0) / ω) / T * Array(η.data.parent)[2:Nx+1, 2:Ny+1]
        U̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(U.data.parent)[2:Nx+1, 2:Ny+1] .+ gu_c * T / 2
        V̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(V.data.parent)[2:Nx+1, 2:Ny+1] .+ gv_c * T / 2

        errU = maximum(abs.(U_computed - U_exact)) / maximum(abs.(U_exact))
        errV = maximum(abs.(V_computed - V_exact)) / maximum(abs.(V_exact))
        errη = maximum(abs.(η_computed - η_exact)) / maximum(abs.(η_exact))

        errU̅ = maximum(abs.(U̅_computed - U̅_exact))
        errV̅ = maximum(abs.(V̅_computed - V̅_exact))
        errη̅ = maximum(abs.(η̅_computed - η̅_exact))

        @test errU < 1e-2
        @test errV < 1e-2
        @test errη < 2e-2

        @test errU̅ < 1e-2
        @test errV̅ < 1e-2
        @test errη̅ < 1e-2
    end

end