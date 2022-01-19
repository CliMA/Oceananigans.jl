using Test
using Statistics
using Oceananigans
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using Oceananigans.Operators
using KernelAbstractions
using Revise
archs = [Oceananigans.CPU(), Oceananigans.GPU()]

# include("dependencies_for_runtests.jl")
using Oceananigans.Models.HydrostaticFreeSurfaceModels
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitFreeSurface
import Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitAuxiliary, SplitExplicitSettings
import Oceananigans.Models.HydrostaticFreeSurfaceModels: barotropic_mode!, barotropic_split_explicit_corrector!, set_average_to_zero!

@testset "Barotropic Kernels" begin

    for arch in archs
        FT = Float64
        topology = (Periodic, Periodic, Bounded)
        Nx = Ny = 16 * 8
        Nz = 32
        Nx = 128
        Ny = 64
        Lx = Ly = Lz = 2π
        grid = RectilinearGrid(arch, topology = topology, size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (-Lz, 0))

        tmp = SplitExplicitFreeSurface()
        sefs = SplitExplicitState(grid)
        sefs = SplitExplicitAuxiliary(grid)
        sefs = SplitExplicitFreeSurface(grid)

        state = sefs.state
        auxiliary = sefs.auxiliary
        U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅
        Gᵁ, Gⱽ = auxiliary.Gᵁ, auxiliary.Gⱽ
        Hᶠᶜ, Hᶜᶠ = sefs.auxiliary.Hᶠᶜ, sefs.auxiliary.Hᶜᶠ

        u = Field{Face,Center,Center}(grid)
        v = Field{Center,Face,Center}(grid)

        @testset "Average to zero" begin
            # set equal to something else
            η̅ .= U̅ .= V̅ .= 1.0
            # now set equal to zero
            set_average_to_zero!(sefs.state)
            # don't forget the ghost points
            fill_halo_regions!(η̅, arch)
            fill_halo_regions!(U̅, arch)
            fill_halo_regions!(V̅, arch)
            # check
            @test all(Array(η̅.data.parent) .== 0.0)
            @test all(Array(U̅.data.parent .== 0.0))
            @test all(Array(V̅.data.parent .== 0.0))
        end

        @testset "Inexact integration" begin
            # Test 2: Check that vertical integrals work on the CPU(). The following should be "inexact"
            Δz = zeros(Nz)
            Δz .= grid.Δzᵃᵃᶠ

            set_u_check(x, y, z) = cos((π / 2) * z / Lz)
            set_U_check(x, y) = (sin(0) - (-2 * Lz / (π)))
            set!(u, set_u_check)
            exact_U = similar(U)
            set!(exact_U, set_U_check)
            barotropic_mode!(U, V, arch, grid, u, v)
            tolerance = 1e-3
            @test all((Array(interior(U) .- interior(exact_U))) .< tolerance)

            set_v_check(x, y, z) = sin(x * y) * cos((π / 2) * z / Lz)
            set_V_check(x, y) = sin(x * y) * (sin(0) - (-2 * Lz / (π)))
            set!(v, set_v_check)
            exact_V = similar(V)
            set!(exact_V, set_V_check)
            barotropic_mode!(U, V, arch, grid, u, v)
            @test all((Array(interior(V) .- interior(exact_V))) .< tolerance)
        end

        @testset "Vertical Integral " begin
            Δz = zeros(Nz)
            Δz .= grid.Δzᵃᵃᶜ

            u .= 0.0
            U .= 1.0
            barotropic_mode!(U, V, arch, grid, u, v)
            @test all(Array(U.data.parent) .== 0.0)

            u .= 1.0
            U .= 1.0
            barotropic_mode!(U, V, arch, grid, u, v)
            @test all(Array(interior(U)) .≈ Lz)

            set_u_check(x, y, z) = sin(x)
            set_U_check(x, y) = sin(x) * Lz
            set!(u, set_u_check)
            exact_U = similar(U)
            set!(exact_U, set_U_check)
            barotropic_mode!(U, V, arch, grid, u, v)
            @test all(Array(interior(U)) .≈ Array(interior(exact_U)))

            set_v_check(x, y, z) = sin(x) * z * cos(y)
            set_V_check(x, y) = -sin(x) * Lz^2 / 2.0 * cos(y)
            set!(v, set_v_check)
            exact_V = similar(V)
            set!(exact_V, set_V_check)
            barotropic_mode!(U, V, arch, grid, u, v)
            @test all(Array(interior(V)) .≈ Array(interior(exact_V)))
        end

        @testset "Barotropic Correction" begin
            # Test 4: Test Barotropic Correction
            FT = Float64
            topology = (Periodic, Periodic, Bounded)
            Nx = Ny = 16 * 8
            Nz = 32
            Nx = 128
            Ny = 64
            Lx = Ly = Lz = 2π
            grid = RectilinearGrid(arch, topology = topology, size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (-Lz, 0))

            sefs = SplitExplicitFreeSurface(grid)

            state = sefs.state
            auxiliary = sefs.auxiliary
            U, V, η̅, U̅, V̅ = state.U, state.V, state.η̅, state.U̅, state.V̅
            Gᵁ, Gⱽ = auxiliary.Gᵁ, auxiliary.Gⱽ
            Hᶠᶜ, Hᶜᶠ = sefs.auxiliary.Hᶠᶜ, sefs.auxiliary.Hᶜᶠ

            u = Field{Face,Center,Center}(grid)
            v = Field{Center,Face,Center}(grid)
            u_corrected = similar(u)
            v_corrected = similar(v)

            set_u(x, y, z) = z + Lz / 2 + sin(x)
            set_U̅(x, y) = cos(x) * Lz
            set_u_corrected(x, y, z) = z + Lz / 2 + cos(x)
            set!(u, set_u)
            set!(U̅, set_U̅)
            set!(u_corrected, set_u_corrected)

            set_v(x, y, z) = (z + Lz / 2) * sin(y) + sin(x)
            set_V̅(x, y) = (cos(x) + x) * Lz
            set_v_corrected(x, y, z) = (z + Lz / 2) * sin(y) + cos(x) + x
            set!(v, set_v)
            set!(V̅, set_V̅)
            set!(v_corrected, set_v_corrected)

            sefs.auxiliary.Hᶠᶜ .= Lz
            sefs.auxiliary.Hᶜᶠ .= Lz

            Δz = zeros(Nz)
            Δz .= grid.Δzᵃᵃᶜ

            barotropic_split_explicit_corrector!(u, v, sefs, grid)
            @test all(Array((interior(u) .- interior(u_corrected))) .< 1e-14)
            @test all(Array((interior(v) .- interior(v_corrected))) .< 1e-14)
        end
    end # end of architecture loop
end # end of testset
