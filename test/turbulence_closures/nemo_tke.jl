include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, NEMOTKEVerticalDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: NEMOMixingLength, NEMOTKEEquation

# A 128 m column at 4 m, stirred by a kinematic stress of 1e-4 m² s⁻² (u★ = 0.01 m s⁻¹) on an f-plane
function nemo_column(arch; Jᵇ = 0)
    grid = RectilinearGrid(arch, size = 32, z = (-128, 0), topology = (Flat, Flat, Bounded))
    boundary_conditions = (b = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇ)),
                           u = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4)))
    return HydrostaticFreeSurfaceModel(grid; closure = NEMOTKEVerticalDiffusivity(), boundary_conditions,
                                       buoyancy = BuoyancyTracer(), tracers = :b, coriolis = FPlane(f = 1e-4))
end

@testset "NEMO's TKE scheme as a flavour of CATKE" begin
    @testset "Construction" begin
        closure = NEMOTKEVerticalDiffusivity()
        @test closure isa CATKEVerticalDiffusivity
        @test closure.mixing_length isa NEMOMixingLength
        @test closure.turbulent_kinetic_energy_equation isa NEMOTKEEquation
        @test closure.minimum_tke == 1e-6
        @test NEMOTKEVerticalDiffusivity(Float32).mixing_length.Cᵏ isa Float32
        @test occursin("NEMO", summary(closure))
    end

    for arch in archs
        @testset "Mixing length and diffusivities [$(typeof(arch))]" begin
            model = nemo_column(arch)
            N², e = 1e-5, 1e-4
            set!(model, b = z -> N² * z, e = e)
            update_state!(model)

            # √(2e)/N, limited by ℓ = 0.01 m at the bottom face and ℓ₀ = Cᴿ u★² at the surface
            ℓ₀ = 0.4 * 2e5 / 9.80665 * 1e-4
            zᶠ = znodes(model.grid, Face())
            expected = [min(sqrt(2e / N²), 0.01 + (z + 128), ℓ₀ - z) for z in zᶠ]
            expected[1], expected[end] = 0.01, ℓ₀
            ℓ = Array(interior(model.closure_fields.ℓ, 1, 1, :))
            @test ℓ ≈ expected rtol = 1e-10
            @test Array(interior(model.closure_fields.e₀))[1] ≈ 67.83e-4

            # Mid-column face: ν = 0.1 ℓ √e; no shear, so Ri = ∞ and the Prandtl number is 10
            k = 16
            κu = Array(interior(model.closure_fields.κu, 1, 1, :))
            κc = Array(interior(model.closure_fields.κc, 1, 1, :))
            @test κu[k] ≈ 0.1 * sqrt(2e / N²) * sqrt(e) rtol = 1e-10
            @test κc[k] ≈ 0.01 * sqrt(2e / N²) * sqrt(e) rtol = 1e-10
        end

        @testset "Fixed surface mixing length (NEMO 3.1-3.2, ln_mxl0 = false) [$(typeof(arch))]" begin
            mixing_length = NEMOMixingLength(Cᴿ = 0.0, ℓᵐⁱⁿ = 0.1, ℓ₀ᵐⁱⁿ = 0.4)
            grid = RectilinearGrid(arch, size = 32, z = (-128, 0), topology = (Flat, Flat, Bounded))
            u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4))
            model = HydrostaticFreeSurfaceModel(grid; closure = NEMOTKEVerticalDiffusivity(; mixing_length),
                                                boundary_conditions = (; u = u_bcs), buoyancy = BuoyancyTracer(),
                                                tracers = :b, coriolis = FPlane(f = 1e-4))
            N², e = 1e-5, 1e-4
            set!(model, b = z -> N² * z, e = e)
            update_state!(model)
            zᶠ = znodes(model.grid, Face())
            expected = [min(sqrt(2e / N²), 0.1 + (z + 128), 0.4 - z) for z in zᶠ]
            expected[1], expected[end] = 0.1, 0.4
            @test Array(interior(model.closure_fields.ℓ, 1, 1, :)) ≈ expected rtol = 1e-10
        end

        @testset "Enhanced vertical diffusion where N² < 0 [$(typeof(arch))]" begin
            model = nemo_column(arch)
            set!(model, b = z -> - 1e-5 * z, e = 1e-4)
            update_state!(model)
            κc = Array(interior(model.closure_fields.κc, 1, 1, :))
            @test all(κc[2:end-1] .== 100)
        end

        @testset "Time stepping [$(typeof(arch))]" begin
            model = nemo_column(arch; Jᵇ = 1e-8)
            set!(model, b = z -> 1e-5 * z, e = 1e-6)
            for _ in 1:36
                time_step!(model, 600)
            end
            e = Array(interior(model.tracers.e, 1, 1, :))
            b = Array(interior(model.tracers.b, 1, 1, :))
            @test all(isfinite, e)
            @test minimum(e) ≥ 1e-6
            @test e[end] > 1e-4                        # the surface value 67.83 u★² reaches the top cell
            @test (b[end] - b[end-1]) / 4 < 0.5e-5     # the top is mixed
        end

        # With an active-cells map the diffusivity kernel is launched over the map (a list of (i, j, k)), and the
        # column kernels must still cover every wet column: same ℓ and κ as without the map
        @testset "Immersed grid with an active-cells map [$(typeof(arch))]" begin
            underlying_grid = RectilinearGrid(arch, size = (4, 4, 32), x = (0, 4e3), y = (0, 4e3), z = (-128, 0),
                                              topology = (Periodic, Bounded, Bounded))
            land_strip(x, y) = x < 1e3 ? 0.0 : -128.0   # i = 1 is land, i = 2:4 full-depth water
            u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4))
            ℓs, κs = [], []
            for active_cells_map in (false, true)
                grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(land_strip); active_cells_map)
                model = HydrostaticFreeSurfaceModel(grid; closure = NEMOTKEVerticalDiffusivity(),
                                                    boundary_conditions = (; u = u_bcs), buoyancy = BuoyancyTracer(),
                                                    tracers = :b, coriolis = FPlane(f = 1e-4))
                set!(model, b = (x, y, z) -> 1e-5 * z, e = 1e-4)
                update_state!(model)
                push!(ℓs, Array(interior(model.closure_fields.ℓ)))
                push!(κs, Array(interior(model.closure_fields.κc)))
            end
            @test ℓs[2][2:4, :, :] ≈ ℓs[1][2:4, :, :] rtol = 1e-10
            @test κs[2][2:4, :, :] ≈ κs[1][2:4, :, :] rtol = 1e-10
        end
    end
end
