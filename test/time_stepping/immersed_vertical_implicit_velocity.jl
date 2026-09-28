include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using LinearAlgebra: Tridiagonal
using Oceananigans.Coriolis: ConstantCartesianCoriolis
using Oceananigans.Fields: interior
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid
using Oceananigans.Models.NonhydrostaticModels: step_velocities!, ab2_substep_velocity!, rk3_substep_velocity!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity, VerticallyImplicitTimeDiscretization

@testset "Implicit velocity step next to an immersed bottom" begin
    Nz = 16
    kb = 5
    Δz = 1 / Nz

    for arch in archs, c in (1.0, 10.0), active_cells_map in (false, true)
        underlying_grid = RectilinearGrid(arch; size=(2, 2, Nz), x=(0, 1), y=(0, 1), z=(0, 1),
                                          topology=(Periodic, Periodic, Bounded))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> (kb - 1) * Δz); active_cells_map)
        closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1)
        Δt = c * Δz^2

        cases = ((; forcing=(; w=(x, y, z, t) -> one(t))),
                 (; coriolis=ConstantCartesianCoriolis(f=1, rotation_axis=(0, sind(45), cosd(45)))))

        for options in cases
            model = NonhydrostaticModel(grid; closure, advection=nothing,
                                        timestepper=:QuasiAdamsBashforth2, options...)
            set!(model, u=1)
            update_state!(model)

            χ = model.timestepper.χ = convert(eltype(grid), -0.5)
            w★ = Ref{Any}(nothing)

            function substep_velocity!(u, Gⁿ, G⁻)
                ab2_substep_velocity!(u, grid, Δt, χ, Gⁿ, G⁻, model.timestepper.implicit_solver)
                u === model.velocities.w && (w★[] = Array(interior(u)))
                return nothing
            end

            step_velocities!(model, substep_velocity!, Δt, Val(keys(model.velocities)))

            n_active = Nz - kb
            implicit_matrix = Tridiagonal(fill(-c, n_active - 1),
                                           fill(1 + 2c, n_active),
                                           fill(-c, n_active - 1))
            expected = implicit_matrix \ vec(w★[][1, 1, kb+1:Nz])
            w_after_implicit_step = Array(interior(model.velocities.w))
            observed = vec(w_after_implicit_step[1, 1, kb+1:Nz])

            @test observed ≈ expected rtol=1e-12 atol=1e-14
            @test iszero(w_after_implicit_step[1, 1, kb])
            @test iszero(w_after_implicit_step[1, 1, end])
        end
    end
end

@testset "RK3 first stage ignores stale previous tendencies" begin
    Nz = 16
    kb = 5
    underlying_grid = RectilinearGrid(CPU(); size=(2, 2, Nz), x=(0, 1), y=(0, 1), z=(0, 1),
                                      topology=(Periodic, Periodic, Bounded))
    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> (kb - 1) / Nz);
                                active_cells_map=true)
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1)
    model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper=:RungeKutta3)
    w = model.velocities.w
    set!(w, 0)
    set!(model.timestepper.Gⁿ.w, 1)
    set!(model.timestepper.G⁻.w, NaN)
    w[1, 1, Nz+1] = 13

    @test isnan(model.timestepper.G⁻.w[1, 1, kb+2])

    γ¹ = model.timestepper.γ¹
    Δt = 0.01
    rk3_substep_velocity!(w, grid, Δt, γ¹, nothing, model.timestepper.Gⁿ.w,
                          model.timestepper.G⁻.w, model.timestepper.implicit_solver)

    @test w[1, 1, kb] == 0
    @test w[1, 1, kb+2] ≈ Δt * γ¹
    @test w[1, 1, Nz+1] == 13
end

@testset "AB2 and RK3 forcing does not cross an immersed bottom" begin
    Nx, Nz = 16, 16
    kb = 5
    Δz = 1 / Nz
    z_bottom = (kb - 1) * Δz
    for arch in archs
        underlying_grid = RectilinearGrid(arch; size=(Nx, Nz), x=(0, 1), z=(0, 1),
                                          topology=(Periodic, Flat, Bounded))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(x -> z_bottom))
        closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1)

        function velocities_after_step(timestepper, zero_immersed_forcing)
            w_forcing(x, z, t) = ifelse(zero_immersed_forcing && z <= z_bottom, zero(x), sinpi(2x))
            model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper,
                                        forcing=(; w=w_forcing))
            set!(model, u=0, w=0)
            time_step!(model, 10 * Δz^2)
            return map(field -> Array(interior(field)), values(model.velocities))
        end

        for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
            with_immersed_forcing = velocities_after_step(timestepper, false)
            zeroed_immersed_forcing = velocities_after_step(timestepper, true)

            for component in eachindex(with_immersed_forcing)
                wet_with = with_immersed_forcing[component][:, :, kb+1:Nz]
                wet_zeroed = zeroed_immersed_forcing[component][:, :, kb+1:Nz]
                @test wet_with ≈ wet_zeroed rtol=1e-10 atol=1e-12
            end

            w = with_immersed_forcing[3]
            @test all(iszero, w[:, :, kb])
            @test all(iszero, w[:, :, end])
        end
    end
end
