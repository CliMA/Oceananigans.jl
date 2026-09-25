include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using LinearAlgebra: Tridiagonal
using Oceananigans.Coriolis: ConstantCartesianCoriolis
using Oceananigans.Fields: interior
using Oceananigans.Grids: architecture
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid
using Oceananigans.Models.NonhydrostaticModels: step_velocities!
using Oceananigans.TimeSteppers: _ab2_step_field!, update_state!
using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity, VerticallyImplicitTimeDiscretization
using Oceananigans.Utils: launch!

@testset "Implicit velocity step next to an immersed bottom" begin
    Nz = 16
    kb = 5
    Δz = 1 / Nz

    for arch in archs, c in (1.0, 10.0)
        underlying_grid = RectilinearGrid(arch; size=(2, 2, Nz), x=(0, 1), y=(0, 1), z=(0, 1),
                                          topology=(Periodic, Periodic, Bounded))
        grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> (kb - 1) * Δz))
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
                launch!(architecture(grid), grid, :xyz, _ab2_step_field!,
                        u, Δt, χ, Gⁿ, G⁻; exclude_periphery=true)
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
