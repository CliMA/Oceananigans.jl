include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using LinearAlgebra: Tridiagonal
using Oceananigans.Coriolis: ConstantCartesianCoriolis
using Oceananigans.Fields: interior
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid
using Oceananigans.Models.NonhydrostaticModels: step_velocities!, ab2_substep_velocity!, rk3_substep_velocity!
using Oceananigans.TimeSteppers: update_state!, stage_Δt
using Oceananigans.Utils: kernel_time_step
using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity, VerticallyImplicitTimeDiscretization

@testset "Implicit velocity predictors on ordinary grids" begin
    for FT in (Float32, Float64)
        grid = RectilinearGrid(CPU(), FT; size=(4, 3, 4), x=(0, 1), y=(0, 1), z=(0, 1))
        closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=FT(1))
        Δt = FT(0.01)

        for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
            model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper)
            χ = FT(0.25)
            γ = FT(0.6)
            ζ = FT(-0.2)

            for name in propertynames(model.velocities)
                velocity = getproperty(model.velocities, name)
                Gⁿ = getproperty(model.timestepper.Gⁿ, name)
                G⁻ = getproperty(model.timestepper.G⁻, name)
                initial_value = name === :w ? FT(2) : FT(1)
                current_tendency = name === :v ? FT(3) : FT(2)
                previous_tendency = name === :u ? FT(-1) : FT(1)
                set!(velocity, initial_value)
                set!(Gⁿ, current_tendency)
                set!(G⁻, timestepper === :RungeKutta3 ? FT(NaN) : previous_tendency)

                initial = Array(interior(velocity))
                current = Array(interior(Gⁿ))
                previous = Array(interior(G⁻))

                if timestepper === :QuasiAdamsBashforth2
                    ab2_substep_velocity!(velocity, grid, Δt, χ, Gⁿ, G⁻, model.timestepper.implicit_solver)
                    α = FT(3/2) + χ
                    β = FT(1/2) + χ
                    expected = initial .+ Δt .* (α .* current .- β .* previous)
                else
                    rk3_substep_velocity!(velocity, grid, Δt, γ, nothing, Gⁿ, G⁻, model.timestepper.implicit_solver)
                    expected = initial .+ Δt .* γ .* current
                    if name === :w
                        expected[:, :, 1] .= initial[:, :, 1]
                        expected[:, :, end] .= initial[:, :, end]
                    end
                    @test Array(interior(velocity)) ≈ expected
                    set!(velocity, initial_value)
                    set!(G⁻, previous_tendency)
                    initial = Array(interior(velocity))
                    previous = Array(interior(G⁻))
                    rk3_substep_velocity!(velocity, grid, Δt, γ, ζ, Gⁿ, G⁻, model.timestepper.implicit_solver)
                    expected_later = initial .+ Δt .* (γ .* current .+ ζ .* previous)
                    if name === :w
                        expected_later[:, :, 1] .= initial[:, :, 1]
                        expected_later[:, :, end] .= initial[:, :, end]
                    end
                    @test Array(interior(velocity)) ≈ expected_later
                    continue
                end

                if name === :w
                    expected[:, :, 1] .= initial[:, :, 1]
                    expected[:, :, end] .= initial[:, :, end]
                end
                @test Array(interior(velocity)) ≈ expected
            end

            integrated_model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper)
            set!(integrated_model, u=1, w=1)
            time_step!(integrated_model, Δt)
            for velocity in values(integrated_model.velocities)
                @test all(isfinite, Array(interior(velocity)))
            end
            w = Array(interior(integrated_model.velocities.w))
            @test all(iszero, w[:, :, 1])
            @test all(iszero, w[:, :, end])
        end
    end
end

@testset "Mixed-precision implicit predictors match shared kernels" begin
    FT = Float32
    grid = RectilinearGrid(CPU(), FT; size=(4, 3, 4), x=(0, 1), y=(0, 1), z=(0, 1))
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=FT(1))
    Δt = 0.1
    expected_value = FT(-FT(0.1) + Δt * one(FT))
    converted_value = -FT(0.1) + FT(Δt) * one(FT)
    @test expected_value != converted_value
    @test !iszero(expected_value)

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        implicit_model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper)
        shared_model = NonhydrostaticModel(grid; advection=nothing, timestepper)

        for name in propertynames(implicit_model.velocities)
            implicit_velocity = getproperty(implicit_model.velocities, name)
            shared_velocity = getproperty(shared_model.velocities, name)
            implicit_Gⁿ = getproperty(implicit_model.timestepper.Gⁿ, name)
            shared_Gⁿ = getproperty(shared_model.timestepper.Gⁿ, name)
            implicit_G⁻ = getproperty(implicit_model.timestepper.G⁻, name)
            shared_G⁻ = getproperty(shared_model.timestepper.G⁻, name)

            for velocity in (implicit_velocity, shared_velocity)
                set!(velocity, -FT(0.1))
            end
            for tendency in (implicit_Gⁿ, shared_Gⁿ)
                set!(tendency, one(FT))
            end
            for tendency in (implicit_G⁻, shared_G⁻)
                set!(tendency, zero(FT))
            end

            if timestepper === :QuasiAdamsBashforth2
                χ = -FT(0.5)
                ab2_substep_velocity!(implicit_velocity, grid, Δt, χ, implicit_Gⁿ, implicit_G⁻,
                                      implicit_model.timestepper.implicit_solver)
                ab2_substep_velocity!(shared_velocity, grid, Δt, χ, shared_Gⁿ, shared_G⁻, nothing)
                implicit_values = Array(interior(implicit_velocity))
                shared_values = Array(interior(shared_velocity))
                @test implicit_values == shared_values
            else
                γ = one(FT)
                rk3_substep_velocity!(implicit_velocity, grid, Δt, γ, nothing, implicit_Gⁿ, implicit_G⁻,
                                      implicit_model.timestepper.implicit_solver)
                rk3_substep_velocity!(shared_velocity, grid, Δt, γ, nothing, shared_Gⁿ, shared_G⁻, nothing)
                implicit_values = Array(interior(implicit_velocity))
                shared_values = Array(interior(shared_velocity))
                @test implicit_values == shared_values

                set!(implicit_velocity, -FT(0.1))
                set!(shared_velocity, -FT(0.1))
                rk3_substep_velocity!(implicit_velocity, grid, Δt, γ, zero(FT), implicit_Gⁿ, implicit_G⁻,
                                      implicit_model.timestepper.implicit_solver)
                rk3_substep_velocity!(shared_velocity, grid, Δt, γ, zero(FT), shared_Gⁿ, shared_G⁻, nothing)
                implicit_values = Array(interior(implicit_velocity))
                shared_values = Array(interior(shared_velocity))
                @test implicit_values == shared_values
            end

            active_index = name === :w ? (1, 1, 2) : (1, 1, 1)
            @test shared_values[active_index...] == expected_value

            if timestepper === :RungeKutta3
                ts = implicit_model.timestepper
                for (γ, ζ) in ((ts.γ¹, nothing), (ts.γ², ts.ζ²), (ts.γ³, ts.ζ³))
                    for velocity in (implicit_velocity, shared_velocity)
                        set!(velocity, FT(2))
                    end
                    for tendency in (implicit_G⁻, shared_G⁻)
                        set!(tendency, ζ === nothing ? FT(NaN) : FT(-2))
                    end
                    rk3_substep_velocity!(implicit_velocity, grid, Δt, γ, ζ, implicit_Gⁿ, implicit_G⁻,
                                          implicit_model.timestepper.implicit_solver)
                    rk3_substep_velocity!(shared_velocity, grid, Δt, γ, ζ, shared_Gⁿ, shared_G⁻, nothing)
                    @test Array(interior(implicit_velocity)) == Array(interior(shared_velocity))
                end
            end
        end
    end
end

@testset "Ordinary-grid implicit solve matches a tridiagonal reference" begin
    Nz = 4
    grid = RectilinearGrid(CPU(); size=(2, 2, Nz), x=(0, 1), y=(0, 1), z=(0, 1))
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1)
    Δt = 0.01

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper)
        ts = model.timestepper
        stages = timestepper === :QuasiAdamsBashforth2 ? ((nothing, nothing),) :
                 ((ts.γ¹, nothing), (ts.γ², ts.ζ²), (ts.γ³, ts.ζ³))

        for (γ, ζ) in stages
            set!(model, w=0)
            update_state!(model)
            set!(ts.Gⁿ.w, 1)
            set!(ts.G⁻.w, ζ === nothing && γ !== nothing ? NaN : 0.25)
            w★ = Ref{Any}(nothing)

            function substep_velocity!(u, Gⁿ, G⁻)
                if timestepper === :QuasiAdamsBashforth2
                    ab2_substep_velocity!(u, grid, Δt, ts.χ, Gⁿ, G⁻, ts.implicit_solver)
                else
                    rk3_substep_velocity!(u, grid, Δt, γ, ζ, Gⁿ, G⁻, ts.implicit_solver)
                end
                u === model.velocities.w && (w★[] = Array(interior(u)))
                return nothing
            end

            implicit_Δt = γ === nothing ? Δt : stage_Δt(Δt, γ, ζ)
            step_velocities!(model, substep_velocity!, implicit_Δt)
            c = implicit_Δt * Nz^2
            matrix = Tridiagonal(fill(-c, Nz - 2), fill(1 + 2c, Nz - 1), fill(-c, Nz - 2))
            expected = matrix \ vec(w★[][1, 1, 2:Nz])
            observed = Array(interior(model.velocities.w))
            @test any(!iszero, expected)
            @test observed[1, 1, 2:Nz] ≈ expected rtol=1e-12 atol=1e-14
            @test all(iszero, observed[:, :, 1])
            @test all(iszero, observed[:, :, end])
        end
    end
end

@testset "CPU public stepping normalizes mixed-precision timesteps" begin
    grid = RectilinearGrid(CPU(), Float32; size=(4, 3, 4), x=(0, 1), y=(0, 1), z=(0, 1))
    closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=1f0)
    Δt = 0.1
    normalized_Δt = kernel_time_step(CPU(), grid, Δt)
    @test normalized_Δt isa Float32
    @test normalized_Δt == Float32(Δt)

    for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3)
        mixed_model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper,
                                          forcing=(; u=(x, y, z, t) -> 1f0))
        normalized_model = NonhydrostaticModel(grid; closure, advection=nothing, timestepper,
                                               forcing=(; u=(x, y, z, t) -> 1f0))
        for model in (mixed_model, normalized_model)
            set!(model, u=-0.1f0, v=0.25f0)
        end
        time_step!(mixed_model, Δt)
        time_step!(normalized_model, normalized_Δt)
        for name in propertynames(mixed_model.velocities)
            mixed_velocity = getproperty(mixed_model.velocities, name)
            normalized_velocity = getproperty(normalized_model.velocities, name)
            @test Array(interior(mixed_velocity)) == Array(interior(normalized_velocity))
        end
    end
end

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

            step_velocities!(model, substep_velocity!, Δt)

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
