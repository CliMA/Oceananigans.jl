#=
Investigation: FFT Compilation / Differentiation (B.6.1)
Status: IN PROGRESS (as of 2026-02-02)
Purpose: Medium WE for FFT gradient computation - UPSTREAM test location
Fix Location: ext/OceananigansReactantExt/Solvers.jl
Related: cursor-toolchain/rules/domains/differentiability/investigations/fft-compilation.md
Synchronized with: Manteia.jl/test/fft-ad/ (downstream)
=#

using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Fields: interior, set!
using CUDA
using Reactant
using Enzyme
using Statistics: mean
using Test

Reactant.set_default_backend("cpu")

grid = RectilinearGrid(ReactantState(),
            size=(4, 4, 4), extent=(100, 100, 100),
            halo=(3, 3, 3), topology=(Periodic, Periodic, Periodic))

        model = NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)

@testset "FFT AD - NonhydrostaticModel" begin

    @testset "Periodic topology (FFT only)" begin
        grid = RectilinearGrid(ReactantState(),
            size=(4, 4, 4), extent=(100, 100, 100),
            halo=(3, 3, 3), topology=(Periodic, Periodic, Periodic))

        model = NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)
        @test model !== nothing
        @test model.pressure_solver !== nothing

        dmodel = Enzyme.make_zero(model)
        T_init = CenterField(grid)
        set!(T_init, (x, y, z) -> 20.0 + 0.01 * x + 0.01 * y + 0.01 * z)
        dT_init = CenterField(grid)
        set!(dT_init, 0.0)

        function loss(model, T_init, Δt)
            set!(model, T=T_init)
            @trace mincut=true checkpointing=true track_numbers=false for i in 1:2
                time_step!(model, Δt)
            end
            return mean(interior(model.tracers.T).^2)
        end

        function grad_loss(model, dmodel, T_init, dT_init, Δt)
            parent(dT_init) .= 0
            _, loss_value = Enzyme.autodiff(
                Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                loss, Enzyme.Active,
                Enzyme.Duplicated(model, dmodel),
                Enzyme.Duplicated(T_init, dT_init),
                Enzyme.Const(Δt))
            return dT_init, loss_value
        end

        Δt = 0.001
        compiled = Reactant.@compile raise_first=true raise=true sync=true grad_loss(
            model, dmodel, T_init, dT_init, Δt)
        @test compiled !== nothing

        dT, loss_val = compiled(model, dmodel, T_init, dT_init, Δt)
        @test loss_val > 0
        @test !isnan(loss_val)
        @test maximum(abs, interior(dT)) > 0
        @test !any(isnan, interior(dT))
    end

    @testset "Bounded z-topology (DCT required) - broken" begin
        grid = RectilinearGrid(ReactantState(),
            size=(4, 4, 4), extent=(100, 100, 100),
            halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))

        # Bounded topology requires DCT which is not yet supported
        @test_throws ErrorException begin
            NonhydrostaticModel(grid; tracers=:T, buoyancy=nothing, closure=nothing)
        end
    end
end
