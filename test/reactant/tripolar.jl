include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))
include(joinpath(@__DIR__, "..", "setup", "reactant_correctness_utils.jl"))

using Reactant: @trace
using Enzyme
using CUDA
using Random
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.BoundaryConditions: UPivot, FPivot
using Oceananigans.TimeSteppers: update_state!, first_time_step!
using Oceananigans.Models: zero_field!

#####
##### TripolarGrid + HydrostaticFreeSurfaceModel compiled with Reactant: forward vs CPU and gradient vs finite differences
#####

# Land over the two north singularities and the southernmost row, where the metrics degenerate
function masked_tripolar_grid(arch; fold_topology, pivot)
    grid = TripolarGrid(arch; size=(16, 12, 1), halo=(4, 4, 2), z=(-4000, 0), fold_topology, pivot)
    near(λ, φ, λ₀) = (abs(mod(λ - λ₀ + 180, 360) - 180) < 15) & (abs(φ - 55) < 15)
    bottom(λ, φ) = near(λ, φ, 70) | near(λ, φ, 250) | (φ < -70) ? 0.0 : -4000 + 1000 * cosd(λ) * cosd(φ)
    return ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))
end

tide_model(grid) = HydrostaticFreeSurfaceModel(grid; free_surface = ExplicitFreeSurface(gravitational_acceleration=10),
                                                     momentum_advection = nothing, tracers = (), buoyancy = nothing,
                                                     coriolis = HydrostaticSphericalCoriolis(),
                                                     timestepper = :SplitRungeKutta3)

function run_timesteps!(model, Δt, Nt)
    @trace track_numbers=false for _ in 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

# Every evaluation starts from the same state, including halos and the time-stepper caches
function free_surface_loss(model, η₀, Δt, Nt)
    foreach(zero_field!, fields(model))
    foreach(zero_field!, model.timestepper.Ψ⁻)
    foreach(zero_field!, model.timestepper.Gⁿ)
    set!(model.free_surface.displacement, η₀)
    update_state!(model)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:Nt
        time_step!(model, Δt)
    end
    return sum(interior(model.free_surface.displacement) .^ 2)
end

function free_surface_gradient(model, dmodel, η₀, dη₀, Δt, Nt)
    parent(dη₀) .= 0
    _, J = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                           free_surface_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel),
                           Enzyme.Duplicated(η₀, dη₀),
                           Enzyme.Const(Δt),
                           Enzyme.Const(Nt))
    return dη₀, J
end

fold_configurations = ((CenterFolded, UPivot), (FaceFolded, FPivot))

@testset "Reactant TripolarGrid" begin
    @testset "fill_halo_regions! with raise=true ($fold_topology, $pivot)" for (fold_topology, pivot) in fold_configurations
        kw = (size=(8, 10, 2), halo=(3, 3, 3), z=(0, 1), fold_topology, pivot)
        vanilla_grid  = TripolarGrid(CPU(); kw...)
        reactant_grid = TripolarGrid(ReactantState(); kw...)

        for LX in (Center, Face), LY in (Center, Face)
            vanilla_field  = Field{LX, LY, Center}(vanilla_grid)
            reactant_field = Field{LX, LY, Center}(reactant_grid)
            data = randn(size(vanilla_field)...)
            set!(vanilla_field, data)
            set!(reactant_field, data)

            fill_halo_regions!(vanilla_field)
            compiled_fill! = @compile raise=true raise_first=true fill_halo_regions!(reactant_field)
            compiled_fill!(reactant_field)
            @test compare_parent("halo", vanilla_field, reactant_field)
        end
    end

    @testset "HydrostaticFreeSurfaceModel ($fold_topology, $pivot)" for (fold_topology, pivot) in fold_configurations
        Random.seed!(1234)
        model = tide_model(masked_tripolar_grid(CPU(); fold_topology, pivot))
        reactant_model = tide_model(masked_tripolar_grid(ReactantState(); fold_topology, pivot))

        Δt = 0.2 * minimum_xspacing(model.grid) / sqrt(10 * 4000)
        Nt = 10
        ηᵢ = randn(size(model.free_surface.displacement)...)

        set!(model.free_surface.displacement, ηᵢ)
        set!(reactant_model.free_surface.displacement, ηᵢ)

        for _ in 1:Nt
            time_step!(model, Δt)
        end

        compiled_first_time_step! = @compile raise=true raise_first=true sync=true first_time_step!(reactant_model, Δt)
        compiled_run! = @compile raise=true raise_first=true sync=true run_timesteps!(reactant_model, Δt, Nt - 1)
        compiled_first_time_step!(reactant_model, Δt)
        compiled_run!(reactant_model, Δt, Nt - 1)

        @test compare_interior("η", model.free_surface.displacement, reactant_model.free_surface.displacement)
        @test compare_interior("u", model.velocities.u, reactant_model.velocities.u)
        @test compare_interior("v", model.velocities.v, reactant_model.velocities.v)

        # Gradient of Σ η(T)² with respect to η(0), checked against centered differences along a random direction
        reactant_grid = reactant_model.grid
        η₀  = Field{Center, Center, Nothing}(reactant_grid)
        dη₀ = Field{Center, Center, Nothing}(reactant_grid)
        set!(η₀, ηᵢ)
        dmodel = Enzyme.make_zero(reactant_model)

        compiled_loss = @compile raise=true raise_first=true sync=true free_surface_loss(reactant_model, η₀, Δt, Nt)
        compiled_gradient = @compile raise=true raise_first=true sync=true free_surface_gradient(reactant_model, dmodel, η₀, dη₀, Δt, Nt)

        compiled_gradient(reactant_model, dmodel, η₀, dη₀, Δt, Nt)
        G = Array(interior(dη₀))
        @test !any(isnan, G)

        bottom_height = model.grid.immersed_boundary.bottom_height
        Nx, Ny, Nz = size(model.grid)
        wet = [bottom_height[i, j, 1] < 0 for i in 1:Nx, j in 1:Ny, k in 1:Nz]
        δ = randn(size(ηᵢ)...) .* wet
        ϵ = 1e-4

        loss_at(η) = (set!(η₀, η); Float64(compiled_loss(reactant_model, η₀, Δt, Nt)))
        finite_difference = (loss_at(ηᵢ .+ ϵ .* δ) - loss_at(ηᵢ .- ϵ .* δ)) / 2ϵ
        @test isapprox(sum(G .* δ), finite_difference; rtol=1e-8)
    end
end
