include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.TurbulenceClosures: MixedLayerEddyRestratification, ArrestedFrontalLength,
                                       step_closure_prognostics!, compute_closure_fields!
using Oceananigans: prognostic_state, restore_prognostic_state!

# Potential energy (per unit area and reference density) needed to mix a layer of depth `d`
# over a mixed layer of depth `h₀` and uniform stratification `N²` below
mixing_energy(d, h₀, N²) = (L = max(d - h₀, 0); N² * (L^3 / 12 + h₀ * L^2 / 4))

function expected_mixed_layer_depth(ε★, h₀, N²)
    d₁, d₂ = h₀, 10h₀
    for _ in 1:100
        d = (d₁ + d₂) / 2
        mixing_energy(d, h₀, N²) < ε★ ? (d₁ = d) : (d₂ = d)
    end
    return (d₁ + d₂) / 2
end

function mixed_layer_test_model(arch, FT; Lz = 200, Nz = 100, closure = MixedLayerEddyRestratification(FT),
                                velocities = PrescribedVelocityFields(), boundary_conditions = NamedTuple())

    grid = RectilinearGrid(arch, FT; size = (16, 4, Nz), halo = (4, 4, 4),
                           x = (0, 64e3), y = (0, 16e3), z = (-Lz, 0),
                           topology = (Bounded, Periodic, Bounded))

    return HydrostaticFreeSurfaceModel(grid; closure, velocities, boundary_conditions,
                                       buoyancy = BuoyancyTracer(),
                                       coriolis = FPlane(f = 1e-4),
                                       tracers = (:b, :c))
end

frontal_buoyancy(x, y, z; h₀ = 50, N² = 1e-5, Δb = 2e-3) = N² * min(z + h₀, 0) + Δb / 2 * tanh((x - 32e3) / 8e3)

@testset "MixedLayerEddyRestratification" begin
    for arch in archs, FT in (Float64, Float32)
        @testset "Energy-based mixed-layer depth [$(typeof(arch)), $FT]" begin
            @info "  Testing energy-based mixed-layer depth [$(typeof(arch)), $FT]..."
            h₀, N² = 50, 1e-5
            model = mixed_layer_test_model(arch, FT)
            closure = model.closure
            set!(model, b = (x, y, z) -> N² * min(z + h₀, 0))
            step_closure_prognostics!(model.closure_fields, closure, model, 60)

            ε★ = 25 / 1020
            h = Array(interior(model.closure_fields.h))
            H = Array(interior(model.closure_fields.H))
            @test all(isapprox.(h, expected_mixed_layer_depth(ε★, h₀, N²), atol = 0.5))
            @test H == h

            # A fully mixed column is mixed to the bottom
            set!(model, b = 0)
            step_closure_prognostics!(model.closure_fields, closure, model, 60)
            @test all(Array(interior(model.closure_fields.h)) .≈ 200)

            # The mixed-layer depth does not depend on a constant buoyancy offset
            set!(model, b = (x, y, z) -> 1e-2 + N² * min(z + h₀, 0))
            step_closure_prognostics!(model.closure_fields, closure, model, 60)
            @test all(isapprox.(Array(interior(model.closure_fields.h)), h, rtol = 1e-3))
        end

        @testset "Filtered mixed-layer depth [$(typeof(arch)), $FT]" begin
            @info "  Testing filtered mixed-layer depth [$(typeof(arch)), $FT]..."
            τ = 5 * 86400
            model = mixed_layer_test_model(arch, FT)
            set!(model, b = (x, y, z) -> 1e-5 * min(z + 50, 0))
            step_closure_prognostics!(model.closure_fields, model.closure, model, 60)
            h = Array(interior(model.closure_fields.h))

            # Shoaling relaxes over the decay time
            set!(model.closure_fields.H, 2 .* h)
            step_closure_prognostics!(model.closure_fields, model.closure, model, τ)
            @test all(Array(interior(model.closure_fields.H)) .≈ 3 .* h ./ 2)

            # Deepening is instantaneous
            set!(model.closure_fields.H, h ./ 2)
            step_closure_prognostics!(model.closure_fields, model.closure, model, τ)
            @test all(Array(interior(model.closure_fields.H)) .≈ h)
        end

        @testset "Eddy velocities [$(typeof(arch)), $FT]" begin
            @info "  Testing mixed-layer eddy velocities [$(typeof(arch)), $FT]..."
            model = mixed_layer_test_model(arch, FT)
            grid = model.grid
            K = model.closure_fields

            # Uniform buoyancy induces no overturning
            set!(model, b = 0)
            step_closure_prognostics!(K, model.closure, model, 60)
            compute_closure_fields!(K, model.closure, model)
            @test all(Array(interior(K.u)) .== 0)
            @test all(Array(interior(K.w)) .== 0)

            set!(model, b = frontal_buoyancy)
            step_closure_prognostics!(K, model.closure, model, 60)
            compute_closure_fields!(K, model.closure, model)

            u = Array(interior(K.u))
            v = Array(interior(K.v))
            w = Array(interior(K.w))
            Δx, Δy, Δz = grid.Δxᶠᵃᵃ, grid.Δyᵃᶠᵃ, grid.z.Δᵃᵃᶜ
            Nx, Ny, Nz = size(grid)

            # The overturning restratifies: light water flows over dense water near the surface
            @test maximum(u[2:Nx, :, Nz]) < 0
            @test maximum(abs, v) == 0
            @test maximum(abs, u) ≤ model.closure.maximum_eddy_velocity

            # The eddy velocity is non-divergent and vanishes at the bottom
            @test all(w[:, :, 1] .== 0)
            div = [(u[i+1, j, k] - u[i, j, k]) / Δx + (w[i, j, k+1] - w[i, j, k]) / Δz
                   for i in 1:Nx-1, j in 1:Ny, k in 1:Nz-1]
            @test maximum(abs, div) < 100eps(FT) * maximum(abs, u) / Δz

            # The overturning is confined to the filtered mixed layer
            H = maximum(Array(interior(K.H)))
            kᴴ = searchsortedfirst(znodes(grid, Center()), -H) - 1
            @test all(u[:, :, 1:kᴴ-1] .== 0)
        end

        @testset "Restratification [$(typeof(arch)), $FT]" begin
            @info "  Testing mixed-layer restratification [$(typeof(arch)), $FT]..."
            model = mixed_layer_test_model(arch, FT)
            set!(model, b = frontal_buoyancy, c = (x, y, z) -> exp(z / 20))
            Nz = size(model.grid, 3)

            ∫b₀ = sum(interior(model.tracers.b))
            ∫c₀ = sum(interior(model.tracers.c))
            Δb₀ = mean(interior(model.tracers.b, :, :, Nz)) - mean(interior(model.tracers.b, :, :, Nz - 20))

            for _ in 1:20
                time_step!(model, 600)
            end

            Δb = mean(interior(model.tracers.b, :, :, Nz)) - mean(interior(model.tracers.b, :, :, Nz - 20))
            @test Δb > Δb₀
            @test sum(interior(model.tracers.b)) ≈ ∫b₀ rtol = 100eps(FT)
            @test sum(interior(model.tracers.c)) ≈ ∫c₀ rtol = 100eps(FT)
            @test all(isfinite, Array(interior(model.tracers.b)))
        end

        @testset "Closure tuples [$(typeof(arch)), $FT]" begin
            @info "  Testing mixed-layer eddies in a closure tuple [$(typeof(arch)), $FT]..."
            closure = (VerticalScalarDiffusivity(κ = 1e-5), MixedLayerEddyRestratification(FT))
            model = mixed_layer_test_model(arch, FT; closure, velocities = nothing)
            set!(model, b = frontal_buoyancy)
            time_step!(model, 600)
            time_step!(model, 600)

            @test maximum(abs, Array(interior(model.closure_fields[2].u))) > 0
            @test all(isfinite, Array(interior(model.tracers.b)))
        end

        @testset "Arrested frontal length [$(typeof(arch)), $FT]" begin
            @info "  Testing mixed-layer eddies with an arrested frontal length [$(typeof(arch)), $FT]..."
            scaling = ArrestedFrontalLength(FT)
            closure = MixedLayerEddyRestratification(FT; frontal_length = scaling)

            function forced_model(τ, Jᵇ)
                boundary_conditions = (u = FieldBoundaryConditions(top = FluxBoundaryCondition(FT(τ))),
                                       b = FieldBoundaryConditions(top = FluxBoundaryCondition(FT(Jᵇ))))
                model = mixed_layer_test_model(arch, FT; closure, velocities = nothing, boundary_conditions)
                set!(model, b = frontal_buoyancy)
                step_closure_prognostics!(model.closure_fields, closure, model, 60)
                compute_closure_fields!(model.closure_fields, closure, model)
                return model
            end

            # Wind stress and convection set the turbulent momentum flux
            τ, Jᵇ = 2e-4, 1e-8
            model = forced_model(τ, Jᵇ)
            h = Array(interior(model.closure_fields.h))
            w′u′ = Array(interior(model.closure_fields.w′u′))
            expected = @. cbrt(0.5 * sqrt(τ)^3 + 0.066 * Jᵇ * h)^2
            @test all(isapprox.(w′u′, expected, rtol = 10eps(FT)))

            # Surface buoyancy gain does not drive convection
            model = forced_model(τ, -Jᵇ)
            @test all(Array(interior(model.closure_fields.w′u′)) .≈ cbrt(0.5 * sqrt(τ)^3)^2)

            # The overturning restratifies and weakens as boundary-layer turbulence strengthens
            model₁ = forced_model(τ, 0)
            model₂ = forced_model(2τ, 0)
            Ψ₁ = Array(interior(model₁.closure_fields.Ψˣ))[2:end-1, :, 1]
            Ψ₂ = Array(interior(model₂.closure_fields.Ψˣ))[2:end-1, :, 1]
            Ψₘₐₓ = closure.maximum_eddy_velocity * minimum(Array(interior(model₁.closure_fields.H))) / 5
            @test all(Ψ₁ .< 0)
            @test maximum(abs, Ψ₁) < Ψₘₐₓ
            @test all(isapprox.(Ψ₁, 2 .* Ψ₂, rtol = 100eps(FT)))

            for _ in 1:2
                time_step!(model₁, 600)
            end
            @test all(isfinite, Array(interior(model₁.tracers.b)))
        end

        @testset "Checkpointing the filtered mixed-layer depth [$(typeof(arch)), $FT]" begin
            @info "  Testing checkpointing of the filtered mixed-layer depth [$(typeof(arch)), $FT]..."
            model = mixed_layer_test_model(arch, FT)
            set!(model, b = frontal_buoyancy)
            time_step!(model, 600)

            state = prognostic_state(model.closure_fields)
            H = Array(interior(model.closure_fields.H))

            new_model = mixed_layer_test_model(arch, FT)
            restore_prognostic_state!(new_model.closure_fields, state)
            @test Array(interior(new_model.closure_fields.H)) == H
            @test keys(state) == (:H,)
        end
    end
end
