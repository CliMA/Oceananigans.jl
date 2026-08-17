include("dependencies_for_runtests.jl")

using Oceananigans.Fields: VelocityFields
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: calculate_substeps,
                                                                                  calculate_adaptive_settings,
                                                                                  ConstantAveragingKernel,
                                                                                  materialize_free_surface,
                                                                                  SplitExplicitFreeSurface,
                                                                                  iterate_split_explicit!,
                                                                                  weights_from_substeps,
                                                                                  LowDissipationAveragingKernel,
                                                                                  SymmetricTrigAveragingKernel,
                                                                                  WideTrig74AveragingKernel,
                                                                                  WideTrig2AveragingKernel,
                                                                                  OptimizedSymmetricAveragingKernel,
                                                                                  OptimizedAsymmetricAveragingKernel

@inline noforcing(args...) = 0

clock = Clock{Float64}(time=0)

@testset "Split-Explicit Dynamics" begin

    for FT in float_types
        for arch in archs
            topology = (Periodic, Periodic, Bounded)

            Nx, Ny, Nz = 128, 64, 1
            Lx = Ly = 2π
            Lz = 1 / Oceananigans.defaults.gravitational_acceleration

            grid = RectilinearGrid(arch, FT;
                                   topology, size = (Nx, Ny, Nz),
                                   x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                                   halo = (1, 1, 1))

            velocities = VelocityFields(grid)

            sefs = SplitExplicitFreeSurface(substeps = 200, averaging_kernel = ConstantAveragingKernel())
            sefs = materialize_free_surface(sefs, velocities, grid)

            sefs.displacement .= 0
            GU = Field{Face, Center, Nothing}(grid)
            GV = Field{Center, Face, Nothing}(grid)

            @testset " One timestep test " begin
                state = sefs.filtered_state
                U, V  = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅

                η = sefs.displacement
                Δτ = 1.0

                η₀(x, y, z) = sin(x)
                set!(η, η₀)

                Nsubsteps = calculate_substeps(sefs.substepping, 1)
                fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                iterate_split_explicit!(sefs, grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, transport_weights, nothing, Val(1))

                U_computed = Array(U.data.parent)[2:Nx+1, 2:Ny+1]
                U_exact = (reshape(-cos.(grid.xᶠᵃᵃ), (length(grid.xᶜᵃᵃ), 1)).+reshape(0 * grid.yᵃᶜᵃ, (1, length(grid.yᵃᶜᵃ))))[2:Nx+1, 2:Ny+1]

                @test maximum(abs.(U_exact - U_computed)) < 1e-3
            end

            @testset "Multi-timestep test " begin
                state = sefs.filtered_state
                U, V = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
                η = sefs.displacement

                T  = 2π
                Δτ = 2π / maximum([Nx, Ny]) * 5e-2 # the last factor is essentially the order of accuracy
                Nt = floor(Int, T / Δτ)
                Δτ_end = T - Nt * Δτ

                sefs = SplitExplicitFreeSurface(substeps = Nt, averaging_kernel = ConstantAveragingKernel())
                sefs = materialize_free_surface(sefs, velocities, grid)

                # set!(η, f(x, y))
                η₀(x, y, z) = sin(x)
                set!(η, η₀)
                set!(U, 0)
                set!(V, 0)

                η̅  .= 0
                U̅  .= 0
                V̅  .= 0
                GU .= 0
                GV .= 0

                weights = sefs.substepping.averaging_weights

                for _ in 1:Nt
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, weights, nothing, Val(1))
                end
                iterate_split_explicit!(sefs, grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, weights, nothing, Val(1))

                U_computed = Array(deepcopy(interior(U)))
                η_computed = Array(deepcopy(interior(η)))
                set!(η, η₀)
                set!(U, 0)
                U_exact = Array(deepcopy(interior(U)))
                η_exact = Array(deepcopy(interior(η)))

                @test maximum(abs.(U_computed - U_exact)) < 1e-3
                @test maximum(abs.(η_computed - η_exact)) < max(100eps(FT), 1e-6)
            end

            sefs = SplitExplicitFreeSurface(substeps = 200, averaging_kernel = ConstantAveragingKernel())
            sefs = materialize_free_surface(sefs, velocities, grid)

            sefs.displacement .= 0

            @testset "Averaging / Do Nothing test " begin
                state = sefs.filtered_state
                U, V  = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
                η = sefs.displacement
                g = sefs.gravitational_acceleration

                Δτ = 2π / maximum([Nx, Ny]) * 1e-2 # the last factor is essentially the order of accuracy

                # set!(η, f(x, y))
                η_avg = 1
                U_avg = 2
                V_avg = 3
                fill!(η, η_avg)
                fill!(U, U_avg)
                fill!(V, V_avg)

                fill!(η̅ , 0)
                fill!(U̅ , 0)
                fill!(V̅ , 0)
                fill!(GU, 0)
                fill!(GV, 0)

                Nsubsteps  = calculate_substeps(sefs.substepping, 1)
                fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs.substepping, Nsubsteps) # barotropic time step in fraction of baroclinic step and averaging weights

                for step in 1:Nsubsteps
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, transport_weights, nothing, Val(1))
                end

                U_computed = Array(deepcopy(interior(U)))
                V_computed = Array(deepcopy(interior(V)))
                η_computed = Array(deepcopy(interior(η)))

                U̅_computed = Array(deepcopy(interior(U̅)))
                V̅_computed = Array(deepcopy(interior(V̅)))
                η̅_computed = Array(deepcopy(interior(η̅)))

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

                sefs = SplitExplicitFreeSurface(grid; substeps = Nt + 1, averaging_kernel = ConstantAveragingKernel())
                sefs = materialize_free_surface(sefs, velocities, grid)

                state = sefs.filtered_state
                U, V = sefs.barotropic_velocities
                η̅, U̅, V̅ = state.η̅, state.U̅, state.V̅
                η = sefs.displacement
                g = sefs.gravitational_acceleration

                # set!(η, f(x, y)) k² = ω²
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
                GU .= gu_c
                GV .= gv_c

                weights = sefs.substepping.averaging_weights
                for i in 1:Nt
                    iterate_split_explicit!(sefs, grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, weights, nothing, Val(1))
                end
                iterate_split_explicit!(sefs, grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, weights, nothing, Val(1))

                η_mean_after = mean(Array(interior(η)))

                tolerance = 10eps(FT)
                @test abs(η_mean_after - η_mean_before) < tolerance

                η_computed = Array(deepcopy(interior(η, :, 1, 1)))
                U_computed = Array(deepcopy(interior(U, :, 1, 1)))
                V_computed = Array(deepcopy(interior(V, :, 1, 1)))

                η̅_computed = Array(deepcopy(interior(η̅, :, 1, 1)))
                U̅_computed = Array(deepcopy(interior(U̅, :, 1, 1)))
                V̅_computed = Array(deepcopy(interior(V̅, :, 1, 1)))

                set!(η, η₀)

                # ∂ₜₜ(η) = Δη
                η_exact = cos(ω * T) * (Array(interior(η, :, 1, 1)) .- 1) .+ 1

                U₀(x, y) = kx * cos(kx * x) * sin(ky * y) # ∂ₜU = - ∂x(η), since we know η
                set!(U, U₀)
                U_exact = -(sin(ω * T) * 1 / ω) .* Array(interior(U, :, 1, 1)) .+ gu_c * T

                V₀(x, y) = ky * sin(kx * x) * cos(ky * y) # ∂ₜV = - ∂y(η), since we know η
                set!(V, V₀)
                V_exact = -(sin(ω * T) * 1 / ω) .* Array(interior(V, :, 1, 1)) .+ gv_c * T

                η̅_exact = (sin(ω * T) / ω - sin(ω * 0) / ω) / T * (Array(interior(η, :, 1, 1)) .- 1) .+ 1
                U̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(interior(U, :, 1, 1)) .+ gu_c * T / 2
                V̅_exact = (cos(ω * T) * 1 / ω^2 - cos(ω * 0) * 1 / ω^2) / T * Array(interior(V, :, 1, 1)) .+ gv_c * T / 2

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

@testset "extend_halos vs fill_halos consistency" begin
    for arch in archs
        topology = (Periodic, Periodic, Bounded)
        Nx, Ny, Nz = 32, 32, 1
        Lx = Ly = 2π
        Lz = 1 / Oceananigans.defaults.gravitational_acceleration

        grid = RectilinearGrid(arch, Float64;
                               topology, size = (Nx, Ny, Nz),
                               x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                               halo = (1, 1, 1))

        velocities = VelocityFields(grid)
        Nsubsteps = 30

        # Create two free surfaces: one with extended halos, one that fills halos each substep
        sefs_extend = SplitExplicitFreeSurface(grid; substeps = Nsubsteps,
                                               averaging_kernel = ConstantAveragingKernel(),
                                               extend_halos = true)
        sefs_extend = materialize_free_surface(sefs_extend, velocities, grid)

        sefs_fill = SplitExplicitFreeSurface(grid; substeps = Nsubsteps,
                                             averaging_kernel = ConstantAveragingKernel(),
                                             extend_halos = false)
        sefs_fill = materialize_free_surface(sefs_fill, velocities, grid)

        # Slow barotropic forcing
        GU = Field{Face, Center, Nothing}(grid)
        GV = Field{Center, Face, Nothing}(grid)
        GU .= 0
        GV .= 0

        # Initial condition
        η₀(x, y, z) = sin(x) * cos(y)

        for (label, sefs) in [("extend_halos", sefs_extend), ("fill_halos", sefs_fill)]
            set!(sefs.displacement, η₀)
            sefs.barotropic_velocities.U .= 0
            sefs.barotropic_velocities.V .= 0
            for field in sefs.filtered_state
                fill!(field, 0)
            end
        end

        Δτ = 1.0
        fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs_extend.substepping, Nsubsteps)

        iterate_split_explicit!(sefs_extend, sefs_extend.displacement.grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, transport_weights, nothing, Val(Nsubsteps))

        fractional_Δt, weights, transport_weights = calculate_adaptive_settings(sefs_fill.substepping, Nsubsteps)

        iterate_split_explicit!(sefs_fill, grid, GU, GV, Δτ, Δτ, noforcing, clock, weights, transport_weights, nothing, Val(Nsubsteps))

        # Compare: both should give the same interior result
        η_extend = Array(interior(sefs_extend.displacement))
        η_fill   = Array(interior(sefs_fill.displacement))
        U_extend = Array(interior(sefs_extend.barotropic_velocities.U))
        U_fill   = Array(interior(sefs_fill.barotropic_velocities.U))
        V_extend = Array(interior(sefs_extend.barotropic_velocities.V))
        V_fill   = Array(interior(sefs_fill.barotropic_velocities.V))

        @test η_extend ≈ η_fill
        @test U_extend ≈ U_fill
        @test V_extend ≈ V_fill

        η̅_extend = Array(interior(sefs_extend.filtered_state.η̅))
        η̅_fill   = Array(interior(sefs_fill.filtered_state.η̅))
        U̅_extend = Array(interior(sefs_extend.filtered_state.U̅))
        U̅_fill   = Array(interior(sefs_fill.filtered_state.U̅))

        @test η̅_extend ≈ η̅_fill
        @test U̅_extend ≈ U̅_fill
    end
end

@testset "Averaging kernel moments" begin
    kernel_moment(Δτ, w, p) = sum(w[m] * (m * Δτ - 1)^p for m in eachindex(w))
    for FT in float_types
        # multiples of 16 land the wide-kernel window edges on the substep grid → exact μ₃ = 0
        for substeps in (48, 64)
            tol = sqrt(eps(FT))
            for (kernel, third_order) in ((LowDissipationAveragingKernel(), false),
                                          (SymmetricTrigAveragingKernel(),  true),
                                          (WideTrig74AveragingKernel(),     true),
                                          (WideTrig2AveragingKernel(),      true),
                                          (OptimizedSymmetricAveragingKernel(),  true),
                                          (OptimizedAsymmetricAveragingKernel(), true))
                Δτ, w, transport_weights = weights_from_substeps(FT, substeps, kernel)
                @test sum(w) ≈ 1                             atol=tol   # μ₀
                @test kernel_moment(Δτ, w, 1) ≈ 0            atol=tol   # μ₁ = 1 (barycenter on the baroclinic step)
                @test kernel_moment(Δτ, w, 2) ≈ 0            atol=tol   # μ₂ = 0
                third_order && @test kernel_moment(Δτ, w, 3) ≈ 0 atol=tol  # μ₃ = 0
                @test sum(transport_weights) ≈ 1            atol=tol   # reversed-cumsum transport ⇒ tracer constancy
            end

            # widening the window deepens μ₄ (more low-frequency dissipation): trig < trig74 < trig2 < 0
            μ₄_trig   = kernel_moment(weights_from_substeps(FT, substeps, SymmetricTrigAveragingKernel())[1:2]...,   4)
            μ₄_trig74 = kernel_moment(weights_from_substeps(FT, substeps, WideTrig74AveragingKernel())[1:2]...,      4)
            μ₄_trig2  = kernel_moment(weights_from_substeps(FT, substeps, WideTrig2AveragingKernel())[1:2]...,       4)
            @test μ₄_trig74 < μ₄_trig < 0
            @test μ₄_trig2  < μ₄_trig74
        end

        # The optimized asymmetric kernel imposes μ₀ = 1 and μ₁ = μ₂ = μ₃ = 0 directly on the substep grid
        # rather than inheriting them from a continuous symmetry, so unlike the WideTrig kernels its moments
        # are exact for EVERY substep count, not only where the window edges land on the grid.
        for substeps in (30, 44, 50, 60)
            tol = sqrt(eps(FT))
            Δτ, w, transport_weights = weights_from_substeps(FT, substeps, OptimizedAsymmetricAveragingKernel())
            @test sum(w) ≈ 1                      atol=tol
            @test kernel_moment(Δτ, w, 1) ≈ 0     atol=tol
            @test kernel_moment(Δτ, w, 2) ≈ 0     atol=tol
            @test kernel_moment(Δτ, w, 3) ≈ 0     atol=tol
            @test sum(transport_weights) ≈ 1      atol=tol
        end
    end
end

@testset "Slow-forcing reconstruction across the barotropic sub-cycle" begin
    using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
        dimensionless_reconstruction_weights, scale_reconstruction_weights,
        reconstruction_degree, stage_sample_times, cached_stages,
        FrozenSlowForcing, StageQuadraticSlowForcing, ProgressiveSlowForcing, RungeKutta3Scheme
    using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper, SSPRungeKuttaTimeStepper,
                                     ModifiedRungeKutta4TimeStepper, SSPRK3_COEFFICIENTS

    # Evaluate F(s) from the weights and the samples, in the layout the substep kernel uses: the leading
    # slots are the cached stage forcings and the last is the live forcing of the stage in progress.
    poly(w, samples, s) = sum(sum(w[k][j] * samples[j] for j in eachindex(samples)) * s^(k-1) for k in 1:3)

    # Lay `M` stage forcings out the way the substep kernel sees them: samples 1 … M-1 in the cache slots,
    # and sample M -- the forcing of the stage in progress, which is never cached -- in the last slot.
    slots(F, N) = ntuple(j -> j == N + 1 ? F[end] : (j < length(F) ? F[j] : 0.0), N + 1)

    # The shape is stored on the reconstruction and only rescaled by Δt at each stage; the tests exercise
    # exactly that composition.
    weights(τ, D, N, Δt) = scale_reconstruction_weights(dimensionless_reconstruction_weights(τ, D, N, Float64), Δt)

    @testset "Interpolation through the stage values" begin
        Δt = 600.0
        F¹, F², F³ = 1.5, 2.25, 2.5

        # the low-storage RK3 nodes: the quadratic passes through the three samples, at s = 0, Δt/3, Δt/2
        w  = weights((0.0, 1/3, 1/2), Val(3), Val(2), Δt)
        Fs = slots((F¹, F², F³), 2)
        @test poly(w, Fs, 0.0)    ≈ F¹ rtol=1e-12
        @test poly(w, Fs, Δt/3)   ≈ F² rtol=1e-12
        @test poly(w, Fs, Δt/2)   ≈ F³ rtol=1e-12

        # a forcing that is exactly quadratic in time is reproduced away from the nodes too
        q(s) = 3.0 - 0.5s + 2e-4 * s^2
        qs = slots((q(0), q(Δt/3), q(Δt/2)), 2)
        for s in (0.0, 100.0, Δt/2, Δt, 2Δt)
            @test poly(w, qs, s) ≈ q(s) rtol=1e-10
        end

        # the Shu-Osher nodes (0, Δt, Δt/2) must interpolate at *their* points
        wS = weights((0.0, 1.0, 1/2), Val(3), Val(2), Δt)
        @test poly(wS, Fs, 0.0)  ≈ F¹ rtol=1e-12
        @test poly(wS, Fs, Δt)   ≈ F² rtol=1e-12
        @test poly(wS, Fs, Δt/2) ≈ F³ rtol=1e-12

        # a constant forcing must reduce to the frozen value at every s
        for s in (0.0, 137.0, Δt, 2Δt)
            @test poly(w, slots((7.0, 7.0, 7.0), 2), s) ≈ 7.0 rtol=1e-12
        end
    end

    @testset "Progressive reconstruction on the four-stage nodes" begin
        Δt = 600.0
        a  = 0.22
        τ  = (0.0, a, 1/3, 1/2)
        q(s) = 3.0 - 0.5s + 2e-4 * s^2

        # stage 2 is linear through two samples: exact for a linear forcing, and it must interpolate
        ℓ(s) = 2.0 + 0.01s
        w2 = weights(τ[1:2], Val(2), Val(3), Δt)
        ℓ2 = slots((ℓ(0), ℓ(a*Δt)), 3)
        @test poly(w2, ℓ2, 0.0)   ≈ ℓ(0)     rtol=1e-12
        @test poly(w2, ℓ2, a*Δt)  ≈ ℓ(a*Δt)  rtol=1e-12
        @test poly(w2, ℓ2, Δt)    ≈ ℓ(Δt)    rtol=1e-10

        # the cache slots the stage does not use carry a zero weight, so their (stale) contents cannot leak in
        @test all(w2[k][2] == 0 && w2[k][3] == 0 for k in 1:3)

        # stage 3 is the square quadratic through three samples
        w3 = weights(τ[1:3], Val(3), Val(3), Δt)
        q3 = slots((q(0), q(a*Δt), q(Δt/3)), 3)
        for s in (0.0, a*Δt, Δt/3, Δt, 2Δt)
            @test poly(w3, q3, s) ≈ q(s) rtol=1e-10
        end
        @test all(w3[k][3] == 0 for k in 1:3)

        # stage 4 has four samples for three coefficients: a least-squares fit, which is still *exact*
        # when the data are exactly quadratic
        w4 = weights(τ, Val(3), Val(3), Δt)
        q4 = slots((q(0), q(a*Δt), q(Δt/3), q(Δt/2)), 3)
        for s in (0.0, a*Δt, Δt/3, Δt/2, Δt, 2Δt)
            @test poly(w4, q4, s) ≈ q(s) rtol=1e-8
        end

        # and it reduces to the frozen value on a constant forcing
        for s in (0.0, 137.0, Δt, 2Δt)
            @test poly(w4, slots((7.0, 7.0, 7.0, 7.0), 3), s) ≈ 7.0 rtol=1e-10
        end
    end

    @testset "Nodes follow the composition" begin
        # RK3 keeps the nodes it always had
        @test all(stage_sample_times(SplitRungeKuttaTimeStepper(stages=3)) .≈ (0.0, 1/3, 1/2))
        @test all(stage_sample_times(SSPRungeKuttaTimeStepper(coefficients=SSPRK3_COEFFICIENTS)) .≈ (0.0, 1.0, 1/2))

        # MRK4's leading node *is* its damping fraction, read off the composition rather than stored
        for a in (0.22, 1/4, 0.18)
            ts = ModifiedRungeKutta4TimeStepper(; a)
            @test ts.Nstages == 4
            @test all(stage_sample_times(ts) .≈ (0.0, a, 1/3, 1/2))
        end

        # the classical four-stage polynomial is the a = 1/4 member
        @test all(ModifiedRungeKutta4TimeStepper(a = 1/4).β .≈ (4, 3, 2, 1))
    end

    @testset "Degree rules" begin
        prog = ProgressiveSlowForcing(ModifiedRungeKutta4TimeStepper())
        quad = StageQuadraticSlowForcing(SplitRungeKuttaTimeStepper(stages=3))

        @test cached_stages(prog) == 3
        @test cached_stages(quad) == 2

        # progressive: constant, linear, quadratic, quadratic
        @test [reconstruction_degree(prog, m, 4) for m in 1:4] == [1, 2, 3, 3]

        # stage-quadratic: frozen until the final stage
        @test [reconstruction_degree(quad, m, 3) for m in 1:3] == [1, 1, 3]

        @test_throws ArgumentError ProgressiveSlowForcing(SplitRungeKuttaTimeStepper(coefficients=(1,)))

        # the stored nodes are the composition's own
        @test all(ProgressiveSlowForcing(ModifiedRungeKutta4TimeStepper(a=0.22)).nodes .≈ (0.0, 0.22, 1/3, 1/2))
    end

    @testset "Model integration and tracer constancy" begin
        underlying = RectilinearGrid(size=(16, 8, 8), x=(0, 2e5), y=(0, 1e5),
                                     topology=(Periodic, Periodic, Bounded),
                                     z=MutableVerticalDiscretization((-2000, 0)), halo=(4, 4, 4))
        seamount(x, y) = -2000 + 1200 * exp(-((x - 1e5)^2 + (y - 5e4)^2) / (2.5e4)^2)
        grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(seamount))

        function run(slow_forcing)
            fs = SplitExplicitFreeSurface(grid; substeps=24, timestepper=RungeKutta3Scheme(), slow_forcing)
            model = HydrostaticFreeSurfaceModel(grid; free_surface=fs, buoyancy=BuoyancyTracer(),
                                                tracers=(:b, :c), timestepper=:SplitRungeKutta3,
                                                vertical_coordinate=ZStarCoordinate())
            bᵢ(x, y, z) = 1e-5 * z + 1e-3 * exp(-((x - 6e4)^2 + (y - 5e4)^2) / (2e4)^2)
            set!(model, b=bᵢ, c=1.0, u=0.1)
            for _ in 1:20
                time_step!(model, 120.0)
            end
            return model
        end

        frozen = run(FrozenSlowForcing())
        quad   = run(StageQuadraticSlowForcing(SplitRungeKuttaTimeStepper(stages=3)))

        # The reconstruction touches the momentum forcing only, never the transport that advects tracers,
        # so constancy on the z★ grid must survive untouched.
        for model in (frozen, quad)
            c = interior(model.tracers.c)
            wet = c .!= 0
            @test maximum(abs, c[wet] .- 1) < 1e-12
        end

        # It must change the answer, but only at the level of a second-order scheme difference.
        ηf = interior(frozen.free_surface.displacement)
        ηq = interior(quad.free_surface.displacement)
        @test ηf != ηq
        @test maximum(abs, ηq .- ηf) < 0.05 * maximum(abs, ηf)
    end
end
