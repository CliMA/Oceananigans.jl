include("dependencies_for_runtests.jl")

using LinearAlgebra

using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z,
                                       ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization,
                                       compute_closure_fields!, FluxTapering

"""
Test that IsopycnalSkewSymmetricDiffusivity can be constructed and timestepped
with both any time discretization.
"""
function time_step_with_triad_isopycnal_diffusivity(arch, time_discretization)
    grid = RectilinearGrid(arch, size=(4, 4, 8), extent=(100, 100, 100))

    closure = IsopycnalSkewSymmetricDiffusivity(time_discretization, Float64,
                                                     κ_skew = 100.0,
                                                     κ_symmetric = 100.0)

    # IsopycnalSkewSymmetricDiffusivity only works with HydrostaticFreeSurfaceModel
    model = HydrostaticFreeSurfaceModel(grid; closure,
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :c))

    # A constant stratification initial condition
    set!(model, b=(x, y, z) -> 1e-5 * z)

    # Attempt to time-step
    time_step!(model, 1)

    return true
end

const Lx = 1000e3
const Lz = 1000.0

sloping_buoyancy(x, z) = 1e-5 * z + 1e-2 * (1 + tanh((x - Lx/2) / 100e3)) / 2

# Return the matrix `L` of the tracer operator (`∂t c = L c`) that `closure` applies over one
# `Δt`, built column by column on a uniform two-dimensional grid with a frozen buoyancy field.
function tracer_operator_matrix(closure, arch; nx=8, nz=6, Δt=1.0, binit=sloping_buoyancy)
    grid = RectilinearGrid(arch, size=(nx, nz), x=(0, Lx), z=(-Lz, 0), halo=(4, 4),
                           topology=(Bounded, Flat, Bounded))

    model = HydrostaticFreeSurfaceModel(grid; closure,
                                        velocities = PrescribedVelocityFields(),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :c),
                                        tracer_advection = Centered())

    L = zeros(nx * nz, nx * nz)

    for j in 1:nx*nz
        cⱼ = zeros(nx, 1, nz)
        cⱼ[j] = 1
        set!(model, b=binit, c=cⱼ)
        model.clock.iteration = 0
        model.clock.time = 0
        time_step!(model, Δt; euler=true)
        cⁿ = Array(interior(model.tracers.c))
        L[:, j] .= (vec(cⁿ[:, 1, :]) .- vec(cⱼ[:, 1, :])) ./ Δt
    end

    return L
end

triad_closure(time_discretization; kw...) =
    IsopycnalSkewSymmetricDiffusivity(time_discretization, Float64; κ_symmetric=1000.0, κ_skew=0, kw...)

# Stratification that is healthy everywhere except across one interface of one column, sitting next
# to a sharp front — the shape a mixed-layer base takes in a global simulation. The front runs along
# x + y so that both horizontal slopes are steep at once, which is what makes `ϵ S² ≤ Sₘ²` per
# component add up to the `2 κ Sₘ²` ceiling on `ϵ κ R₃₃`.
function patchy_front_model(closure, arch; n=8, nz=6, N²=1e-5, Δb=1e-2, stratification_contrast=1e-2)
    grid = RectilinearGrid(arch, size=(n, n, nz), x=(0, Lx), y=(0, Lx), z=(-Lz, 0), halo=(4, 4, 4),
                           topology=(Bounded, Bounded, Bounded))

    model = HydrostaticFreeSurfaceModel(grid; closure,
                                        velocities = PrescribedVelocityFields(),
                                        buoyancy = BuoyancyTracer(),
                                        tracers = (:b, :c))

    z = znodes(grid, Center())
    b = zeros(n, n, nz)

    for i in 1:n, j in 1:n, k in 1:nz
        b[i, j, k] = N² * z[k] + Δb * (i + j > n)
    end

    i, j, k = n ÷ 2, n ÷ 2, nz ÷ 2
    b[i, j, k] = b[i, j, k+1] - stratification_contrast * N² * (z[k+1] - z[k])

    set!(model, b = b)
    update_state!(model)

    return model
end

@testset "IsopycnalSkewSymmetricDiffusivity" begin
    @info "Testing IsopycnalSkewSymmetricDiffusivity..."

    for arch in archs
        @testset "Time stepping with IsopycnalSkewSymmetricDiffusivity [$arch]" begin
            for time_discretization in [ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization()]
                @info "  Time-stepping IsopycnalSkewSymmetricDiffusivity with $(typeof(time_discretization)) on $arch..."
                @test time_step_with_triad_isopycnal_diffusivity(arch, time_discretization)
           end
        end

        @testset "Triad operator is self-adjoint and negative semi-definite [$arch]" begin
            @info "  Testing that the triad operator is self-adjoint and negative semi-definite on $arch..."

            L = tracer_operator_matrix(triad_closure(ExplicitTimeDiscretization()), arch)
            λ = eigvals(Symmetric((L .+ L') ./ 2))

            @test maximum(abs, L .- L') < 1e-12 * maximum(abs, L)
            @test maximum(λ) < 1e-10 * abs(minimum(λ))
        end

        @testset "Tapering bounds the triad slopes [$arch]" begin
            @info "  Testing that the tapering factor bounds the triad slopes on $arch..."

            max_slope = 1e-2
            κ = 1000.0

            closure = IsopycnalSkewSymmetricDiffusivity(VerticallyImplicitTimeDiscretization(), Float64;
                                                             κ_symmetric = κ, κ_skew = 0,
                                                             slope_limiter = FluxTapering(max_slope))

            model = patchy_front_model(closure, arch)

            # Each triad obeys ϵ S² ≤ Sₘ², so the eight meeting a vertical face average to ϵ κ R₃₃ ≤ 2 κ Sₘ².
            @test maximum(Array(interior(model.closure_fields.ϵκR₃₃))) <= 2κ * max_slope^2 * (1 + 1e-12)
        end

        @testset "Vertically implicit triads reproduce the explicit operator [$arch]" begin
            @info "  Testing that vertically implicit triads reproduce the explicit operator on $arch..."

            # The R₃₃ component of the Redi tensor is deferred to the implicit solver, which reaches
            # it through `κzᶜᶜᶠ`. A signature mismatch there silently drops the whole component.
            Lᵉ = tracer_operator_matrix(triad_closure(ExplicitTimeDiscretization()), arch, Δt=1)
            Lⁱ = tracer_operator_matrix(triad_closure(VerticallyImplicitTimeDiscretization()), arch, Δt=1)

            @test maximum(abs, Lⁱ .- Lᵉ) < 1e-4 * maximum(abs, Lᵉ)
        end
    end
end
