include("dependencies_for_runtests.jl")

using LinearAlgebra

using Oceananigans.TurbulenceClosures: TriadIsopycnalSkewSymmetricDiffusivity
using Oceananigans.TurbulenceClosures: diffusive_flux_x, diffusive_flux_y, diffusive_flux_z,
                                       ExplicitTimeDiscretization, VerticallyImplicitTimeDiscretization,
                                       compute_closure_fields!

"""
Test that TriadIsopycnalSkewSymmetricDiffusivity can be constructed and timestepped
with both any time discretization.
"""
function time_step_with_triad_isopycnal_diffusivity(arch, time_discretization)
    grid = RectilinearGrid(arch, size=(4, 4, 8), extent=(100, 100, 100))

    closure = TriadIsopycnalSkewSymmetricDiffusivity(time_discretization, Float64,
                                                     κ_skew = 100.0,
                                                     κ_symmetric = 100.0)

    # TriadIsopycnalSkewSymmetricDiffusivity only works with HydrostaticFreeSurfaceModel
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
    TriadIsopycnalSkewSymmetricDiffusivity(time_discretization, Float64; κ_symmetric=1000.0, κ_skew=0, kw...)

@testset "TriadIsopycnalSkewSymmetricDiffusivity" begin
    @info "Testing TriadIsopycnalSkewSymmetricDiffusivity..."

    for arch in archs
        @testset "Time stepping with TriadIsopycnalSkewSymmetricDiffusivity [$arch]" begin
            for time_discretization in [ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization()]
                @info "  Time-stepping TriadIsopycnalSkewSymmetricDiffusivity with $(typeof(time_discretization)) on $arch..."
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
