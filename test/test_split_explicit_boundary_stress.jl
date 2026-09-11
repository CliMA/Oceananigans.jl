include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.BoundaryConditions: IMEXFluxBoundaryCondition

#####
##### The boundary stress the split-explicit correction must not spread over the column
#####

const r_drag = 2e-3

# A linear drag `J = -r 𝓋` is affine with no explicit part, so it is carried entirely by `λ = -r`.
@inline linear_drag(i, j, k, grid, clock, Φ, r) = @inbounds -r * Φ.u[i, j, k]

drag_boundary_conditions(r; implicit) =
    FieldBoundaryConditions(immersed = ImmersedBoundaryCondition(bottom =
        implicit ? IMEXFluxBoundaryCondition(0, -r) :
                   FluxBoundaryCondition(linear_drag, discrete_form=true, parameters=r)))

function drag_model(grid; implicit, free_surface, drag_on_v = false)
    bcs = drag_on_v ? (u = drag_boundary_conditions(r_drag; implicit),
                       v = drag_boundary_conditions(r_drag; implicit)) :
                      (u = drag_boundary_conditions(r_drag; implicit),)

    return HydrostaticFreeSurfaceModel(grid; free_surface,
                                       momentum_advection = nothing,
                                       tracer_advection = nothing,
                                       tracers = (),
                                       buoyancy = nothing,
                                       coriolis = nothing,
                                       closure = nothing,
                                       boundary_conditions = bcs)
end

@testset "Split-explicit boundary stress" begin
    for arch in archs
        @info "Barotropic mode realizes the true boundary stress [$(typeof(arch))]"

        # The drag is weak enough that the explicit treatment is stable and serves as the reference.
        zfaces = [-1000, -800, -750, -650, -500, -350, -200, -80, 0]
        underlying = RectilinearGrid(arch, size=(4, 4, 8), x=(0, 1e5), y=(0, 1e5), z=zfaces, topology=(Periodic, Periodic, Bounded))
        grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(-800))

        transport(m) = sum(Array(interior(m.velocities.u))[1, 1, :] .* diff(zfaces))

        function evolve(implicit)
            m = drag_model(grid; implicit, free_surface = SplitExplicitFreeSurface(substeps=30))
            set!(m, u = 1)
            U₀ = transport(m)
            for _ in 1:10; time_step!(m, 900); end
            return transport(m) / U₀
        end

        @test evolve(true) ≈ evolve(false) rtol=2e-2

        @info "A bottom drag does not accelerate the surface [$(typeof(arch))]"

        for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3), Δt in (900, 7200)
            m = HydrostaticFreeSurfaceModel(grid; timestepper,
                                            free_surface = SplitExplicitFreeSurface(substeps=30),
                                            momentum_advection = nothing, tracer_advection = nothing,
                                            tracers = (), buoyancy = nothing, coriolis = nothing,
                                            closure = nothing,
                                            boundary_conditions = (u = drag_boundary_conditions(r_drag; implicit=true),))
            set!(m, u = 1)
            for _ in 1:10; time_step!(m, Δt); end

            u = Array(interior(m.velocities.u))[:, :, 3:end]
            @test maximum(abs, u .- 1) < 1e-12
        end

        @testset "Steady drag balance feels the free surface [$(typeof(arch))]" begin
            # One level, periodic: continuity holds u uniform, so the steady balance r u = F(x) - g ∂ₓη
            # sets u from the mean forcing and the free-surface tilt from the rest.
            Nx, Lx, H, Δt, Cᴰ = 16, 1e4, 10.0, 20.0, 0.5
            F₀ = F₁ = 1e-4
            kx = 2π / Lx

            grid = RectilinearGrid(arch, size=(Nx, 1), x=(0, Lx), z=(-H, 0), topology=(Periodic, Flat, Bounded))
            m = HydrostaticFreeSurfaceModel(grid;
                    free_surface = SplitExplicitFreeSurface(grid; substeps=30),
                    momentum_advection = nothing, tracer_advection = nothing,
                    tracers = (), buoyancy = nothing, coriolis = nothing, closure = nothing,
                    boundary_conditions = (u = FieldBoundaryConditions(bottom = IMEXFluxBoundaryCondition(0, -Cᴰ)),),
                    forcing = (u = Forcing((x, z, t) -> F₀ + F₁ * sin(kx * x)),))

            for _ in 1:2000; time_step!(m, Δt); end

            g  = m.free_surface.gravitational_acceleration
            xu = xnodes(m.velocities.u)
            η  = Array(interior(m.free_surface.displacement))[:, 1, 1]
            u  = Array(interior(m.velocities.u))[:, 1, 1]
            δxη = (η .- circshift(η, 1)) ./ (Lx / Nx)

            @test u ≈ fill(F₀ * H / Cᴰ, Nx) rtol=1e-6
            @test -g .* δxη ≈ -F₁ .* sin.(kx .* xu) rtol=1e-6
        end

        @testset "Implicit free surface needs no correction [$(typeof(arch))]" begin
            Δt = 3600
            zfaces = collect(range(-1000, 0, length = 21))
            underlying = RectilinearGrid(arch, size=(4, 4, 20), x=(0, 1e5), y=(0, 1e5), z=zfaces, topology=(Periodic, Periodic, Bounded))
            grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(-800))

            Δz = zfaces[2] - zfaces[1]
            β = r_drag * Δt / Δz

            # Unsplit backward Euler reference.
            Nwet = 16
            𝓋ᵦ = 1.0
            𝓋̄ = 1.0
            for _ in 1:10
                𝓋ᵦ⁺ = 𝓋ᵦ / (1 + β)
                𝓋̄ -= (𝓋ᵦ - 𝓋ᵦ⁺) / Nwet
                𝓋ᵦ = 𝓋ᵦ⁺
            end

            m = drag_model(grid; implicit = true, free_surface = ImplicitFreeSurface())
            set!(m, u = 1)
            for _ in 1:10; time_step!(m, Δt); end
            u = Array(interior(m.velocities.u))[1, 1, :]
            wet = [zfaces[k+1] <= 0 && zfaces[k] >= -800 for k in 1:20]

            @test sum(u .* diff(zfaces) .* wet) / 800 ≈ 𝓋̄ rtol=1e-3
        end
    end
end
