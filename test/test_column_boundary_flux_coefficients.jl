include("dependencies_for_runtests.jl")

using Oceananigans
using Oceananigans.BoundaryConditions: IMEXFluxBoundaryCondition
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: RightFaceFolded, RightCenterFolded, halo_size
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

#####
##### The column-integrated boundary stress the barotropic mode carries
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
                                       closure = VerticalScalarDiffusivity(VerticallyImplicitTimeDiscretization(), ν=0),
                                       boundary_conditions = bcs)
end

function pivotable_indices(jmin, jmax, jpivot)
    idx = jmin:jmax
    rotidx = Int.(2jpivot .- idx)
    valid = @. jmin ≤ rotidx ≤ jmax
    return idx[valid]
end

# The tripolar singularities are not orthogonal and must be masked out of the computation.
function masked_bottom(λ¹ₚ, φₚ, depth)
    return function (λ, φ)
        singular = ((abs(λ - λ¹ₚ)       < 5) & (abs(φₚ - φ) < 5)) |
                   ((abs(λ - λ¹ₚ - 180) < 5) & (abs(φₚ - φ) < 5)) |
                   ((abs(λ - λ¹ₚ - 360) < 5) & (abs(φₚ - φ) < 5)) | (φ < -78)
        return singular ? 0.0 : -depth
    end
end

@testset "Column-integrated implicit boundary-flux coefficients" begin
    for arch in archs

        @testset "Conditional allocation [$(typeof(arch))]" begin
            underlying = RectilinearGrid(arch, size=(4, 4, 8), x=(0, 1e5), y=(0, 1e5), z=(-1000, 0),
                                         topology=(Periodic, Periodic, Bounded))
            grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(-800))

            # Only `u` carries an implicit boundary flux, so only `U` is allocated.
            explicit_model = drag_model(grid; implicit = false, free_surface = SplitExplicitFreeSurface(substeps=12))
            implicit_model = drag_model(grid; implicit = true,  free_surface = SplitExplicitFreeSurface(substeps=12))

            @test explicit_model.free_surface.implicit_boundary_coefficients.U === nothing
            @test explicit_model.free_surface.implicit_boundary_coefficients.V === nothing
            @test implicit_model.free_surface.implicit_boundary_coefficients.U !== nothing
            @test implicit_model.free_surface.implicit_boundary_coefficients.V === nothing

            # `Ω` is the stress the drag realizes on the boundary cell, `-r 𝓋ᵦ`.
            set!(implicit_model, u = 1)
            time_step!(implicit_model, 600)
            # `Ω` is built at the top of the step, so it carries the velocity the step started from.
            Ω = implicit_model.free_surface.implicit_boundary_coefficients.U
            @test minimum(interior(Ω)) ≈ -r_drag
        end

        @testset "Tripolar fold signs [$(typeof(arch))]" begin
            λ¹ₚ, φₚ = 75, 35

            for fold_topology in (RightCenterFolded, RightFaceFolded)
                underlying = TripolarGrid(arch; size = (20, 12, 4), z = (-800, 0), halo = (4, 4, 4),
                                          first_pole_longitude = λ¹ₚ, north_poles_latitude = φₚ,
                                          fold_topology)
                grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(masked_bottom(λ¹ₚ, φₚ, 600.0)))

                # Both components carry an implicit flux so that both `U` and `V` coefficients exist,
                # and the flow is sheared so that `Ω` is not identically zero.
                model = drag_model(grid; implicit = true, drag_on_v = true,
                                   free_surface = SplitExplicitFreeSurface(substeps=12))
                set!(model, u = (λ, φ, z) -> 1 + z / 400, v = (λ, φ, z) -> 1 - z / 400)
                time_step!(model, 600)

                coefficients = model.free_surface.implicit_boundary_coefficients
                Nx, Ny, _ = size(grid)
                Hx, Hy, _ = halo_size(coefficients.U.grid)

                pivotjᶜ, pivotjᶠ = fold_topology == RightFaceFolded ? (Ny + 1/2, Ny + 1) : (Ny, Ny + 1/2)
                iᶠ = iᶜ = 1-Hx:Nx+Hx
                jᶜ = pivotable_indices(1 - Hy, Ny + Hy, pivotjᶜ)
                jᶠ = pivotable_indices(1 - Hy, Ny + Hy + (fold_topology == RightFaceFolded), pivotjᶠ)

                # The fold pivot is offset by one for fields on `Face`s in x.
                iᶜ′ = mod1.(reverse(iᶜ), Nx)
                iᶠ′ = mod1.(reverse(iᶠ) .+ 1, Nx)
                jᶜ′ = reverse(jᶜ)
                jᶠ′ = reverse(jᶠ)

                # `.data` keeps the halo-offset axes the fold indices are written in.
                Ωᵁ = on_architecture(CPU(), coefficients.U.data)
                Ωⱽ = on_architecture(CPU(), coefficients.V.data)

                # Guard against a vacuous test: a zero array satisfies both symmetry and antisymmetry.
                @test maximum(abs, view(Ωᵁ, iᶠ, jᶜ, 1)) > 0
                @test maximum(abs, view(Ωⱽ, iᶜ, jᶠ, 1)) > 0

                # `Ω` is a tendency on a velocity component: it changes sign across the fold.
                @test view(Ωᵁ, iᶠ, jᶜ, 1) == -view(Ωᵁ, iᶠ′, jᶜ′, 1)
                @test view(Ωⱽ, iᶜ, jᶠ, 1) == -view(Ωⱽ, iᶜ′, jᶠ′, 1)
            end
        end

        @testset "Barotropic mode realizes the true boundary stress [$(typeof(arch))]" begin
            # The drag is weak enough that the explicit treatment is stable and serves as the reference.
            zfaces = [-1000, -800, -750, -650, -500, -350, -200, -80, 0]
            underlying = RectilinearGrid(arch, size=(4, 4, 8), x=(0, 1e5), y=(0, 1e5), z=zfaces,
                                         topology=(Periodic, Periodic, Bounded))
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
        end

        @testset "A bottom drag does not accelerate the surface [$(typeof(arch))]" begin
            # With no vertical mixing the drag reaches only the bottom cell, so every other cell must end
            # at its initial velocity: the depth-uniform correction would otherwise double count the stress.
            zfaces = [-1000, -800, -750, -650, -500, -350, -200, -80, 0]
            underlying = RectilinearGrid(arch, size=(4, 4, 8), x=(0, 1e5), y=(0, 1e5), z=zfaces,
                                         topology=(Periodic, Periodic, Bounded))
            grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(-800))

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
        end

        @testset "Implicit free surface needs no correction [$(typeof(arch))]" begin
            # Its predictor already carries `λ` before the free surface is solved, so it should match the
            # unsplit reference with no correction of its own.
            Δt = 3600
            zfaces = collect(range(-1000, 0, length = 21))
            underlying = RectilinearGrid(arch, size=(4, 4, 20), x=(0, 1e5), y=(0, 1e5), z=zfaces,
                                         topology=(Periodic, Periodic, Bounded))
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
