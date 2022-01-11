include("dependencies_for_runtests.jl")

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary, GridFittedBottom
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

@inline surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)
@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

@testset "Immersed boundaries with hydrostatic free surface models" begin
    @info "Testing immersed boundaries with hydrostatic free surface models..."

    for arch in archs

        arch_str = string(typeof(arch))

        @testset "GridFittedBoundary [$arch_str]" begin
            @info "Testing GridFittedBoundary with HydrostaticFreeSurfaceModel [$arch_str]..."

            underlying_grid = RectilinearGrid(arch, size=(8, 8, 8), x = (-5, 5), y = (-5, 5), z = (0, 2))

            bump(x, y, z) = z < exp(-x^2 - y^2)
            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(bump))
            
            for closure in (IsotropicDiffusivity(ν=1, κ=0.5),
                            IsotropicDiffusivity(ν=1, κ=0.5, time_discretization=VerticallyImplicitTimeDiscretization()))

                model = HydrostaticFreeSurfaceModel(grid = grid, 
                                                    tracers = :b,
                                                    buoyancy = BuoyancyTracer(),
                                                    closure = closure)

                u = model.velocities.u
                b = model.tracers.b

                # Linear stratification
                set!(model, u = 1, b = (x, y, z) -> 4 * z)
            
                # Inside the bump
                @test b[4, 4, 2] == 0 
                @test u[4, 4, 2] == 0

                simulation = Simulation(model, Δt = 1e-3, stop_iteration=2)

                run!(simulation)

                # Inside the bump
                @test b[4, 4, 2] == 0
                @test u[4, 4, 2] == 0
            end
        end

        @testset "Surface boundary conditions with immersed boundaries [$arch_str]" begin
            @info "  Testing surface boundary conditions with ImmersedBoundaries in HydrostaticFreeSurfaceModel [$arch_str]..."
        
            Nx = 60
            Ny = 60

            # A spherical domain
            underlying_grid = LatitudeLongitudeGrid(arch,
                                                    size = (Nx, Ny, 1),
                                                    longitude = (-30, 30),
                                                    latitude = (15, 75),
                                                    z = (-4000, 0))

            bathymetry = zeros(Nx, Ny) .- 4000
            view(bathymetry, 31:34, 43:47) .= 0
            bathymetry = arch_array(arch, bathymetry)

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

            free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)
            coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

            surface_wind_stress_parameters = (τ₀ = 1e-4,
                                              Lφ = grid.Ly,
                                              φ₀ = 15)

            surface_wind_stress_bc = FluxBoundaryCondition(surface_wind_stress,
                                                           parameters = surface_wind_stress_parameters)

            μ = 1 / 60days

            u_bottom_drag_bc = FluxBoundaryCondition(u_bottom_drag,
                                                     discrete_form = true,
                                                     parameters = μ)

            v_bottom_drag_bc = FluxBoundaryCondition(v_bottom_drag,
                                                     discrete_form = true,
                                                     parameters = μ)

            u_bcs = FieldBoundaryConditions(top = surface_wind_stress_bc, bottom = u_bottom_drag_bc)
            v_bcs = FieldBoundaryConditions(bottom = v_bottom_drag_bc)

            νh₀ = 5e3 * (60 / grid.Nx)^2
            constant_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh₀)

            model = HydrostaticFreeSurfaceModel(; grid,
                                                momentum_advection = VectorInvariant(),
                                                free_surface = free_surface,
                                                coriolis = coriolis,
                                                boundary_conditions = (u=u_bcs, v=v_bcs),
                                                closure = constant_horizontal_diffusivity,
                                                tracers = nothing,
                                                buoyancy = nothing)


            simulation = Simulation(model, Δt=3600, stop_iteration=1)

            run!(simulation)

            # If reached here it didn't error, so pass for now!
            @test true
        end

        @testset "Correct vertically-integrated lateral face areas with immersed boundaries [$arch_str]" begin
            @info "  Testing correct vertically-integrated lateral face areas with immersed boundaries [$arch_str]..."

            Nx = 5
            Ny = 5

            underlying_grid = RectilinearGrid(arch,
                                              size = (Nx, Ny, 3),
                                              halo = (3, 3, 3),
                                              extent = (Nx, Ny, 3),
                                              topology = (Periodic, Periodic, Bounded))

            # B for bathymetry
            B = [-3. for i=1:Nx, j=1:Ny ]
            B[2:Nx-1,2:Ny-1] .= [-2. for i=2:Nx-1, j=2:Ny-1 ]
            B[3:Nx-2,3:Ny-2] .= [-1. for i=3:Nx-2, j=3:Ny-2 ]

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(B))

            model = HydrostaticFreeSurfaceModel(; grid,
                                                free_surface = ImplicitFreeSurface(),
                                                tracer_advection = WENO5(),
                                                buoyancy = nothing,
                                                tracers = nothing,
                                                closure = nothing)

            x_ref = [0.0  0.0  0.0  0.0  0.0  0.0  0.0
                     0.0  3.0  3.0  3.0  3.0  3.0  0.0
                     0.0  3.0  2.0  2.0  2.0  2.0  0.0
                     0.0  3.0  2.0  1.0  1.0  2.0  0.0
                     0.0  3.0  2.0  2.0  2.0  2.0  0.0
                     0.0  3.0  3.0  3.0  3.0  3.0  0.0
                     0.0  0.0  0.0  0.0  0.0  0.0  0.0]'

            y_ref = [0.0  0.0  0.0  0.0  0.0  0.0  0.0
                     0.0  3.0  3.0  3.0  3.0  3.0  0.0
                     0.0  3.0  2.0  2.0  2.0  3.0  0.0
                     0.0  3.0  2.0  1.0  2.0  3.0  0.0
                     0.0  3.0  2.0  1.0  2.0  3.0  0.0
                     0.0  3.0  2.0  2.0  2.0  3.0  0.0
                     0.0  0.0  0.0  0.0  0.0  0.0  0.0]'

            fs = model.free_surface
            vertically_integrated_lateral_areas = fs.implicit_step_solver.vertically_integrated_lateral_areas

            ∫Axᶠᶜᶜ = vertically_integrated_lateral_areas.xᶠᶜᶜ
            ∫Ayᶜᶠᶜ = vertically_integrated_lateral_areas.yᶜᶠᶜ

            ∫Axᶠᶜᶜ = Array(parent(∫Axᶠᶜᶜ))
            ∫Ayᶜᶠᶜ = Array(parent(∫Ayᶜᶠᶜ))

            Ax_ok = ∫Axᶠᶜᶜ[:, :, 1] ≈ x_ref
            Ay_ok = ∫Ayᶜᶠᶜ[:, :, 1] ≈ y_ref

            @test (Ax_ok & Ay_ok)
        end
    end
end

