using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization

@inline surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)
@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

@testset "Immersed boundaries with hydrostatic free surface models" begin
    @info "Testing immersed boundaries with hydrostatic free surface models..."

    for arch in archs
        Nx = 60
        Ny = 60

        # A spherical domain
        underlying_grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                                       longitude = (-30, 30),
                                                       latitude = (15, 75),
                                                       z = (-4000, 0))

        @inline raster_depth(i, j) = 30 < i < 35 && 42 < j < 48

        bathymetry = zeros(Nx,Ny) .- 4000
        bathymetry[31:34,43:47] .= 0
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

        model = HydrostaticFreeSurfaceModel(grid = grid,
                                            architecture = arch,
                                            momentum_advection = VectorInvariant(),
                                            free_surface = free_surface,
                                            coriolis = coriolis,
                                            boundary_conditions = (u=u_bcs, v=v_bcs),
                                            closure = constant_horizontal_diffusivity,
                                            tracers = nothing,
                                            buoyancy = nothing)

        simulation = Simulation(model,
                                Δt = 3600,
                                stop_time = 3600,
                                iteration_interval = 100)

        run!(simulation)

        # If reached here it didn't error, so pass for now!
        @test true
    end
end

