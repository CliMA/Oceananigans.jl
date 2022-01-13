using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, ExplicitTimeDiscretization
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.Grids: ZRegRectilinearGrid
using Oceananigans.Diagnostics: accurate_cell_advection_timescale

function run_tracer_1D_immersed_diffusion(underlying_grid, time_stepping)
    
    immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> 0))

    closure = IsotropicDiffusivity(κ = 1.0, time_discretization = time_stepping)

    model_kwargs = (tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields())

    model = HydrostaticFreeSurfaceModel(grid = underlying_grid, 
                                     closure = closure;
                                     model_kwargs...)

    immersed_model = HydrostaticFreeSurfaceModel(grid = immersed_grid,
                                              closure = closure; 
                                              model_kwargs...)
                                        
    initial_temperature(x, y, z) = exp(-z^2 / 0.02)
    set!(         model, c = initial_temperature)
    set!(immersed_model, c = initial_temperature)

    diffusion_time_scale = minimum(model.grid.Δzᵃᵃᶜ)^2 / model.closure.κ.c
    stop_time = 100diffusion_time_scale

    simulations = [simulation = Simulation(m, Δt = 1e-1 * diffusion_time_scale, stop_time = stop_time) for m in (model, immersed_model)]
    [run!(sim) for sim in simulations]

    test_domain = Int.((underlying_grid.Nz/2 + 1 + underlying_grid.Hz):(underlying_grid.Nz + underlying_grid.Hz))

    c          = Array(parent(model.tracers.c))[1, 1, test_domain]
    c_immersed = Array(parent(immersed_model.tracers.c))[1, 1, test_domain]

    @test all(c_immersed .≈ c)
end

function run_velocity_1D_immersed_diffusion(underlying_grid, time_stepping)
    
    immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom((x, y) -> 0))

    closure = IsotropicDiffusivity(ν = 1.0, time_discretization = time_stepping)

    model = HydrostaticFreeSurfaceModel(grid = underlying_grid, 
                                     closure = closure,
                                    buoyancy = nothing)

    immersed_model = HydrostaticFreeSurfaceModel(grid = immersed_grid,
                                              closure = closure,
                                             buoyancy = nothing)
                                        
    initial_velocity(x, y, z) = exp(-z^2 / 0.02)
    set!(         model, u = initial_velocity)
    set!(immersed_model, u = initial_velocity)

    diffusion_time_scale = minimum(model.grid.Δzᵃᵃᶜ)^2 / model.closure.ν
    stop_time = 100diffusion_time_scale

    simulations = [simulation = Simulation(m, Δt = 1e-1 * diffusion_time_scale, stop_time = stop_time) for m in (model, immersed_model)]
    [run!(sim) for sim in simulations]

    test_domain = Int.((underlying_grid.Nz/2 + 1 + underlying_grid.Hz):(underlying_grid.Nz + underlying_grid.Hz))

    u          = Array(parent(model.velocities.u))[1, 1, test_domain]
    u_immersed = Array(parent(immersed_model.velocities.u))[1, 1, test_domain]

    @test all(u_immersed .≈ u)
end

function run_velocity_2D_immersed_diffusion(arch)

    Nx, Nz = 128, 64
    
    underlying_grid = RectilinearGrid(arch, size = (Nx, Nz), extent = (10, 5), topology = (Periodic, Flat, Bounded))

    Δz = underlying_grid.Δzᵃᵃᶜ
    Lz = underlying_grid.Lz
    
    @inline wedge(x, y) = @. max(0, min( 1/2.5 * x - 1, -2/5 * x + 3))

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(wedge))

    explicit_closure = IsotropicDiffusivity(ν = 1.0)
    implicit_closure = IsotropicDiffusivity(ν = 1.0, time_discretization = VerticallyImplicitTimeDiscretization())

    explicit_model = HydrostaticFreeSurfaceModel(grid = grid, 
                                              closure = explicit_closure,
                                         free_surface = ImplicitFreeSurface())

    implicit_model = HydrostaticFreeSurfaceModel(grid = grid, 
                                              closure = implicit_closure,
                                         free_surface = ImplicitFreeSurface()) 

    # initial divergence-free velocity
    initial_velocity(x, y, z) = z > - Lz / 2 ? 1 : 0
    
    set!(explicit_model, u = initial_velocity)
    # CFL condition (advective and diffusion) = 0.01
    Δt = accurate_cell_advection_timescale(grid, explicit_model.velocities)
    Δt = min(Δt, Δz^2 / explicit_closure.ν) / 100

    for step in 1:20
        time_step!(explicit_model, Δt)
        time_step!(implicit_model, Δt)
    end

    u_explicit = interior(explicit_model.velocities.u)[:, 1, :]
    u_implicit = interior(implicit_model.velocities.u)[:, 1, :]
    η_explicit = interior(explicit_model.free_surface.η)[:, 1, 1]
    η_implicit = interior(implicit_model.free_surface.η)[:, 1, 1]
end

function stretched_coord(N)
    Δz(k)   = k < N / 2 + 1 ? 2 / (N - 1) * (k - 1) + 1 : - 2 / (N - 1) * (k - N) + 1 
    z_faces = zeros(N+1) 
    for k = 2:N+1
        z_faces[k] = z_faces[k-1] + 3 - Δz(k-1)
    end
    z_faces = z_faces ./ z_faces[end] .- 0.5
    return z_faces
end

@inline show_coord(::ZRegRectilinearGrid) = "Regular grid"
@inline show_coord(::RectilinearGrid)     = "Stretched grid"

@testset "ImmersedVerticalDiffusion" begin
    for arch in archs
        
        time_steppings = (ExplicitTimeDiscretization(), VerticallyImplicitTimeDiscretization())
        
        N = 128
        z_stretched = stretched_coord(N)

        regular_grid   = RectilinearGrid(arch, size = N, z = (-0.5, 0.5), topology = (Flat, Flat, Bounded))
        stretched_grid = RectilinearGrid(arch, size = N, z = z_stretched, topology = (Flat, Flat, Bounded))
        
        for step in time_steppings, grid in (stretched_grid, )
            @info "  Testing 1D immersed diffusion [$(typeof(arch)), $(typeof(step)), $(show_coord(grid))]"
            run_tracer_1D_immersed_diffusion(grid, step)
            run_velocity_1D_immersed_diffusion(grid, step)
        end
    end
end