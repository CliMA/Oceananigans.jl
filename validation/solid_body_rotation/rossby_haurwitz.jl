# # Rossby Haurwitz solutions
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# ## A spherical domain
#
# We use a one-dimensional domain of geophysical proportions,

using Oceananigans
using Oceananigans.Utils: prettytime
using Oceananigans.Units
using Oceananigans.Operators
using Printf
using Oceananigans.Diagnostics: accurate_cell_advection_timescale
using Oceananigans.Advection: VelocityStencil, VorticityStencil, EnstrophyConservingScheme
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
# using GLMakie

#  λ for latitude and ϕ for latitude is
using Oceananigans.Advection: EnergyConservingScheme, EnstrophyConservingScheme
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis

# include("visualization.jl")

# ## Building a `HydrostaticFreeSurfaceModel`
#
# We use `grid` and `coriolis` to build a simple `HydrostaticFreeSurfaceModel`,

function run_rossby_haurwitz(; architecture = CPU(),
                               Nx = 90,        
                               Ny = 30,
                               coriolis_scheme = EnstrophyConservingScheme(),
                               advection_scheme = VectorInvariant(),
                               prefix = "vector_invariant")
    
    h₀ = 8e3

    coriolis = HydrostaticSphericalCoriolis(scheme=coriolis_scheme)

    grid = LatitudeLongitudeGrid(architecture, size = (Nx, Ny, 5),
                                 longitude = (-180, 180), 
                                 latitude = (-80, 80),
                                 z = (-h₀, 0), 
                                 halo = (4, 4, 4), 
                                 precompute_metrics = true)

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> -h₀-1))

    free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, gravitational_acceleration=900)
    
    model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                        tracers = (),
                                        momentum_advection = advection_scheme, 
                                        buoyancy = nothing,
                                        coriolis = coriolis,
                                        closure = nothing)

    R = model.grid.radius   # [m]
    ω = 0.0 # 7.848e-6            # [s⁻¹]
    K = 7.848e-6            # [s⁻¹]
    n = 4                   # dimensionless
    g = model.free_surface.gravitational_acceleration          # [m/s²]
    Ω = 7.292115e-5
    ϵ = 0.0 # perturbation veloctiy # [m/s]

    A(θ) = ω/2 * (2 * Ω + ω) * cos(θ)^2 + 1/4 * K^2 * cos(θ)^(2*n) * ((n+1) * cos(θ)^2 + (2 * n^2 - n - 2) - 2 * n^2 * sec(θ)^2 )
    B(θ) = 2 * K * (Ω + ω) * ((n+1) * (n+2))^(-1) * cos(θ)^(n) * ( n^2 + 2*n + 2 - (n+1)^2 * cos(θ)^2) # why not  (n+1)^2 sin(θ)^2 + 1
    C(θ)  = 1/4 * K^2 * cos(θ)^(2 * n) * ( (n+1) * cos(θ)^2 - (n+2))

    # here: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
    u₁(θ, ϕ) =  R * ω * cos(θ) + R * K * cos(θ)^(n-1) * (n * sin(θ)^2 - cos(θ)^2) * cos(n*ϕ) 
    v₁(θ, ϕ) = -n * K * R * cos(θ)^(n-1) * sin(θ) * sin(n*ϕ) 
    h₁(θ, ϕ) =  0*h₀ + R^2/g * (  A(θ)  +  B(θ)  * cos(n * ϕ) + C(θ) * cos(2 * n * ϕ) ) 

    # Total initial conditions
    # previously: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
    # oceanoganigans: ϕ ∈ [-90, 90], λ ∈ [-180, 180], 
    rescale¹(λ) = (λ + 180)/ 360 * 2π # λ to θ
    rescale²(ϕ) = ϕ / 180 * π # θ to ϕ
    # arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
    uᵢ(λ, ϕ, z) = u₁(rescale²(ϕ), rescale¹(λ))
    vᵢ(λ, ϕ, z) = v₁(rescale²(ϕ), rescale¹(λ))
    hᵢ(λ, ϕ)    = h₁(rescale²(ϕ), rescale¹(λ)) # (rescale¹(λ), rescale²(ϕ))

    u, v, w = model.velocities
    η = model.free_surface.η

    set!(u, uᵢ)
    set!(v, vᵢ)
    set!(η, hᵢ) 

    # Time step restricted on the gravity wave speed. If using the implicit free surface method it is possible to increase it
    Δt =0.1*accurate_cell_advection_timescale(model) 

    simulation = Simulation(model, Δt = Δt, stop_time = 50days)

    # wizard = TimeStepWizard(cfl=0.5, max_change=3.0, max_Δt=3minutes)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
    # simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

    u, v, w = model.velocities
    η=model.free_surface.η

    ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid; computed_dependencies=(u, v))

    ζ = Field(ζ_op)

    output_fields = (; u = u, v = v, η = η, ζ = ζ)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                        schedule = TimeInterval(40Δt),
                                                        prefix = "rh_$(prefix)_Nx$(Nx)",
                                                        overwrite_existing = true)
    run!(simulation)

    return simulation.output_writers[:fields].filepath
end

filepath_w = run_rossby_haurwitz(architecture=GPU(), Nx=512, Ny=256, advection_scheme=WENO5(vector_invariant=VelocityStencil()), prefix = "WENOVectorInvariantVel")
filepath_w = run_rossby_haurwitz(architecture=GPU(), Nx=512, Ny=256, advection_scheme=WENO5(vector_invariant=VorticityStencil()), prefix = "WENOVectorInvariantVort")
filepath_w = run_rossby_haurwitz(architecture=GPU(), Nx=512, Ny=256, advection_scheme=VectorInvariant(), prefix = "VectorInvariant")

