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

h₀ = 8e3

grid = LatitudeLongitudeGrid(size = (70, 70, 5), longitude = (-180, 180), latitude = (-80, 80), z = (-h₀, 0), halo = (3, 3, 3), precompute_metrics = true)
#  λ for latitude and ϕ for latitude is
using Oceananigans.Coriolis: VectorInvariantEnergyConserving, HydrostaticSphericalCoriolis

# ## Building a `HydrostaticFreeSurfaceModel`
#
# We use `grid` and `coriolis` to build a simple `HydrostaticFreeSurfaceModel`,

for (adv, scheme) in enumerate([VectorInvariant(), WENO5(vector_invariant=true)])

    free_surface = ImplicitFreeSurface(solver_method=:HeptadiagonalIterativeSolver, gravitational_acceleration=90)
    if adv == 1
        model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                            tracers = (),
                                            momentum_advection = scheme, 
                                            buoyancy = nothing,
                                            coriolis = HydrostaticSphericalCoriolis(scheme=VectorInvariantEnergyConserving()),
                                            closure = nothing)
    else
        model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                            tracers = (),
                                            momentum_advection = scheme, 
                                            buoyancy = nothing,
                                            coriolis =nothing,
                                            closure = nothing)
    end

    # ## The Bickley jet on a sphere
    # λ ∈ [-180°, 180°]
    # ϕ ∈ [-90°, 90°] # 0° is equator
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

    # Create Simulation

    speed = (n * (3 + n ) * ω - 2*Ω) / ((1+n) * (2+n))
    # angles per day speed / π * 180  * 86400
    # abs(45 * π / 180 / speed / 86400) days for 45 degree rotation
    numdays = abs(45 * π / 180 / speed / 86400)

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

    wave_propagation_time_scale = h₀ * model.grid.Δλᶜᵃᵃ / gravity_wave_speed

    # Time step restricted on the gravity wave speed. If using the implicit free surface method it is possible to increase it
    Δt =  10wave_propagation_time_scale

    simulation = Simulation(model, Δt = Δt, stop_time = 60days)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    u, v, w = model.velocities
    η=model.free_surface.η
    ζ_op = KernelFunctionOperation{Face, Face, Center}(ζ₃ᶠᶠᶜ, grid; computed_dependencies=(u, v))

    ζ = Field(ζ_op)

    output_fields = (; u = u, v = v, η = η, ζ = ζ)

    using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                        schedule = TimeInterval(100wave_propagation_time_scale),
                                                        prefix = "rh_$(adv)_solution_",
                                                        force = true)
    ##
    run!(simulation)
    ##
end

#=
using JLD2, GLMakie, Printf

const hours = 3600

λ = range(-180,179,length = 70)
ϕ = range(-79.5,79.5,length = 70)

filename1 = "rh_1_solution_"
filename2 = "rh_2_solution_"

file1 = jldopen(filename1 * ".jld2")
file2 = jldopen(filename2 * ".jld2")

iterations = parse.(Int, keys(file1["timeseries/t"]))

iter = Observable(0)
plot_title = @lift @sprintf("Rossby-Haurwitz Test: u, v, η @ time = %s", file1["timeseries/t/" * string($iter)])
up = @lift file1["timeseries/u/" * string($iter)][:, :, 1]
cp = @lift file1["timeseries/ζ/" * string($iter)][:, :, 1]
ηp = @lift file1["timeseries/η/" * string($iter)][:, :, 1]

uw = @lift file2["timeseries/u/" * string($iter)][:, :, 1]
cw = @lift file2["timeseries/ζ/" * string($iter)][:, :, 1]
ηw = @lift file2["timeseries/η/" * string($iter)][:, :, 1]

function geographic2cartesian(λ, φ; r=1.01)
    Nλ = length(λ)
    Nφ = length(φ)

    λ = repeat(reshape(λ, Nλ, 1), 1, Nφ)
    φ = repeat(reshape(φ, 1, Nφ), Nλ, 1)

    λ_azimuthal = λ .+ 180  
    φ_azimuthal = 90 .- φ   

    x = @. r * cosd(λ_azimuthal) * sind(φ_azimuthal)
    y = @. r * sind(λ_azimuthal) * sind(φ_azimuthal)
    z = @. r * cosd(φ_azimuthal)

    return x, y, z
end

x, y, z = geographic2cartesian(λ, ϕ)

fig = Figure(resolution = (3000, 3000))

fontsize_theme = Theme(fontsize = 25)
set_theme!(fontsize_theme)

ax1 = fig[1, 1] = LScene(fig) # make plot area wider
wireframe!(ax1, Sphere(Point3f0(0), 1f0), show_axis=false)
hm1 = surface!(ax1, x, y, z, color=up, colormap=:balance)

ax2 = fig[1, 2] = LScene(fig) # make plot area wider
wireframe!(ax2, Sphere(Point3f0(0), 1f0), show_axis=false)
hm2 = surface!(ax2, x, y, z, color=uw, colormap=:balance)

init = (π/5, π/6, 0)
rotate_cam!(ax1.scene, init)
rotate_cam!(ax2.scene, init)

rot  = (0, π/300, 0)

display(fig)

record(fig, "test_adv_rossby.mp4", iterations[1:end-2], framerate=10) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    rotate_cam!(ax1.scene, rot)
    rotate_cam!(ax2.scene, rot)
    iter[] = i
end
=#
