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

grid = LatitudeLongitudeGrid(size = (360, 180, 1), longitude = (-180, 180), latitude = (-80, 80), z = (-1, 0), radius = 1.2, halo = (3, 3, 3), precompute_metrics=true)
#  λ for latitude and ϕ for latitude is
using Oceananigans.Coriolis: VectorInvariantEnergyConserving, HydrostaticSphericalCoriolis

Ω = 0
coriolis = HydrostaticSphericalCoriolis(Float64, scheme=VectorInvariantEnergyConserving(), rotation_rate = Ω)

s = readline()

# ## Building a `HydrostaticFreeSurfaceModel`
#
# We use `grid` and `coriolis` to build a simple `HydrostaticFreeSurfaceModel`,

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant
using Oceananigans.Models: HydrostaticFreeSurfaceModel

# closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=0, κh=0)
# VectorInvariantEnergyConserving()
# HydrostaticSphericalCoriolis(Float64, scheme=VectorInvariantEnergyConserving())
if s == "1"
    momentum_advection = WENO5(vector_invariant=true) 
else
    momentum_advection =  VectorInvariant() 
end

νh = κh = 1e-8
closure = HorizontalScalarDiffusivity(κ=κh, ν=νh)

model = HydrostaticFreeSurfaceModel(; grid, momentum_advection, closure,
                                    buoyancy = nothing,
                                    tracers = :c,
                                    coriolis = coriolis,
                                    free_surface = ImplicitFreeSurface(gravitational_acceleration=1e-20, 
                                    solver_method=:HeptadiagonalIterativeSolver))

# ## The Bickley jet on a sphere
# θ ∈ [-π/2, π/2]
# ϕ ∈ [-π, π] # 0° is equator
ℓᵐ = 10
ℓ = 20

m = 2
θᵖ = π/2 * 0.05
ϵ = 0.3
vˢ = 5e-4
g = model.free_surface.gravitational_acceleration 
  
# u =   - r⁻¹∂ψ/∂θ
# v =  ( r cos(θ) )⁻¹ ∂ψ/∂ϕ

ψᵐ(θ, ϕ) = tanh(ℓᵐ * θ) 
ψᵖ(θ, ϕ) = exp(-ℓ * (θ - θᵖ)^2) * cos(θ) * cos(2 * (θ - θᵖ)) * sin(m * ϕ)

# here: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
uᵐ(θ, ϕ) =  ℓᵐ * sech(ℓᵐ * θ)^2 
vᵐ(θ, ϕ) =  0.0
hᵐ(θ, ϕ) =  1.0 

u1(θ, ϕ) =  ℓ * 2 * (θ - θᵖ)* exp(-ℓ * (θ - θᵖ)^2) * cos(θ) * cos(2 * (θ - θᵖ)) * sin(m * ϕ)
u2(θ, ϕ) =  exp(-ℓ * (θ - θᵖ)^2) * sin(θ) * cos(2 * (θ - θᵖ)) * sin(m * ϕ)
u3(θ, ϕ) =  2*exp(-ℓ * (θ - θᵖ)^2) * cos(θ) * sin(2 * (θ - θᵖ)) * sin(m * ϕ)
uᵖ(θ, ϕ) =  u1(θ, ϕ) + u2(θ, ϕ) + u3(θ, ϕ)
vᵖ(θ, ϕ) =  m * exp(-ℓ * (θ - θᵖ)^2) * cos(2 * (θ - θᵖ)) * cos(m * ϕ)
hᵖ(θ, ϕ) =  0.0 

u₁(θ, ϕ) = vˢ * (uᵐ(θ, ϕ) + ϵ * uᵖ(θ, ϕ))
v₁(θ, ϕ) = vˢ * (vᵐ(θ, ϕ) + ϵ * vᵖ(θ, ϕ))
h₁(θ, ϕ) = hᵐ(θ, ϕ) + ϵ * hᵖ(θ, ϕ)
tracer(θ, ϕ) = tanh(ℓᵐ * θ)

# Total initial conditions
# previously: θ ∈ [-π/2, π/2] is latitude, ϕ ∈ [0, 2π) is longitude
# oceanoganigans: ϕ ∈ [-90, 90], λ ∈ [-180, 180], 
rescale¹(λ) = (λ + 180)/ 360 * 2π # λ to θ
rescale²(ϕ) = ϕ / 180 * π # θ to ϕ
# arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
uᵢ(λ, ϕ, z) = u₁(rescale²(ϕ), rescale¹(λ))
vᵢ(λ, ϕ, z) = v₁(rescale²(ϕ), rescale¹(λ))
hᵢ(λ, ϕ) = h₁(rescale²(ϕ), rescale¹(λ)) # (rescale¹(λ), rescale²(ϕ))
tracerᵢ(λ, ϕ, z) = tracer(rescale²(ϕ), rescale¹(λ)) # (rescale¹(λ), rescale²(ϕ))

u, v, w = model.velocities
η = model.free_surface.η
c = model.tracers.c
set!(u, uᵢ)
set!(v, vᵢ)
set!(η, hᵢ) 
set!(c, tracerᵢ)

# Create Simulation

gravity_wave_speed = sqrt(model.free_surface.gravitational_acceleration * grid.Lz) # hydrostatic (shallow water) gravity wave speed

Δx = grid.Δxᶠᶜᵃ[1]

# 20wave_propagation_time_scale,
Δt = 0.2 #Δx / gravity_wave_speed 
simulation = Simulation(model, Δt = Δt, stop_time = 6000Δt)

using Printf

progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                        iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

output_fields = merge(model.velocities, (η=model.free_surface.η,), model.tracers)

using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval

if s == "1"
    output = "sb_weno"
else
    output = "sb_noweno"
end

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(100Δt),
                                                      prefix = output,
                                                      force = true)

run!(simulation)

##
#=
using JLD2, GLMakie, Printf

const hours = 3600

λ = range(-180,179,length = 360)
ϕ = range(-79.5,79.5,length = 180)

filename = "sb_weno"

file = jldopen(filename * ".jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

iter = Observable(0)
plot_title = @lift @sprintf("Rossby-Haurwitz Test: u, v, η @ time = %s", file["timeseries/t/" * string($iter)])
up = @lift file["timeseries/u/" * string($iter)][:, :, 1]
cp = @lift file["timeseries/c/" * string($iter)][:, :, 1]
ηp = @lift file["timeseries/η/" * string($iter)][:, :, 1]

up0 = file["timeseries/u/" * string(0)][:, :, 1]
vp0 = file["timeseries/v/" * string(0)][:, :, 1]
ηp0 = file["timeseries/η/" * string(0)][:, :, 1]

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

fig = Figure(resolution = (3500, 3000))

fontsize_theme = Theme(fontsize = 25)
set_theme!(fontsize_theme)

ax1 = fig[1, 1] = LScene(fig) # make plot area wider
wireframe!(ax1, Sphere(Point3f0(0), 1f0), show_axis=false)
hm1 = surface!(ax1, x, y, z, color=up, colormap=:blues, colorrange=(-0.0004, 0.0004))

ax2 = fig[1, 2] = LScene(fig) # make plot area wider
wireframe!(ax2, Sphere(Point3f0(0), 1f0), show_axis=false)
hm2 = surface!(ax2, x, y, z, color=cp, colormap=:solar)

ax3 = fig[1, 3] = LScene(fig) # make plot area wider
wireframe!(ax3, Sphere(Point3f0(0), 1f0), show_axis=false)
hm2 = surface!(ax3, x, y, z, color=ηp, colormap=:balance, colorrange=(0.9,  1.05))

init = (π/5, π/6, 0)
rotate_cam!(ax1.scene, init)
rotate_cam!(ax2.scene, init)
rotate_cam!(ax3.scene, init)

rot  = (0, π/300, 0)

display(fig)

record(fig, filename * ".mp4", iterations[1:end-2], framerate=10) do i
    @info "Plotting iteration $i of $(iterations[end])..."
    # rotate_cam!(ax1.scene, rot)
    # rotate_cam!(ax2.scene, rot)
    # rotate_cam!(ax3.scene, rot)
    iter[] = i
end
=#