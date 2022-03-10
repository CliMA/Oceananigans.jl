using Printf
using Statistics

using LinearAlgebra
using Distributions: MvNormal
using LinearAlgebra: cholesky, Symmetric

using Random
using Oceananigans
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using Oceananigans.Diagnostics: accurate_cell_advection_timescale

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant
using Oceananigans.OutputReaders: FieldTimeSeries, @compute

using Oceananigans.Advection: ZWENO, WENOVectorInvariant
#####
##### The Bickley jet
#####

⊗(a, b) = a * b'
Random.seed!(1234)


Ψ(y) = - tanh(y)
U(y) = sech(y)^2

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

"""
    run_bickley_jet(output_time_interval = 2, stop_time = 200, arch = CPU(), Nh = 64, ν = 0,
                    momentum_advection = VectorInvariant())

Run the Bickley jet validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, on `arch`itecture.
"""
function run_bickley_jet(coeff, Nh = 64)

    C3₀, C3₁ = coeff
    C3₂ = 1 - C3₀ - C3₁

    grid = RectilinearGrid(size=(Nh, Nh, 1),
                                x = (-2π, 2π), y=(-2π, 2π), z=(0, 1), halo = (4, 4, 4),
                                topology = (Periodic, Periodic, Bounded))

    model = HydrostaticFreeSurfaceModel(momentum_advection = WENO5((C3₀, C3₁, C3₂), vector_invariant=true),
                                                      grid = grid,
                                                   tracers = (),
                                                   closure = nothing,
                                              free_surface = ImplicitFreeSurface(gravitational_acceleration=10.0),
                                                  coriolis = nothing,
                                                  buoyancy = nothing)

    # ** Initial conditions **
    #
    # u, v: Large-scale jet + vortical perturbations
    #    c: Sinusoid

    # Parameters
    ϵ = 0.1 # perturbation magnitude
    ℓ = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    # Total initial conditions
    uᵢ(x, y, z) = U(y) + ϵ * ũ(x, y, ℓ, k)
    vᵢ(x, y, z) = ϵ * ṽ(x, y, ℓ, k)
    # cᵢ(x, y, z) = C(y, grid.Ly)

    set!(model, u=uᵢ, v=vᵢ) #, c=cᵢ)

    Δt = 0.1 * accurate_cell_advection_timescale(grid, model.velocities)

    nsteps  = floor(Int, 200 / Δt)
    ninterv = floor(Int, 10 / Δt)
    
    nsave = floor(Int, nsteps/ninterv)

    u, v, w = model.velocities

    # u₀, v₀, _ = deepcopy(model.velocities)

    # @compute ζ₀ = Field(∂x(v₀) - ∂y(u₀))

    enst   = zeros(nsave)
    loss   = zeros(nsave)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    time = 0
    for i in 1:nsave
        enst[i] = sum((∂x(v) - ∂y(u))^ 2) / Nh^2
        for j in 1:ninterv 
            time_step!(model, Δt)
            time = time + Δt
        end
        if i > 1
            loss[i] = (enst[i] - enst[1]) + 0.5 * Int(enst[i] > enst[i-1]) * (enst[i] - enst[i-1])
        end
        @info "iteration $i, time $time, var $(enst[i]), loss $(loss[i])"
    end

    @show model.clock.time

    # @compute ζ = Field(∂x(v) - ∂y(u))
    for i in 2:nsave
    end


    return enst #(model, enst, ζ₀, ζ)
end
   
N = 10
# timestep size (h = 1/N implies T=1 at the final time)
h = 100 / N

M = 20
# Implicitly define the likelihood function via the covariance 
MM = randn(M, M)
Γ = (MM' * MM + I)

ξ = MvNormal(1 / h * Γ)

Nparticles = 10
# number of ensemble members
J = Nparticles 

function regularize(vec)
    if vec[1] > 1
        vec[1] = 1 - (vec[1] - 1)
    elseif vec[1] < 0
        vec[1] = - vec[1]
    end
    if vec[2] > 1
        vec[2] = 1 - (vec[2] - 1) 
    elseif vec[1] < 0
        vec[2] = - vec[2]
    end
    if vec[1] + vec[2] > 1
        x = vec[1] 
        y = vec[2] 
        vec[2] = 1 - x
        vec[1] = 1 - y
    end
    return vec
end

u = [regularize([rand() - 0.2, rand() + 0.1]) for i in 1:Nparticles]

ȳ = zeros(20)
timeseries = []

# EKI Algorithm
for i = 1:N
    u̅ = mean(u)
    G = run_bickley_jet.(u, Ref(32)) # error handling needs to go here
    G̅ = mean(G)

    # define covariances
    Cᵘᵖ = (u[1] - u̅) ⊗ (G[1] - G̅)
    Cᵖᵖ = (G[1] - G̅) ⊗ (G[1] - G̅)
    for j = 2:J
        Cᵘᵖ += (u[j] - u̅) ⊗ (G[j] - G̅)
        Cᵖᵖ += (G[j] - G̅) ⊗ (G[j] - G̅)
    end
    Cᵘᵖ *= 1 / (J - 1)
    Cᵖᵖ *= 1 / (J - 1)

    # ensemblize the data
    y = [ȳ + rand(ξ) for i = 1:J]
    r = y - G

    # update
    Cᵖᵖ_factorized = cholesky(Symmetric(Cᵖᵖ + 1 / h * Γ))
    for j = 1:J
        u[j] += Cᵘᵖ * (Cᵖᵖ_factorized \ r[j])
    end

    u = regularize.(u)
    @info "updated priors tstep $i"
    push!(timeseries, copy(u))
end



# function


using GLMakie, Printf

fig = Figure()
ax = Axis(fig[2, 1])
sc_init = scatter!(ax, [(timeseries[1][i][1], timeseries[1][i][2]) for i in eachindex(u)], color = :red)
sc_final = scatter!(ax, [(timeseries[end][i][1], timeseries[end][i][2]) for i in eachindex(u)], color = :blue)

time_slider = Slider(fig, range = 1:length(timeseries), startvalue = 1)
ti = time_slider.value

ensemble = @lift [(timeseries[$ti][i][1], timeseries[$ti][i][2]) for i in eachindex(u)]
sc_transition = scatter!(ax, ensemble, color = :purple)

scstar = scatter!(ax, [(exact[1], exact[2])], marker = '⋆', color = :yellow, markersize = 30)

ax.xlabel = "c¹"
ax.ylabel = "c²"

time_string = @lift("Time = " * @sprintf("%0.2f", ($ti - 1) / (length(timeseries) - 1)))
fig[3, 1] = vgrid!(
    Label(fig, time_string, width = nothing),
    time_slider,
)

c¹ = @lift([timeseries[$ti][i][1] for i in eachindex(timeseries[end])])
c² = @lift([timeseries[$ti][i][2] for i in eachindex(timeseries[end])])

ax.xlabelsize = 25
ax.ylabelsize = 25

ax_above = Axis(fig[1, 1])
ax_side = Axis(fig[2, 2])

ax_above.ylabel = "probability density"
ax_side.xlabel = "probability density"


hideydecorations!(ax_side, ticks = false, grid = false)
hidexdecorations!(ax_above, ticks = false, grid = false)

# colsize!(fig.layout, 1, Relative(2 / 3))
# rowsize!(fig.layout, 1, Relative(1 / 3))
colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)

update!(fig.scene)
display(fig)
##
seconds = 5
fps = 30
frames = round(Int, fps * seconds)
frames = lengt
fps = 30
record(fig, pwd() * "/example.mp4"; framerate = fps) do io
    for i = 1:frames
        ti[] = i
        sleep(1 / fps)
        recordframe!(io)
    end
end

