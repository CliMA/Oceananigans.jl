"""
This example sets up a dry, warm thermal bubble perturbation in a uniform
lateral mean flow which buoyantly rises. Identical to the verification
experiment in ../dry_rising_thermal_bubble except that entropy is used as
the prognostic thermodynamic variable rather than potential temperature.
"""

using Printf
using Profile
using Plots
using VideoIO
using FileIO
using Oceananigans
using JULES

const km = 1000
const hPa = 100

Lx = 20km
Lz = 10km

Δ = 0.2km  # grid spacing [m]

Nx = Int(Lx/Δ)
Ny = 1
Nz = Int(Lz/Δ)

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(2, 2, 2),
                            x=(-Lx/2, Lx/2), y=(-Lx/2, Lx/2), z=(0, Lz))

model = CompressibleModel(
                      grid = grid,
                 gases = DryEarth(),
    thermodynamic_variable = Entropy(),
                   closure = ConstantIsotropicDiffusivity(ν=75.0, κ=75.0)
)

#####
##### Dry thermal bubble perturbation
#####

gas = model.densities.ρ
R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
sref, Tref, ρref = gas.s₀, gas.T₀, gas.ρ₀
g  = model.gravity
pₛ = 1000hPa
Tₛ = 300

# Define an approximately hydrostatic background state
θ₀(x, y, z) = Tₛ
p₀(x, y, z) = pₛ * (1 - g*z/(cₚ*Tₛ))^(cₚ/R)
T₀(x, y, z) = Tₛ*(p₀(x, y, z)/pₛ)^(R/cₚ)
ρ₀(x, y, z) = p₀(x, y, z)/(R*T₀(x, y, z))
ρs₀(x, y, z) = ρ₀(x, y, z) * (sref + cᵥ*log(T₀(x, y, z)/Tref) - R*log(ρ₀(x, y, z)/ρref))

# Define the initial density perturbation
xᶜ, zᶜ = 0km, 2km
xʳ, zʳ = 2km, 2km
L(x, y, z) = sqrt(((x - xᶜ)/xʳ)^2 + ((z - zᶜ)/zʳ)^2)
function ρ′(x, y, z; θᶜ′ = 2.0)
    l = L(x, y, z)
    θ′ = (l <= 1) * θᶜ′ * cos(π/2 * L(x, y, z))^2
    return -ρ₀(x, y, z) * θ′ / θ₀(x, y, z)
end

# Define initial state
ρᵢ(x, y, z) = ρ₀(x, y, z) + ρ′(x, y, z)
pᵢ(x, y, z) = p₀(x, y, z)
Tᵢ(x, y, z) = pᵢ(x, y, z) / (R * ρᵢ(x, y, z))
ρsᵢ(x, y, z) = ρᵢ(x, y, z) * (sref + cᵥ*log(Tᵢ(x, y, z)/Tref) - R*log(ρᵢ(x, y, z)/ρref))

# Set initial state after saving perturbation-free background
ρ, ρs = model.total_density, model.tracers.ρs
xC, zC = grid.xC, grid.zC
set!(model.tracers.ρ, ρ₀)
set!(model.tracers.ρs, ρs₀)
update_total_density!(model.total_density, model.grid, model.densities, model.tracers)
ρʰᵈ = ρ.data[1:Nx, 1, 1:Nz]
ρsʰᵈ = ρs.data[1:Nx, 1, 1:Nz]
set!(model.tracers.ρ, ρᵢ)
set!(model.tracers.ρs, ρsᵢ)
update_total_density!(model.total_density, model.grid, model.densities, model.tracers)

ρ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km,
    rotr90(ρ.data[1:Nx, 1, 1:Nz] .- ρʰᵈ), fill=true, levels=10, xlims=(-5, 5),
    clims=(-0.008, 0.008), color=:balance, dpi=200)
savefig(ρ_plot, "rho_prime_initial_condition.png")

s_slice = rotr90(ρs.data[1:Nx, 1, 1:Nz] ./ ρ.data[1:Nx, 1, 1:Nz])
s_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, s_slice,
                 fill=true, levels=10, xlims=(-5, 5), color=:thermal, dpi=200)
savefig(s_plot, "entropy_initial_condition.png")

#####
##### Watch the thermal bubble rise!
#####

ρ̄ᵢ = sum(ρ.data[1:Nx,1,1:Nz])/(Nx*Nz)
ρ̄s̄ᵢ = sum(ρs.data[1:Nx,1,1:Nz])/(Nx*Nz)
Δt=0.1
for n in 1:200

    time_step!(model, Δt = Δt, Nt = 50)

    CFL = cfl(model, Δt)
    ρ̄ = sum(ρ.data[1:Nx,1:Ny,1:Nz])/(Nx*Ny*Nz)
    ρ̄s̄ = sum(ρs.data[1:Nx,1:Ny,1:Nz])/(Nx*Ny*Nz)
    @printf("t = %.2f s, CFL = %.2e, ρ̄ = %.2e (rerr = %.2e), ρ̄s̄ = %.2e (rerr = %.2e)\n",
        model.clock.time, CFL, ρ̄, (ρ̄ - ρ̄ᵢ)/ρ̄, ρ̄s̄, (ρ̄s̄ - ρ̄s̄ᵢ)/ρ̄s̄)

    xC, yC, zC = model.grid.xC ./ km, model.grid.yC ./ km, model.grid.zC ./ km
    xF, yF, zF = model.grid.xF ./ km, model.grid.yF ./ km, model.grid.zF ./ km

    j = 1
    u_slice = rotr90(model.momenta.ρu.data[1:Nx, j, 1:Nz] ./ model.tracers.ρ.data[1:Nx, j, 1:Nz])
    w_slice = rotr90(model.momenta.ρw.data[1:Nx, j, 1:Nz] ./ model.tracers.ρ.data[1:Nx, j, 1:Nz])
    ρ_slice = rotr90(model.tracers.ρ.data[1:Nx, j, 1:Nz] .- ρʰᵈ)
    s_slice = rotr90(model.tracers.ρs.data[1:Nx, j, 1:Nz] ./ model.tracers.ρ.data[1:Nx, j, 1:Nz])

    u_title = @sprintf("u, t = %d s", round(Int, model.clock.time))
    pu = heatmap(xC, zC, u_slice, title=u_title, fill=true, levels=50,
        xlims=(-5, 5), color=:balance, linecolor = nothing, clims=(-10, 10))
    pw = heatmap(xC, zC, w_slice, title="w", fill=true, levels=50,
        xlims=(-5, 5), color=:balance, linecolor = nothing, clims=(-10, 10))
    pρ = heatmap(xC, zC, ρ_slice, title="rho_prime", fill=true, levels=50,
        xlims=(-5, 5), color=:balance, linecolor = nothing, clims=(-0.006, 0.006))
    ps = heatmap(xC, zC, s_slice, title="s", fill=true, levels=50,
        xlims=(-5, 5), color=:oxy_r, linecolor = nothing, clims=(99, 105))

    p = plot(pu, pw, pρ, ps, layout=(2, 2), dpi=200, show=true)
    savefig(p, @sprintf("frames/thermal_bubble_%03d.png", n))
end

ρ′_1000 = (model.tracers.ρ.data[1:Nx, 1, 1:Nz] .- ρʰᵈ)
w_1000 = (model.momenta.ρw.data[1:Nx, 1, 1:Nz] ./ model.tracers.ρ.data[1:Nx, 1, 1:Nz])

@printf("ρ′: min=%.2f, max=%.2f\n", minimum(ρ′_1000), maximum(ρ′_1000))
@printf("w:  min=%.2f, max=%.2f\n", minimum(w_1000), maximum(w_1000))

@printf("Rendering MP4\n")
imgs = filter(x -> occursin(".png", x), readdir("frames"))
imgorder = map(x -> split(split(x, ".")[1], "_")[end], imgs)
p = sortperm(parse.(Int, imgorder))
frames = []
for img in imgs[p]
    push!(frames, convert.(RGB, load("frames/$img")))
end
encodevideo("thermal_bubble.mp4", frames, framerate = 30)
