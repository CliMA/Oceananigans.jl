"""
This example sets up a cold bubble perturbation which develops into a non-linear
density current. This numerical test case is described by Straka et al. (1993).
Also see: http://www2.mmm.ucar.edu/projects/srnwp_tests/density/density.html

Straka et al. (1993). "Numerical Solutions of a Nonlinear Density-Current -
    A Benchmark Solution and Comparisons." International Journal for Numerical
    Methods in Fluids 17, pp. 1-22.
"""

using Printf
using Plots
using VideoIO
using FileIO
using JULES
using Oceananigans

using Oceananigans.Fields: interiorparent
interiorxz(field) = dropdims(interiorparent(field), dims=2)

const km = 1000.0
const hPa = 100.0

Lx = 51.2km
Lz = 6.4km

Δ = 100.0  # grid spacing [m]

Nx = Int(Lx/Δ)
Ny = 1
Nz = Int(Lz/Δ)

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), halo=(2, 2, 2),
                            x=(-Lx/2, Lx/2), y=(-Lx/2, Lx/2), z=(0, Lz))
tvar = Energy()
# tvar = Entropy()

model = CompressibleModel(
                      grid = grid,
                     gases = DryEarth(),
    thermodynamic_variable = tvar,
                   closure = IsotropicDiffusivity(ν=75.0, κ=75.0)
)

#####
##### Initial perturbation
#####

gas = model.gases.ρ
R, cₚ, cᵥ = gas.R, gas.cₚ, gas.cᵥ
g  = model.gravity
pₛ = 1000hPa
Tₛ = 300.0

# Define an approximately hydrostatic background state
θ₀(x, y, z) = Tₛ
p₀(x, y, z) = pₛ * (1 - g*z / (cₚ*Tₛ))^(cₚ/R)
T₀(x, y, z) = Tₛ * (p₀(x, y, z)/pₛ)^(R/cₚ)
ρ₀(x, y, z) = p₀(x, y, z) / (R*T₀(x, y, z))

# Define both energy and entropy
uᵣ, Tᵣ, ρᵣ, sᵣ = gas.u₀, gas.T₀, gas.ρ₀, gas.s₀  # Reference values
ρe₀(x, y, z) = ρ₀(x, y, z) * (uᵣ + cᵥ * (T₀(x, y, z) - Tᵣ) + g*z)
ρs₀(x, y, z) = ρ₀(x, y, z) * (sᵣ + cᵥ * log(T₀(x, y, z)/Tᵣ) - R * log(ρ₀(x, y, z)/ρᵣ))

# Define the initial density perturbation
xᶜ, zᶜ = 0km, 2km
xʳ, zʳ = 2km, 2km
L(x, y, z) = sqrt(((x - xᶜ)/xʳ)^2 + ((z - zᶜ)/zʳ)^2)

function ρ′(x, y, z; θᶜ′ = -15.0)
    l = L(x, y, z)
    θ′ = (l <= 1) * θᶜ′ * (1 + cos(π*l))/2
    return -ρ₀(x, y, z) * θ′ / θ₀(x, y, z)
end

# Define initial state
ρᵢ(x, y, z) = ρ₀(x, y, z) + ρ′(x, y, z)
pᵢ(x, y, z) = p₀(x, y, z)
Tᵢ(x, y, z) = pᵢ(x, y, z) / (R * ρᵢ(x, y, z))

ρeᵢ(x, y, z) = ρᵢ(x, y, z) * (uᵣ + cᵥ * (Tᵢ(x, y, z) - Tᵣ) + g*z)
ρsᵢ(x, y, z) = ρᵢ(x, y, z) * (sᵣ + cᵥ * log(Tᵢ(x, y, z)/Tᵣ) - R * log(ρᵢ(x, y, z)/ρᵣ))

# Set hydrostatic background state
set!(model.tracers.ρ, ρ₀)
tvar isa Energy  && set!(model.tracers.ρe, ρe₀)
tvar isa Entropy && set!(model.tracers.ρs, ρs₀)
update_total_density!(model)

# Save hydrostatic base state
ρʰᵈ = interiorxz(model.total_density)
tvar isa Energy  && (ρeʰᵈ = interiorxz(model.tracers.ρe))
tvar isa Entropy && (ρsʰᵈ = interiorxz(model.tracers.ρs))

# Set initial state (which includes the thermal perturbation)
set!(model.tracers.ρ, ρᵢ)
tvar isa Energy  && set!(model.tracers.ρe, ρeᵢ)
tvar isa Entropy && set!(model.tracers.ρs, ρsᵢ)
update_total_density!(model)

ρ_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km,
                 rotr90(interiorxz(model.total_density) .- ρʰᵈ),
                 fill=true, levels=10, ylims=(0, 6.4), clims=(-0.05, 0.05),
                 color=:balance, aspect_ratio=:equal, dpi=200)
savefig(ρ_plot, "rho_prime_initial_condition_with_$(typeof(tvar)).png")

if tvar isa Energy
    e_slice = rotr90(interiorxz(model.tracers.ρe) ./ interiorxz(model.total_density))
    e_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, e_slice,
                     fill=true, levels=10, ylims=(0, 6.4), color=:thermal,
                     aspect_ratio=:equal, dpi=200)
    savefig(e_plot, "energy_initial_condition.png")
elseif tvar isa Entropy
    s_slice = rotr90(interiorxz(model.tracers.ρs) ./ interiorxz(model.total_density))
    s_plot = contour(model.grid.xC ./ km, model.grid.zC ./ km, s_slice,
                     fill=true, levels=10, ylims=(0, 6.4), color=:thermal,
                     aspect_ratio=:equal, dpi=200)
    savefig(s_plot, "entropy_initial_condition.png")
end

#####
##### Watch the density current evolve!
#####

for n = 1:180
    @printf("t = %.2f s\n", model.clock.time)
    time_step!(model, Δt=0.1, Nt=50)

    xC, yC, zC = model.grid.xC ./ km, model.grid.yC ./ km, model.grid.zC ./ km
    xF, yF, zF = model.grid.xF ./ km, model.grid.yF ./ km, model.grid.zF ./ km

    u_slice = rotr90(interiorxz(model.momenta.ρu) ./ interiorxz(model.total_density))
    w_slice = rotr90(interiorxz(model.momenta.ρw) ./ interiorxz(model.total_density))
    ρ_slice = rotr90(interiorxz(model.total_density) .- ρʰᵈ)

    u_title = @sprintf("u, t = %d s", round(Int, model.clock.time))
    u_plot = heatmap(xC, zC, u_slice, title=u_title, fill=true, levels=10, color=:balance,
                     clims=(-20, 20), linewidth=0, xticks=nothing, titlefontsize=10)
    w_plot = heatmap(xC, zC, w_slice, title="w", fill=true, levels=10, color=:balance,
                     clims=(-20, 20), linewidth=0, xticks=nothing, titlefontsize=10)
    ρ_plot = heatmap(xC, zC, ρ_slice, title="rho_prime", fill=true, levels=10, color=:balance,
                     clims=(-0.05, 0.05), linewidth=0, xticks=nothing, titlefontsize=10)

    if tvar isa Energy
        e_slice = rotr90((interiorxz(model.tracers.ρe) .- ρeʰᵈ) ./ interiorxz(model.total_density))
        tvar_plot = heatmap(xC, zC, e_slice, title="e_prime", fill=true, levels=10, color=:oxy,
                            clims=(-8000, 0), linewidth=0, titlefontsize=10)
    elseif tvar isa Entropy
        s_slice = rotr90(interiorxz(model.tracers.ρs) ./ interiorxz(model.total_density))
        tvar_plot = heatmap(xC, zC, s_slice, title="s", fill=true, levels=10, color=:oxy,
                            clims=(45, 94), linewidth=0, titlefontsize=10)
    end

    p = plot(u_plot, w_plot, ρ_plot, tvar_plot, layout=(4, 1), dpi=300, show=true)
    n == 1 && !isdir("frames") && mkdir("frames")
    savefig(p, @sprintf("frames/density_current_%s_%03d.png", typeof(tvar), n))
end

# Print min/max of ρ′ and w at t = 900.
ρ′₉₀₀ = (interiorxz(model.tracers.ρ) .- ρʰᵈ)
w₉₀₀  = (interiorxz(model.momenta.ρw) ./ interiorxz(model.tracers.ρ))

@printf("ρ′: min=%.2e, max=%.2e\n", minimum(ρ′₉₀₀), maximum(ρ′₉₀₀))
@printf("w:  min=%.2e, max=%.2e\n", minimum(w₉₀₀), maximum(w₉₀₀))

@printf("Rendering MP4...\n")
imgs = filter(x -> occursin("$(typeof(tvar))", x) && occursin(".png", x), readdir("frames"))
imgorder = map(x -> split(split(x, ".")[1], "_")[end], imgs)
p = sortperm(parse.(Int, imgorder))

frames = []
for img in imgs[p]
    push!(frames, convert.(RGB, load("frames/$img")))
end

encodevideo("density_current_$(typeof(tvar)).mp4", frames, framerate = 30)
