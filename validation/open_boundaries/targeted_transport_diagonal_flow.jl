# Uniform diagonal flow through four targeted open boundaries, `NormalRadiation` vs `PerturbationAdvection`. The exact
# solution is u = v = U with a flat pressure. Under RK3 `NormalRadiation` misses it: lacking an interior normal gradient,
# its phase speed flips along the outflow faces and the stage anchoring imprints a non-uniform profile. AB2 is uniform.

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: NormalRadiation, PerturbationAdvection
using CairoMakie
using Printf
using Statistics: mean

Lx = Ly = 1kilometers
Nx = Ny = 64
Δt = 20
stop_time = 2hours
ν = 0.5                       # viscosity [m² s⁻¹]
U = 0.1                       # speed through each side [m s⁻¹]
Q = U * Ly                    # transport per unit depth through each side [m² s⁻¹]
timestepper = :RungeKutta3    # :QuasiAdamsBashforth2 makes both schemes uniform

grid = RectilinearGrid(CPU(); size = (Nx, Ny), x = (0, Lx), y = (0, Ly), halo = (4, 4),
                       topology = (Bounded, Bounded, Flat))

function run_diagonal_flow(Scheme)
    normal() = NormalFlowBoundaryCondition(0; scheme = Scheme(; inflow_timescale = Inf, outflow_timescale = Inf,
                                                              target_transport = Q))
    boundary_conditions = (u = FieldBoundaryConditions(west = normal(), east = normal()),
                           v = FieldBoundaryConditions(south = normal(), north = normal()))

    model = NonhydrostaticModel(grid; boundary_conditions, advection = WENO(order = 5),
                                closure = ScalarDiffusivity(; ν), timestepper)

    run!(Simulation(model; Δt, stop_time, verbose = false))
    return model
end

models = (NormalRadiation = run_diagonal_flow(NormalRadiation),
          PerturbationAdvection = run_diagonal_flow(PerturbationAdvection))

function total_pressure(model)
    pHY′, pNHS = model.pressures.pHY′, model.pressures.pNHS
    isnothing(pHY′) && return pNHS
    p = Field(pHY′ + pNHS)
    compute!(p)
    return p
end

spread(f) = (maximum(f) - minimum(f)) / U

for (name, model) in pairs(models)
    u, v, _ = model.velocities
    @printf("%-22s spread of the normal velocity along the outflow faces, in units of U: east %.1e, north %.1e\n",
            name, spread(interior(u, Nx + 1, :, 1)), spread(interior(v, :, Ny + 1, 1)))
end

#####
##### Figure: rows are schemes, columns are speed with velocity vectors and total pressure anomaly
#####

xc = xnodes(grid, Center()) / kilometers
yc = ynodes(grid, Center()) / kilometers
stride = 4
xa, ya = xc[1:stride:end], yc[1:stride:end]
arrow_lengthscale = 0.5 * stride * (Lx / kilometers) / Nx / U   # a speed U spans half the arrow spacing
limits = (0, Lx / kilometers, 0, Ly / kilometers)

fields = map(models) do model
    u, v, _ = model.velocities
    uᶜ = (interior(u, 1:Nx, :, 1) .+ interior(u, 2:Nx+1, :, 1)) ./ 2
    vᶜ = (interior(v, :, 1:Ny, 1) .+ interior(v, :, 2:Ny+1, 1)) ./ 2
    p = interior(total_pressure(model), :, :, 1)
    (; uᶜ, vᶜ, speed = sqrt.(uᶜ.^2 .+ vᶜ.^2), p′ = p .- mean(p))
end

speed_range = U * sqrt(2) .* (0.9, 1.1)                          # the exact solution is uniform at U√2
p′_max = max(maximum(maximum(abs, f.p′) for f in fields), 1e-12)   # shared, so a flat field shows flat

fig = Figure(size = (1150, 950))
Label(fig[0, 1:4], @sprintf("Uniform diagonal flow through four targeted open boundaries, t = %.0f h", stop_time / hours);
      fontsize = 18, font = :bold)

for (row, (name, f)) in enumerate(pairs(fields))
    ax = Axis(fig[row, 1]; title = "$name: speed (exact: uniform U√2)", xlabel = "x (km)", ylabel = "y (km)",
              aspect = DataAspect(), limits)
    hm = heatmap!(ax, xc, yc, f.speed; colormap = :viridis, colorrange = speed_range)
    arrows2d!(ax, xa, ya, f.uᶜ[1:stride:end, 1:stride:end], f.vᶜ[1:stride:end, 1:stride:end];
              lengthscale = arrow_lengthscale, color = :white)
    Colorbar(fig[row, 2], hm; label = "m s⁻¹")

    ax = Axis(fig[row, 3]; title = "$name: total pressure anomaly (exact: flat)", xlabel = "x (km)", ylabel = "y (km)",
              aspect = DataAspect(), limits)
    hm = heatmap!(ax, xc, yc, f.p′; colormap = :balance, colorrange = (-p′_max, p′_max))
    Colorbar(fig[row, 4], hm; label = "m² s⁻²")
end

save("targeted_transport_diagonal_flow.png", fig)
@info "Saved targeted_transport_diagonal_flow.png"
