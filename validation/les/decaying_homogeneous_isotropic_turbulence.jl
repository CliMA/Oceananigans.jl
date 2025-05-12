using Oceananigans, FFTW, StatsBase, CairoMakie

using  Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!

arch = CPU()

grid = RectilinearGrid(arch, size = (64, 64, 64), extent = (1, 1, 1))

closure = AnisotropicMinimumDissipation(VerticallyImplicitTimeDiscretization(), C=1/3)

model = NonhydrostaticModel(; grid, closure)

u, v, w = model.velocities

set!(u, (args...)->randn())
set!(v, (args...)->randn())

compute_w_from_continuity!((; u, v, w), arch, grid)

function extract_energy(u, v, w, grid)
    û = fftshift(fft(@at (Center, Center, Center) u))
    v̂ = fftshift(fft(@at (Center, Center, Center) v))
    ŵ = fftshift(fft(@at (Center, Center, Center) w))

    Nx, Ny, Nz = u.grid.Nx, u.grid.Ny, u.grid.Nz

    kx = fftshift(fftfreq(Nx, 1/u.grid.Δxᶜᵃᵃ))
    ky = fftshift(fftfreq(Ny, 1/u.grid.Δyᵃᶜᵃ))
    kz = fftshift(fftfreq(Nz, 1/u.grid.z.Δᵃᵃᶜ))

    Nx, Ny, Nz = length(kx), length(ky), length(kz)

    KX = reshape(kx, Nx, 1, 1)
    KY = reshape(ky, 1, Ny, 1)
    KZ = reshape(kz, 1, 1, Nz)

    K = sqrt.(KX.^2 .+ KY.^2 .+ KZ.^2)

    E = abs.((û.^2 .+ v̂.^2 .+ ŵ.^2)./2)

    return E, K
end

fig = Figure()

ax = Axis(fig[1, 1], xscale=log, yscale=log, xlabel = "Wavenumber (1/m)", ylabel = "Energy density (m³/s²)")

k_bins = exp.(0:0.25:5)

xlims!(ax, minimum(k_bins), maximum(k_bins))
ylims!(ax, exp(-1), exp(25))

E, K = extract_energy(u, v, w, grid)

E_binned = fit(Histogram, [K...], weights([E...]), k_bins).weights

lines!(ax, (k_bins[1:end-1] .+ k_bins[2:end])./2, E_binned, color = 0, colorrange = (0, 10), colormap = :oslo)

simulation = Simulation(model, Δt = 0.5 * minimum_xspacing(grid) / 5, stop_time = 10)

add_callback!(simulation, (sim)->(@info "$(prettytime(time(sim))) in $(prettytime(sim.run_wall_time))"), IterationInterval(100))

function add_line!(sim)
    E, K = extract_energy(u, v, w, grid)
    E_binned = fit(Histogram, [K...], weights([E...]), k_bins).weights
    lines!(ax, (k_bins[1:end-1] .+ k_bins[2:end])./2, E_binned, color = time(sim), colorrange = (0, 10), colormap = :oslo)
end

add_callback!(simulation, add_line!, TimeInterval(0.1))

run!(simulation)

lines!(ax, [exp(1), exp(3)], x->(x/exp(1))^(-5/3)*exp(15), color = :red, linestyle = :dash)

text!(ax, exp(0.8), exp(13); text = L"$E(k)\sim k^{-5/3}$", color = :white) 

Colorbar(fig[1, 2], colormap = :oslo, colorrange = (0, 10), label = "Time (s)")

k_filt = 1/sqrt(closure.Cν * 3 / (1/(2*minimum_xspacing(grid))^2 + 1/(2*minimum_yspacing(grid))^2 + 1/(2*minimum_zspacing(grid))^2))

lines!(ax, ones(2) .* k_filt, [exp(0), exp(10)], color = :red, linestyle = :dash)

text!(ax, k_filt, exp(10); text = "1/δ")

save("energy.png", fig)