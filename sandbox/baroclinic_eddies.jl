using Oceananigans, PyPlot

include("../examples/utils.jl")

function update_plot!(ax, model)
    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC
    T = model.tracers.T

    sca(ax[1, 1])
    PyPlot.pcolormesh(xC, yC, data(T)[:, :, 1])

    sca(axs[2, 1])
    PyPlot.pcolormesh(xC, zC, data(T)[:, 10, :])
end

Lx, Ly, Lz = 160e3, 500e3, 1e3  # 160×500×1 km
Δh, Δz = 10e3, 50  # Horizontal and vertical grid spacing [m].
Nx, Ny, Nz = Int(Lx/Δh), Int(Ly/Δh), Int(Lz/Δz)

α = Δz/Δh # Grid cell aspect ratio.
νh, κh = 20, 20
νv, κv = α*νh, α*κh

@show Nx, Ny, Nz
@show νh, κh, νv, κv

model = ChannelModel(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz),
                       νh=νh, νv=νv, κh=κh, κv=κv)

Ty = 1e-5  # Meridional temperature gradient [K/m].
Tz = 5e-3  # Vertical temperature gradient [K/m].
T₀(x, y, z) = 15 + Ty*y * Tz*z  # Initial temperature field [°C].

set_ic!(model, T=T₀)

fig, ax = subplots(ncols=2, nrows=1, figsize=(16, 9))

for n in 1:100
    update_plot!(ax, model)
    time_step!(model; Nt=100, Δt=60)
end
