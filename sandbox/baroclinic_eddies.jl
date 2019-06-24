using Oceananigans, PyPlot, Statistics

include("../examples/utils.jl")

function update_plot!(ax, model)
    xC, yC, zC = model.grid.xC / 1000, model.grid.yC / 1000, model.grid.zC
    T = model.tracers.T

    sca(ax[1, 1])
    PyPlot.contourf(xC, yC, data(T)[:, :, 1]', vmin=7.5, vmax=15, cmap=:inferno, levels=40)
    PyPlot.xlim(0, 160)
    PyPlot.ylim(0, 500)
    PyPlot.xlabel("x (km)")
    PyPlot.ylabel("y (km)")
    # PyPlot.colorbar()

    t = Int(round(model.clock.time / 3600))
    PyPlot.title("surface (t = $t hours)")

    sca(ax[2, 1])
    PyPlot.contourf(yC, zC, data(T)[Int(Nx/2), :, :]', vmin=7.5, vmax=15, cmap=:inferno, levels=20)
    PyPlot.xlim(0, 500)
    PyPlot.ylim(-1000, 0)
    PyPlot.xlabel("x (km)")
    PyPlot.ylabel("z (m)")
    PyPlot.title("x = 80 km")
    # PyPlot.colorbar()

    sca(ax[3, 1])
    PyPlot.contourf(xC, zC, data(T)[:, Int(Ny/2), :]', vmin=7.5, vmax=15, cmap=:inferno, levels=20)
    PyPlot.xlim(0, 160)
    PyPlot.ylim(-1000, 0)
    PyPlot.xlabel("x (km)")
    PyPlot.ylabel("z (m)")
    PyPlot.title("y = 250 km")
    # PyPlot.colorbar()

    PyPlot.draw()
end

Lx, Ly, Lz = 160e3, 500e3, 1e3  # 160×500×1 km
Δh, Δz = 5e3, 50  # Horizontal and vertical grid spacing [m].
Nx, Ny, Nz = Int(Lx/Δh), Int(Ly/Δh), Int(Lz/Δz)

α = Δz/Δh # Grid cell aspect ratio.
νh, κh = 20.0, 20.0
νv, κv = α*νh, α*κh

@show Nx, Ny, Nz
@show νh, κh, νv, κv

model = ChannelModel(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz),
                       νh=νh, νv=νv, κh=κh, κv=κv)

nan_checker = NaNChecker(10, [model.velocities.w, model.tracers.T], ["w", "T"])
push!(model.diagnostics, nan_checker)

Ty = 1e-4  # Meridional temperature gradient [K/m].
Tz = 5e-3  # Vertical temperature gradient [K/m].
T₀(x, y, z) = 10 + Ty*min(max(0, y-225e3), 50e3) + Tz*z + 0.0001*rand()  # Initial temperature field [°C].

set_ic!(model, T=T₀)

fig, ax = subplots(ncols=3, nrows=1, figsize=(21, 7))
update_plot!(ax, model)

for n in 1:1000
    i, t = model.clock.iteration, model.clock.time
    Tavg = mean(data(model.tracers.T))
    @info "i = $i, t = $(Int(round(t/3600))) hours, ⟨T⟩=$Tavg °C"

    update_plot!(ax, model)
    time_step!(model; Nt=100, Δt=5*60)
end
