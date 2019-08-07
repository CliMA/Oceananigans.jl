using Statistics, Printf
using CuArrays, JLD2, FileIO
using Oceananigans

Lx, Ly, Lz = 250e3, 500e3, 1000  # 160×512×1 km
Nx, Ny, Nz = 256, 512, 128

Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

α = Δz/Δx # Grid cell aspect ratio.
νh, κh = 0.25, 0.25
νv, κv = α*νh, α*κh

@show Nx, Ny, Nz
@show Δx, Δy, Δz
@show νh, κh, νv, κv

model = ChannelModel(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=GPU(),
                     νh=νh, νv=νv, κh=κh, κv=κv)

Ty = 4e-5  # Meridional temperature gradient [K/m].
Tz = 2e-3  # Vertical temperature gradient [K/m].

# Initial temperature field [°C].
T₀(x, y, z) = 10 + Ty*min(max(0, y-225e3), 50e3) + Tz*z + 0.0001*rand()

xs = reshape(model.grid.xC, Nx, 1, 1)
ys = reshape(model.grid.yC, 1, Ny, 1)
zs = reshape(model.grid.zC, 1, 1, Nz)

T0 = T₀.(xs, ys, zs)
T0 = CuArray(T0)
ardata_view(model.tracers.T) .= T0

nan_checker = NaNChecker(1000, [model.tracers.T], ["T"])
push!(model.diagnostics, nan_checker)

Δt_wizard = TimeStepWizard(cfl=0.15, Δt=30.0, max_change=1.5, max_Δt=300.0)

Ni = 40
for n in 1:100000
    walltime = @elapsed time_step!(model; Nt=Ni, Δt=Δt_wizard.Δt)

    umax = maximum(abs, model.velocities.u.data.parent)
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)
    CFL = Δt_wizard.Δt / cell_advection_timescale(model)

    update_Δt!(Δt_wizard, model)

    filename = "eddying_channel_" * string(model.clock.iteration) * ".jld2"
    io_time = @elapsed save(filename,
        Dict("t" => model.clock.time,
        "u" => Array(model.velocities.u.data.parent),
        "v" => Array(model.velocities.v.data.parent),
        "w" => Array(model.velocities.w.data.parent),
        "T" => Array(model.tracers.T.data.parent)))

    @info @sprintf(
        "i: %d, t: %.3f days, umax: (%6.3g, %6.3g, %6.3g) m/s, CFL: %.4f, next Δt: %3.2f s, ⟨wall time⟩: %s, IO time: %s\n",
        model.clock.iteration, model.clock.time / 86400, umax, vmax, wmax, CFL, Δt_wizard.Δt,
        prettytime(1e9*walltime / Ni), prettytime(1e9*io_time))
end
