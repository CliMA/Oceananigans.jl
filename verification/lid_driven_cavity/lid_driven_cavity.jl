using Oceananigans, PyPlot, Printf
using Oceananigans: NoPenetrationBC, NonDimensionalModel

Nx, Ny, Nz = 1, 128, 128
Lx, Ly, Lz = 1, 1, 1

vbcs = ChannelBCs(top    = BoundaryCondition(Value, 1),
                  bottom = BoundaryCondition(Value, 0),
                  north  = NoPenetrationBC(),
                  south  = NoPenetrationBC())

wbcs = ChannelBCs(top    = NoPenetrationBC(),
                  bottom = NoPenetrationBC(),
                  north  = BoundaryCondition(Value, 0),
                  south  = BoundaryCondition(Value, 0))

bcs = ChannelSolutionBCs(v=vbcs, w=wbcs)

Re = 400
model = NonDimensionalModel(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), Re=Re, Ri=0, Pr=Inf, Ro=Inf, boundary_conditions=bcs)

fig, ax = subplots(nrows=1, ncols=1, figsize=(10, 10))

Δt = 2e-4
Δ = max(model.grid.Δy, model.grid.Δz)
CFL = Δt / Δ
dCFL = (1/Re) * Δt / Δ^2
@show CFL
@show dCFL

# while model.clock.time < 1
    time_step!(model; Δt=Δt, Nt=100, init_with_euler=model.clock.time == 0 ? true : false)
    @printf("Time: %.4f\n", model.clock.time)

    y = collect(model.grid.yC)
    z = collect(model.grid.zC)
    v = model.velocities.v.data[1, :, :]
    w = model.velocities.w.data[1, :, :]

    Δy, Δz = model.grid.Δy, model.grid.Δz
    dvdz = (v[1:Ny, 2:Nz+1] - v[1:Ny, 1:Nz])/ Δz
    dwdy = (w[1:Ny, 1:Nz] - w[2:Ny+1, 1:Nz])/ Δy
    ζ = dwdy - dvdz
    ζ = reverse(log10.(abs.(ζ)), dims=1)

    # ax.streamplot(y, z, v, w)
    pcolormesh(y, z, ζ, vmin=1e-3, cmap="viridis"); title(); xlabel("y"); ylabel("z");
    # fig.colorbar(im, ax=ax)

    ax.set_title(@sprintf("Lid-driven cavity vorticity log₁₀(ζ): Re=%d, t=%.4f", Re, model.clock.time))
    ax.set_xlabel("\$y\$"); ax.set_ylabel("\$z\$");
    ax.set_xlim([0, 1]); ax.set_ylim([-1, 0]);
    ax.set_aspect(1)
    gcf();
# end
