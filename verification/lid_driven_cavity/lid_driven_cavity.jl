using Oceananigans, PyPlot, Printf
using Oceananigans: NoPenetrationBC, NonDimensionalModel

####
#### Data from tables 1 and 2 of Ghia et al. (1982).
####

j̃ = [1,   8,      9,      10,     14,     23,     37,     59,     65,  80,     95,     110,    123,    124,    156,    126,    129]
ỹ = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]

ũ = Dict(
    100 => [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0],
    400 => [0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,  0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0]
)

####
#### Model setup
####

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

nan_checker = NaNChecker(model; frequency=10, fields=Dict(:w => model.velocities.w))
push!(model.diagnostics, nan_checker)

fig, ax = subplots(nrows=1, ncols=1, figsize=(10, 10))

Δt = 0.5e-4
Δ = max(model.grid.Δy, model.grid.Δz)
CFL = Δt / Δ
dCFL = (1/Re) * Δt / Δ^2
@show CFL
@show dCFL

while model.clock.time < 5
    # Δt = model.clock.time < 0.3 ? 0.5e-4 : 5e-4
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
    pcolormesh(y, z, ζ, vmin=1e-3, cmap="viridis")
    # fig.colorbar(im, ax=ax)

    ax.set_title(@sprintf("Lid-driven cavity vorticity log₁₀(ζ): Re=%d, t=%.4f", Re, model.clock.time))
    ax.set_xlabel("\$y\$"); ax.set_ylabel("\$z\$");
    ax.set_xlim([0, 1]); ax.set_ylim([-1, 0]);
    ax.set_aspect(1)
    gcf();
end
