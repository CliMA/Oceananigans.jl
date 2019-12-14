using Oceananigans
using Plots, Printf

using Oceananigans: NoPenetrationBC
using Oceananigans.Diagnostics: AdvectiveCFL

# Workaround for plotting many frames.
# See: https://github.com/JuliaPlots/Plots.jl/issues/1723
# import GR
# GR.inline("png")

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

vbcs = ChannelBCs(top    = BoundaryCondition(Value, 0.0),
                  bottom = BoundaryCondition(Value, 0.0),
                  north  = NoPenetrationBC(),
                  south  = NoPenetrationBC())

wbcs = ChannelBCs(top    = NoPenetrationBC(),
                  bottom = NoPenetrationBC(),
                  north  = BoundaryCondition(Value, 0.0),
                  south  = BoundaryCondition(Value, 0.0))

bcs = ChannelSolutionBCs(v=vbcs, w=wbcs)

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz))

Re = 100
model = NonDimensionalModel(grid=grid, Re=Re, Pr=Inf, Ro=Inf,
                            coriolis=nothing, tracers=nothing, buoyancy=nothing,
                            boundary_conditions=bcs)

v₀(x, y, z) = -1 + 2z
w₀(x, y, z) = 1 - 2y
set!(model, v=v₀, w=w₀)

Δ = max(model.grid.Δy, model.grid.Δz)

y = collect(model.grid.yC)
z = collect(model.grid.zC)
# p = heatmap(y, z, zeros(Ny, Nz), color=:viridis, show=true)

# Δt = 0.5e-4

wizard = TimeStepWizard(cfl=0.01, Δt=1e-7, max_change=1.1, max_Δt=1e-6)
cfl = AdvectiveCFL(wizard)

while model.clock.time < 4e-4
    update_Δt!(wizard, model)

    time_step!(model; Δt=wizard.Δt, Nt=10, init_with_euler=model.clock.time == 0 ? true : false)

    v = model.velocities.v.data[1, :, :]
    w = model.velocities.w.data[1, :, :]

    Δy, Δz = model.grid.Δy, model.grid.Δz
    dvdz = (v[1:Ny, 2:Nz+1] - v[1:Ny, 1:Nz]) / Δz
    dwdy = (w[2:Ny+1, 1:Nz] - w[1:Ny, 1:Nz]) / Δy
    ζ = dwdy - dvdz
    ζ = log10.(abs.(ζ))

    u, v, w = model.velocities

    # heatmap!(p, y, z, ζ, color=:viridis, show=true)
    # heatmap!(p, y, z, interior(v)[1, :, :], color=:viridis, show=true)

    v_max = maximum(abs, interior(v))
    w_max = maximum(abs, interior(w))
    CFL = cfl(model)
    dCFL = (1/Re) * wizard.Δt / Δ^2
    @printf("Time: %1.2e, Δt: %1.2e CFL: %1.2e, dCFL: %1.2e, max (v, w, ζ): %1.2e, %1.2e, %1.2e\n",
            model.clock.time, wizard.Δt, CFL, dCFL, v_max, w_max, 10^maximum(ζ))
    # @printf("(u, v) @ corner: %.2e, %.2e\n", )
end
