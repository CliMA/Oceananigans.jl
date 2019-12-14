using Plots, Printf

using Oceananigans
using Oceananigans.Diagnostics
using Oceananigans.AbstractOperations

using Oceananigans: Face, Cell, NoPenetrationBC

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

vbcs = ChannelBCs(top    = BoundaryCondition(Value, 1.0),
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
                            boundary_conditions=bcs, forcing=forcing)

nan_checker = NaNChecker(model; frequency=10, fields=Dict(:w => model.velocities.w))
push!(model.diagnostics, nan_checker)

u, v, w = model.velocities
ζ_op = ∂y(w) - ∂z(v)
ζ = Field(Face, Face, Cell, model.architecture, model.grid)
ζ_comp = Computation(ζ_op, ζ)


y = collect(model.grid.yC)
z = collect(model.grid.zC)
# p = heatmap(y, z, zeros(Ny, Nz), color=:viridis, show=true)

# Δt = 0.5e-4

wizard = TimeStepWizard(cfl=0.1, Δt=1e-6, max_change=1.1, max_Δt=1e-5)
cfl = AdvectiveCFL(wizard)

while model.clock.time < 4e-3
    t = model.clock.time

    update_Δt!(wizard, model)

    time_step!(model; Δt=wizard.Δt, Nt=10, init_with_euler = t == 0 ? true : false)

    compute!(ζ_comp)

    # heatmap!(p, y, z, interior(ζ)[1, :, :], color=:viridis, show=true)
    # heatmap!(p, y, z, interior(v)[1, :, :], color=:viridis, show=true)

    v_max = maximum(abs, interior(v))
    w_max = maximum(abs, interior(w))
    ζ_max = maximum(abs, interior(ζ))
    CFL = cfl(model)
    dCFL = (1/Re) * wizard.Δt / Δ^2

    @printf("Time: %1.2e, Δt: %1.2e CFL: %1.2e, dCFL: %1.2e, max (v, w, ζ): %1.2e, %1.2e, %1.2e\n",
            model.clock.time, wizard.Δt, CFL, dCFL, v_max, w_max, ζ_max)
end
