using Printf
using Oceananigans
using Oceananigans: Face, Cell
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations

#####
##### Data from tables 1 and 2 of Ghia et al. (1982).
#####

j̃ = [1,   8,      9,      10,     14,     23,     37,     59,     65,  80,     95,     110,    123,    124,    156,    126,    129]
ỹ = [0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]

ũ = Dict(
    100 => [0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0],
    400 => [0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477,  0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0]
)

#####
##### Model setup
#####

Ny = Nz = 128
Ly = Lz = 1.0

topology = (Flat, Bounded, Bounded)
domain = (x=(0, 1), y=(0, Ly), z=(0, Lz))
grid = RegularCartesianGrid(topology=topology, size=(1, Ny, Nz); domain...)

v_bcs = VVelocityBoundaryConditions(grid,
       top = ValueBoundaryCondition(1.0),
    bottom = ValueBoundaryCondition(0.0)
)

w_bcs = WVelocityBoundaryConditions(grid,
    north = ValueBoundaryCondition(0.0),
    south = ValueBoundaryCondition(0.0)
)

Re = 100  # Reynolds number

model = IncompressibleModel(
    grid=grid,
    buoyancy=nothing,
    tracers=nothing,
    coriolis=nothing,
    boundary_conditions = (v=v_bcs, w=w_bcs),
    closure = ConstantIsotropicDiffusivity(ν=1/Re)
)

u, v, w = model.velocities
ζ_op = ∂y(w) - ∂z(v)
ζ = Field(Cell, Face, Face, model.architecture, model.grid, TracerBoundaryConditions(grid))
ζ_comp = Computation(ζ_op, ζ)

max_Δt = 0.25 * model.grid.Δy^2 * Re / 2
wizard = TimeStepWizard(cfl=0.1, Δt=1e-6, max_change=1.1, max_Δt=max_Δt)
cfl = AdvectiveCFL(wizard)
dcfl = DiffusiveCFL(wizard)

function print_progress(simulation)
    model = simulation.model

    # Calculate simulation progress in %.
    progress = 100 * (model.clock.time / simulation.stop_time)

    # Find maximum velocities.
    vmax = maximum(abs, model.velocities.v.data.parent)
    wmax = maximum(abs, model.velocities.w.data.parent)

    i, t = model.clock.iteration, model.clock.time
    @printf("[%06.2f%%] i: %d, t: %.3f, U_max: (%.2e, %.2e), CFL: %.2e, dCFL: %.2e, next Δt: %.2e\n",
            progress, i, t, vmax, wmax, cfl(model), dcfl(model), simulation.Δt.Δt)
end

simulation = Simulation(model, Δt=wizard, stop_time=10, progress=print_progress, progress_frequency=20)

fields = Dict("v" => model.velocities.v, "w" => model.velocities.w, "ζ" => model -> ζ_comp(model))
dims = Dict("ζ" => ("xC", "yF", "zF"))
global_attributes = Dict("Re" => Re)
output_attributes = Dict("ζ" => Dict("longname" => "vorticity", "units" => "1/s"))

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filename="lid_driven_cavity_Re$Re.nc", interval=0.1,
                       global_attributes=global_attributes, output_attributes=output_attributes,
                       dimensions=dims)

run!(simulation)
