using Printf
using Oceananigans
using Oceananigans: Face, Cell
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.AbstractOperations

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

simulation = Simulation(model, Δt=wizard, stop_time=100, progress=print_progress, progress_frequency=20)

fields = Dict("v" => model.velocities.v, "w" => model.velocities.w, "ζ" => model -> ζ_comp(model))
dims = Dict("ζ" => ("xC", "yF", "zF"))
global_attributes = Dict("Re" => Re)
output_attributes = Dict("ζ" => Dict("longname" => "vorticity", "units" => "1/s"))

simulation.output_writers[:fields] =
    NetCDFOutputWriter(model, fields, filename="lid_driven_cavity_Re$Re.nc", interval=0.1,
                       global_attributes=global_attributes, output_attributes=output_attributes,
                       dimensions=dims)

run!(simulation)
