using Oceananigans
using Statistics
using Printf

Oceananigans.defaults.FloatType = Float64
Nx = Ny = 512

grid = RectilinearGrid(size = (Nx, Ny),
                       halo = (7, 7),
                       extent = (Nx, Ny),
                       topology = (Periodic, Periodic, Flat))

clock = Clock{Float64}(time=0)
model = NonhydrostaticModel(; grid, clock, advection=WENO(order=9))

ui(x, y) = randn()
set!(model, u=ui, v=ui)

Δx = minimum_xspacing(grid)
simulation = Simulation(model; Δt=1e-2, stop_iteration=200)
#conjure_time_step_wizard!(simulation, cfl=0.5)

u, v, w = model.velocities
eop = @at (Center, Center, Center) (u^2 + v^2) / 2
e = Field(eop)

wall_clock = Ref(time_ns())
function progress(sim)
    compute!(e)
    avg_e = mean(e)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("Iter: %s, time: %.2f, wall clock: %s, ⟨e⟩: %.2e",
                   iteration(sim), time(sim), prettytime(elapsed), avg_e)
    wall_clock[] = time_ns()
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

FT = Oceananigans.defaults.FloatType
outputs = merge(model.velocities, (; e))
ow = JLD2Writer(model, outputs;
                filename = "float_point_test_$(Nx)_$FT.jld2",
                schedule = TimeInterval(0.1),
                overwrite_existing = true)

simulation.output_writers[:jld2] = ow

run!(simulation)

