using Oceananigans.Architectures: CPU
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded, RegularCartesianGrid
using Oceananigans.Grids: xnodes, ynodes, interior
using Oceananigans.Simulations: Simulation, set!, run!
using Oceananigans.Coriolis: FPlane
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval
using Oceananigans.Fields, Oceananigans.AbstractOperations

using Plots
using Printf
using JLD2

Lx = 10
Ly = Lx

grid = RegularCartesianGrid(
    size=(64, 64, 1), 
    x=(-Lx, Lx),
    y=(-Ly, Ly),
    z=( -1,  1), 
    topology=(Periodic, Bounded, Bounded) 
    )

model = ShallowWaterModel(
    grid=grid, 
    gravitational_acceleration=1, 
    architecture=CPU(), 
    coriolis=FPlane(f=1),
    advection=WENO5() 
    )

### Parameters
#H₀ = 1.0
Δη = 0.1
g  = model.gravitational_acceleration
f  = model.coriolis.f

k   = 10
ℓ   = 1
amp = 1e-4

### Basic State
H₀(x, y, z) =   1.0
H(x, y, z)  =   H₀(x, y, z) - Δη * tanh(y) 
U(x, y, z)  =   g / f * Δη * sech(y)^2
UH(x, y, z) = U(x, y, z) * H(x, y, z)

### Perturbation
h_perturbation( x, y, z) = amp * exp(-y^2 / 2ℓ^2) * cos(k * x) * cos(k * y)
uh_perturbation(x, y, z) = amp * exp(-y^2 / 2ℓ^2) * cos(k * x) * cos(k * y)
vh_perturbation(x, y, z) = amp * exp(-y^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

## Total fields
uh(x, y, z) = UH(x, y, z) + uh_perturbation(x, y, z)
vh(x, y, z) = 0           + vh_perturbation(x, y, z)
h(x, y, z) =  H(x, y, z)  +  h_perturbation(x, y, z)

set!(model, uh = uh, h = h)

### FJP: uh and vh must averaged on the cell centers for this to be correct
u_op = @at (Face, Cell, Cell) uh / h
v_op = @at (Cell, Face, Cell) vh / h
η_op = @at (Cell, Cell, Cell) h - H₀

u = ComputedField(u_op)
v = ComputedField(v_op)
η = ComputedField(η_op)

#u = model.solution.uh/model.solution.h
#v = model.solution.vh/model.solution.h
#η = model.solution.h - H₀

#u_field = ComputedField(u)
#v_field = ComputedField(v)
#η_field = ComputedField(η)

xC = model.grid.xC
yC = model.grid.yC
#xc = xnodes(model.solution.h)
#yc = ynodes(model.solution.h)

kwargs = (
         xlabel = "x",
         ylabel = "y",
           fill = true,
         levels = 20,
      linewidth = 0,
          color = :balance,
       colorbar = true,
    aspectratio = 1,
           xlim = (-Lx, Lx),
           ylim = (-Ly, Ly)
)

#η = interior(model.solution.h)[:,:,1] .- H₀# - H(x, y, 0)
plt = contour(
    xC, 
    yC, 
    η,
    title = "Free-surface height";
    kwargs...
    )
display(plt)

simulation = Simulation(model, Δt=1e-2, stop_iteration=10)

simulation.output_writers[:height] =
    JLD2OutputWriter(model, model.solution, prefix = "Bickley_Jet",
                     schedule=IterationInterval(1), force = true)


run!(simulation)

file = jldopen(simulation.output_writers[:height].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)

    T = file["timeseries/h/$iter"][:, :, 1]
    t = file["timeseries/t/$iter"]

    local η = interior(model.solution.h)[:,:,1] .- H₀

    contour(xc, yc, η, title=@sprintf("t = %.3f", t); kwargs...)
end