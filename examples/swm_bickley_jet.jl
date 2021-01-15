using Oceananigans.Architectures: CPU, architecture
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
using LinearAlgebra

Lx = 10
Ly = Lx
Nx = 64
Ny = Nx

grid = RegularCartesianGrid(
    size=(Nx, Ny, 1),
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
Δη = 0.1
g  = model.gravitational_acceleration
f  = model.coriolis.f

k   = 4
ℓ   = 1
amp = 0e-3

### Basic State
H₀(x, y, z) =   1.0
H(x, y, z)  =   H₀(x, y, z) - Δη * tanh(y)
U(x, y, z)  =   g / f * Δη * sech(y)^2
UH(x, y, z) = U(x, y, z) * H(x, y, z)
Ω( x, y, z) =  2 * g / f * Δη * sech(y)^2 * tanh(y)

### Perturbation
h_perturbation( x, y, z) = amp * exp(-y^2 / 2ℓ^2) * cos(k * x * 2 * π / Lx )
uh_perturbation(x, y, z) = amp * exp(-y^2 / 2ℓ^2) * cos(k * x * 2 * π / Lx )
vh_perturbation(x, y, z) = amp * exp(-y^2 / 2ℓ^2) * cos(k * x * 2 * π / Lx )

## Total fields
uh(x, y, z) = UH(x, y, z) + uh_perturbation(x, y, z)
vh(x, y, z) = 0           + vh_perturbation(x, y, z)
h(x, y, z) =  H(x, y, z)  +  h_perturbation(x, y, z)

set!(model, uh = uh, vh = vh, h = h)

u_op   = model.solution.uh / model.solution.h
v_op   = model.solution.vh / model.solution.h
η_op   = model.solution.h - H₀
ω_op   = @at (Cell, Cell, Cell) ∂x(v_op) - ∂y(u_op)
ω_pert = @at (Cell, Cell, Cell) ω_op - Ω

u_field = ComputedField(u_op)
v_field = ComputedField(v_op)
η_field = ComputedField(η_op)
ω_field = ComputedField(ω_op)
ω_pert  = ComputedField(ω_pert)

simulation = Simulation(model, Δt=1e-2, stop_iteration=1)

simulation.output_writers[:fields] =
    JLD2OutputWriter(
        model,
        (u = u_field, v = v_field, η = η_field, ω = ω_field, ωp = ω_pert),
        prefix = "Bickley_Jet",
        schedule=IterationInterval(1),
        force = true)

run!(simulation)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

xc = xnodes(model.solution.h)
yc = ynodes(model.solution.h)

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

@info "Making a movie of the free-surface height and vorticity fields..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."

    t = file["timeseries/t/$iteration"]
    η_snapshot  = file["timeseries/η/$iteration"][:, :, 1]
    ω_snapshot  = file["timeseries/ω/$iteration"][:, :, 1]
    ωp_snapshot = file["timeseries/ωp/$iteration"][:, :, 1]

    #η_plot = contour(xc, yc, η_snapshot, title="free-surface"; kwargs...)
    ω_plot  = contour(xc, yc, ω_snapshot, title="Total vorticity";          kwargs...)
    ωp_plot = contour(xc, yc, ωp_snapshot, title="Perturbation vorticity";  kwargs...)

    plot(
        ω_plot, 
        ωp_plot, 
        layout = (1,2), 
        size=(1200,500) 
        )

    print("Norm of perturbation = ", norm(ωp_snapshot), "with N = ", model.grid.Nx, "\n")

end


# To-Do-List
# 1) Plot perturbation fields
# 2) Get timestepping wizard working
# 3) Speed up

