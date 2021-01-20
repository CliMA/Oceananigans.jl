using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded, RegularCartesianGrid
using Oceananigans.Grids: xnodes, ynodes, interior
using Oceananigans.Simulations: Simulation, set!, run!, TimeStepWizard
using Oceananigans.Coriolis: FPlane
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval
using Oceananigans.Fields, Oceananigans.AbstractOperations

using Plots
using Printf
using JLD2
using LinearAlgebra

### Parameters

Lx = 2 * π       # Geometry
Ly = Lx
Nx = 64
Ny = Nx

f = 10           # Physics
g = 10

Δη = 1.0         # Initial Conditions
k   = 0.5
ℓ   = 0.5
amp = 0.1

grid = RegularCartesianGrid(
    size=(Nx, Ny, 1),
    x=(-Lx, Lx),
    y=(-Ly, Ly),
    z=( -1,  1),
    topology=(Periodic, Bounded, Bounded)
    )


model = ShallowWaterModel(
    architecture=CPU(),
 #   timestepper = :RungeKutta3, 
    advection=WENO5(),
    grid=grid,
    gravitational_acceleration=g,
    coriolis=FPlane(f=f)
    )

### Basic State
H₀(x, y, z) =   10.0
H(x, y, z)  =   H₀(x, y, z) - f / g * Δη * tanh(y) + f / g * 2 * y / Ly
U(x, y, z)  =   Δη * sech(y)^2 - 2 / Ly
UH(x, y, z) = U(x, y, z) * H(x, y, z)
Ω( x, y, z) =  2 * Δη * sech(y)^2 * tanh(y)

### Perturbation
ψ(x, y, z, ℓ , k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)
h_perturbation( x, y, z) = amp * ψ(x, y, z, ℓ, k)
uh_perturbation(x, y, z) = amp * ψ(x, y, z, ℓ, k)
vh_perturbation(x, y, z) = amp * ψ(x, y, z, ℓ, k)

## Total fields
uh(x, y, z) = UH(x, y, z) + uh_perturbation(x, y, z)
vh(x, y, z) = 0           + vh_perturbation(x, y, z)
h(x, y, z) =  H(x, y, z)  +  h_perturbation(x, y, z)

set!(model, uh = uh, vh = vh, h = h)

wizard = TimeStepWizard(cfl=1.0, Δt=1e-3, max_change=1.1, max_Δt=1e-1)

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

#simulation = Simulation(model, Δt=1e-3, stop_iteration=5)
simulation = Simulation(model, Δt=wizard, stop_iteration=5)

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
    
    print("Norm of perturbation = ", norm(ωp_snapshot), " with N = ", model.grid.Nx, "\n")

end

gif(anim, "swm_bickley_jet.gif", fps=8)

# To-Do-List
# 1) Figure out why convergence is linear and not quadratic!
# 2) Get timestepping wizard working
# 3) Speed up

