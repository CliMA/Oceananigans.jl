using Oceananigans.Architectures: CPU, architecture
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded, RegularCartesianGrid
using Oceananigans.Grids: xnodes, ynodes, interior
using Oceananigans.Simulations: Simulation, set!, run!, TimeStepWizard
using Oceananigans.Coriolis: FPlane
using Oceananigans.Advection: WENO5
using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval
using Oceananigans.Fields, Oceananigans.AbstractOperations

using Statistics
using Plots
using Printf
using JLD2
using LinearAlgebra

Lx = 2π 
Ly = 2π
Lz = 2π

Nx = 128
Ny = Nx

f = 100         
g = 1

grid = RegularCartesianGrid(
        size=(Nx, Ny, 1), 
      extent=(Lx, Ly, Lz),
    topology=(Periodic, Periodic, Bounded)
    )

model = ShallowWaterModel(
                  architecture=CPU(),
                     advection=WENO5(),
                          grid=grid,
    gravitational_acceleration=g,
                      coriolis=FPlane(f=f)
    )

hᵢ = model.grid.Lz .+ 0.1*rand(size(model.grid)...)

set!(model, h = hᵢ)

uh, vh, h = model.solution

u_op   = model.solution.uh / model.solution.h
v_op   = model.solution.vh / model.solution.h
η_op   = model.solution.h - model.grid.Lz
ω_op   = @at (Center, Center, Center) ∂x(v_op) - ∂y(u_op)
s_op   = sqrt(u_op^2 + v_op^2) 

u_field = ComputedField(u_op)
v_field = ComputedField(v_op)
η_field = ComputedField(η_op)
ω_field = ComputedField(ω_op)
s_field = ComputedField(s_op)

simulation = Simulation(model, Δt=1e-3, stop_iteration=1000)

simulation.output_writers[:fields] =
    JLD2OutputWriter(
        model,
        (u = u_field, v = v_field, η = η_field, ω = ω_field, s = s_field),
        prefix = "two_dimensional_turbulence_shallow_water",
        schedule=IterationInterval(20),
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
           xlim = (0, Ly),
           ylim = (0, Lx)
)

@info "Making a movie of the vorticity and speed fields..."

anim = @animate for (i, iteration) in enumerate(iterations[2:end])

    @info "Plotting frame $i from iteration $iteration..."

    t = file["timeseries/t/$iteration"]
    ω_snapshot = file["timeseries/ω/$iteration"][:, :, 1]
    s_snapshot = file["timeseries/s/$iteration"][:, :, 1]

    ω_plot = contour(yc, xc, ω_snapshot, title=@sprintf("ζ at t = %.3f", t); kwargs...)
    s_plot = contour(yc, xc, s_snapshot, title=@sprintf("s at t = %.3f", t); kwargs...)

    plot(
        ω_plot, 
        s_plot, 
        layout = (1,2), 
        size=(1200,500) 
        )
    
end

gif(anim, "two_dimensional_turbulence_shallow_water.gif", fps=8)
