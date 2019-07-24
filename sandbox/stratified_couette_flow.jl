using Oceananigans, Printf, PyPlot, Random, Distributions

function terse_message(model, walltime, Δt)
    wmax = maximum(abs, model.velocities.w.data.parent)
    cfl = Δt / Oceananigans.cell_advection_timescale(model)

    return @sprintf(
        "i: %d, t: %.4f, Δt: %.3f s, wmax: %.6f ms⁻¹, cfl: %.3f, wall time: %s\n",
        model.clock.iteration, model.clock.time, Δt, wmax, cfl, prettytime(1e9*walltime),
       )
end

#
# Initial condition, boundary condition, and tracer forcing
#

 N = 16
Pr = 0.7
Re = 10^4
Ri = 0.01
 L = 1.0
ΔU = 1.0

#=
Ri = L Δb / ΔU^2
Ri*Re² = Ra = Δb * L³ / νκ = [m⁴/s²] / [m⁴/s²]
=#

# Computed parameters
Δb = Ri * ΔU^2 / L
 ν = ΔU * L / Re
 κ = ν / Pr

Tbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Value,  Δb),
    bottom = BoundaryCondition(Value, -Δb)
   ))

ubcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Value,  ΔU),
    bottom = BoundaryCondition(Value, -ΔU),
   ))

#
# Model setup
#

arch = CPU()
#@hascuda arch = GPU() # use GPU if it's available

model = Model(
         arch = arch,
            N = (4N, 16, 4N),
            L = (4L,  L,  L),
      closure = AnisotropicMinimumDissipation(ν=ν, κ=κ),
          eos = LinearEquationOfState(βT=1.0, βS=0.),
    constants = PlanetaryConstants(f=0.0, g=1.0),
          bcs = BoundaryConditions(u=ubcs, T=Tbcs)
    )

filename(model) = @sprintf("stratified_couette_Re%d_Ri%.3f_Nz%d", Re, Ri, model.grid.Nz)

# Add a bit of surface-concentrated noise to the initial condition
Ξ(z) = rand(Normal(0, 1)) * z/model.grid.Lz * (1 + z/model.grid.Lz)

T₀(x, y, z) = 2Δb * (1/2 + z/model.grid.Lz) * (1 + 1e-2 * Ξ(z))
u₀(x, y, z) = 2ΔU * (1/2 + z/model.grid.Lz) * (1 + 1e-2 * Ξ(z)) * (1 + 0.1*sin(4π/model.grid.Lx * x))
v₀(x, y, z) = 1e-6 * Ξ(z)
w₀(x, y, z) = 1e-2 * Ξ(z)
S₀(x, y, z) = 1e-2 * Ξ(z)

set_ic!(model, u=u₀, v=v₀, w=w₀, T=T₀, S=S₀)

@printf(
    """
    Crunching stratified Couette flow with

            n : %d, %d, %d
           Re : %.0e
           Ri : %.1e

    Let's spin the gears.

    """, model.grid.Nx, model.grid.Ny, model.grid.Nz, Re, Ri

)

close("all")
fig, axs = subplots()

wizard = TimeStepWizard(cfl=0.1, Δt=0.01, max_change=1.1, max_Δt=1.0)

# Spin up
for i = 1:100
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)
end

# Reset CFL condition values
wizard.cfl = 0.2
wizard.max_change = 1.5

# Run the model (1000 steps with Δt = 1.0)
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)

    sca(axs); cla()
    imshow(rotr90(view(data(model.velocities.u), :, 2, :)))
    show()
end
