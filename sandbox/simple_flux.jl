using Oceananigans, Random, Distributions, PyPlot, Printf

function terse_message(model, walltime, Δt)
    wmax = maximum(abs, model.velocities.w.data.parent)
    cfl = Δt / Oceananigans.cell_advection_timescale(model)

    return @sprintf(
        "i: %d, t: %.4f hours, Δt: %.1f s, wmax: %.6f ms⁻¹, cfl: %.3f, wall time: %s\n",
        model.clock.iteration, model.clock.time/3600, Δt, wmax, cfl, prettytime(1e9*walltime),
       )
end

   N = 64
   Δ = 1.0
  Fb = 1e-8
  Fu = 0.0
  N² = 1e-4
  tf = 4 * 3600

  βT = 2e-4
   g = 9.81
  Fθ = Fb / (g*βT)
dTdz = N² / (g*βT)

# Instantiate a model
Tbcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Flux, Fθ),
    bottom = BoundaryCondition(Gradient, dTdz) ))

ubcs = FieldBoundaryConditions(z=ZBoundaryConditions(
    top    = BoundaryCondition(Flux, Fu)
   ))

model = Model(      arch = CPU(), # GPU(),
                       N = (N, 4, N),
                       L = (N*Δ, N*Δ, N*Δ),
                     eos = LinearEquationOfState(βT=βT, βS=0.0),
               constants = PlanetaryConstants(f=1e-4, g=g),
                 closure = AnisotropicMinimumDissipation(),
                #closure = ConstantSmagorinsky(),
                     bcs = BoundaryConditions(u=ubcs, T=Tbcs))

# Set initial condition
Ξ(z) = rand(Normal(0, 1)) * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
T₀(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
S₀(x, y, z) = 1e-6 * Ξ(z)
u₀(x, y, z) = 1e-6 * Ξ(z)

set_ic!(model, u=u₀, T=T₀, S=S₀)

close("all")
fig, axs = subplots()

wizard = TimeStepWizard(cfl=0.1, Δt=1.0, max_change=1.1, max_Δt=90.0)

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
    imshow(rotr90(view(data(model.velocities.w), :, 2, :)))
    show()
end
