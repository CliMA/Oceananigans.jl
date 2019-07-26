using Oceananigans, Random, Distributions, PyPlot, Printf

function terse_message(model, walltime, Δt)
    wmax = maximum(abs, model.velocities.w.data.parent)
    cfl = Δt / Oceananigans.cell_advection_timescale(model)

    return @sprintf(
        "i: %d, t: %.4f hours, Δt: %.1f s, wmax: %.6f ms⁻¹, cfl: %.3f, wall time: %s\n",
        model.clock.iteration, model.clock.time/3600, Δt, wmax, cfl, prettytime(1e9*walltime),
       )
end

# Parameters from Van Roekel et al (JAMES, 2018)
parameters = Dict(
    :free_convection => Dict(:Fb=>3.39e-8, :Fu=>0.0,     :f=>1e-4, :N²=>1.96e-5),
    :wind_stress     => Dict(:Fb=>0.0,     :Fu=>9.66e-5, :f=>0.0,  :N²=>9.81e-5)
   )

# Simulation parameters
case = :wind_stress
   N = 32                       # Resolution    
   Δ = 0.5                      # Grid spacing
  tf = day/2                    # Final simulation time
  N² = parameters[case][:N²]
  Fb = parameters[case][:Fb]
  Fu = parameters[case][:Fu]
   f = parameters[case][:f]

# Physical constants
  βT = 2e-4                     # Thermal expansion coefficient
   g = 9.81                     # Gravitational acceleration
  Fθ = Fb / (g*βT)              # Temperature flux
dTdz = N² / (g*βT)              # Initial temperature gradient

# Create boundary conditions
ubcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Fu))
Tbcs = HorizontallyPeriodicBCs(    top = BoundaryCondition(Flux, Fθ),
                                bottom = BoundaryCondition(Gradient, dTdz))

# Instantiate the model
model = Model(      arch = HAVE_CUDA ? GPU() : CPU(),
                       N = (N, N, 2N),
                       L = (N*Δ, N*Δ, N*Δ),
                     eos = LinearEquationOfState(βT=βT, βS=0.0),
               constants = PlanetaryConstants(f=f, g=g),
                 closure = AnisotropicMinimumDissipation(),
                 #closure = ConstantSmagorinsky(),
                     bcs = BoundaryConditions(u=ubcs, T=Tbcs))

# Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
Ξ(z) = rand(Normal(0, 1)) * z / model.grid.Lz * (1 + z / model.grid.Lz) # noise
u₀(x, y, z) = 1e-9 * Ξ(z)
T₀(x, y, z) = 20 + dTdz * z + dTdz * model.grid.Lz * 1e-3 * Ξ(z)
S₀(x, y, z) = 1e-9 * Ξ(z)

set_ic!(model, u=u₀, T=T₀, S=S₀)

close("all")
fig, axs = subplots()

wizard = TimeStepWizard(cfl=0.01, Δt=1.0, max_change=1.1, max_Δt=90.0)

# Spin up
for i = 1:100
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 10, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)
end

# Reset CFL condition values
wizard.cfl = 0.2
wizard.max_change = 1.5

# Run the model
while model.clock.time < tf
    update_Δt!(wizard, model)
    walltime = @elapsed time_step!(model, 100, wizard.Δt)
    @printf "%s" terse_message(model, walltime, wizard.Δt)
    
    sca(axs); cla()
    imshow(rotr90(view(data(model.velocities.w), :, 2, :)))
    show()
end
