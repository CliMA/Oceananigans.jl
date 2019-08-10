using Oceananigans, Random, Distributions, Printf
using PyPlot

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
   N = 128                      # Resolution
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

function savebcs(file, model)
    file["boundary_conditions/top/FT"] = Fθ
    file["boundary_conditions/top/Fb"] = Fb
    file["boundary_conditions/top/Fu"] = Fu
    file["boundary_conditions/bottom/dTdz"] = dTdz
    file["boundary_conditions/bottom/dbdz"] = dTdz * g * βT
    return nothing
end

u(model) = Array{Float32}(model.velocities.u.data.parent)
v(model) = Array{Float32}(model.velocities.v.data.parent)
w(model) = Array{Float32}(model.velocities.w.data.parent)
θ(model) = Array{Float32}(model.tracers.T.data.parent)
ν(model) = Array{Float32}(model.diffusivities.νₑ.parent)

const avgs = Oceananigans.HorizontalAverages(model)

function hmean!(ϕavg, ϕ::Field)
    ϕavg .= mean(ϕ.data.parent, dims=(1, 2))
    return nothing
end

function U(model)
    hmean!(avgs.U, model.velocities.u)
    return Array{Float32}(avgs.U)
end

function V(model)
    hmean!(avgs.V, model.velocities.v)
    return Array{Float32}(avgs.V)
end

function T(model)
    hmean!(avgs.T, model.tracers.T)
    return Array{Float32}(avgs.T)
end

profiles = Dict(:U=>U, :V=>V, :T=>T)
  fields = Dict(:u=>u, :v=>v, :w=>w, :θ=>θ)

filename(model) = @sprintf(
                       "simple_flux_Fb%.0e_Fu%.0e_Nsq%.0e_Lz%d_Nz%d",
                       Fb, Fu, N², model.grid.Lz, model.grid.Nz
                      )

profile_writer = JLD2OutputWriter(model, profiles; dir="data",
                                  prefix=filename(model)*"_profiles",
                                  init=savebcs, interval=0.5*hour, force=true)

field_writer = JLD2OutputWriter(model, fields; dir="data",
                                prefix=filename(model)*"_fields",
                                init=savebcs, interval=2hour, force=true)

#push!(model.output_writers, profile_writer, field_writer)

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
