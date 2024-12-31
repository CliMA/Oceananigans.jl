using Oceananigans
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver
using Oceananigans.Solvers: DiagonallyDominantPreconditioner
using Oceananigans.Operators: ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using Oceananigans.Solvers: FFTBasedPoissonSolver
using Printf
using CUDA

using Oceananigans.BoundaryConditions: PerturbationAdvectionOpenBoundaryCondition

u∞ = 1
r = 1/2
arch = GPU()
stop_time = 200

cylinder(x, y) = (x^2 + y^2) ≤ r^2

"""
    drag(modell; bounding_box = (-1, 3, -2, 2), ν = 1e-3)

Returns the drag within the `bounding_box` computed by:

```math
\\frac{\\partial \\vec{u}}{\\partial t} + (\\vec{u}\\cdot\\nabla)\\vec{u}=-\\nabla P + \\nabla\\cdot\\vec{\\tau} + \\vec{F},\\newline
\\vec{F}_T=\\int_\\Omega\\vec{F}dV = \\int_\\Omega\\left(\\frac{\\partial \\vec{u}}{\\partial t} + (\\vec{u}\\cdot\\nabla)\\vec{u}+\\nabla P - \\nabla\\cdot\\vec{\\tau}\\right)dV,\\newline
\\vec{F}_T=\\int_\\Omega\\left(\\frac{\\partial \\vec{u}}{\\partial t}\\right)dV + \\oint_{\\partial\\Omega}\\left(\\vec{u}(\\vec{u}\\cdot\\hat{n}) + P\\hat{n} - \\vec{\\tau}\\cdot\\hat{n}\\right)dS,\\newline
\\F_u=\\int_\\Omega\\left(\\frac{\\partial u}{\\partial t}\\right)dV + \\oint_{\\partial\\Omega}\\left(u(\\vec{u}\\cdot\\hat{n}) - \\tau_{xx}\\right)dS + \\int_{\\partial\\Omega}P\\hat{x}\\cdot d\\vec{S},\\newline
F_u=\\int_\\Omega\\left(\\frac{\\partial u}{\\partial t}\\right)dV - \\int_{\\partial\\Omega_1} \\left(u^2 - 2\\nu\\frac{\\partial u}{\\partial x} + P\\right)dS + \\int_{\\partial\\Omega_2}\\left(u^2 - 2\\nu\\frac{\\partial u}{\\partial x}+P\\right)dS -  \\int_{\\partial\\Omega_2} uvdS + \\int_{\\partial\\Omega_4} uvdS,
```
where the bounding box is ``\\Omega`` which is formed from the boundaries ``\\partial\\Omega_{1}``, ``\\partial\\Omega_{2}``, ``\\partial\\Omega_{3}``, and ``\\partial\\Omega_{4}`` 
which have outward directed normals ``-\\hat{x}``, ``\\hat{x}``, ``-\\hat{y}``, and ``\\hat{y}``
"""
function drag(model;
              bounding_box = (-1, 3, -2, 2),
              ν = 1e-3)

    u, v, _ = model.velocities

    uᶜ = Field(@at (Center, Center, Center) u)
    vᶜ = Field(@at (Center, Center, Center) v)

    xc, yc, _ = nodes(uᶜ)

    i₁ = findfirst(xc .> bounding_box[1])
    i₂ = findlast(xc .< bounding_box[2])

    j₁ = findfirst(yc .> bounding_box[3])
    j₂ = findlast(yc .< bounding_box[4])

    uₗ² = Field(uᶜ^2, indices = (i₁, j₁:j₂, 1))
    uᵣ² = Field(uᶜ^2, indices = (i₂, j₁:j₂, 1))

    uvₗ = Field(uᶜ*vᶜ, indices = (i₁:i₂, j₁, 1))
    uvᵣ = Field(uᶜ*vᶜ, indices = (i₁:i₂, j₂, 1))

    ∂₁uₗ = Field(∂x(uᶜ), indices = (i₁, j₁:j₂, 1))
    ∂₁uᵣ = Field(∂x(uᶜ), indices = (i₂, j₁:j₂, 1))

    ∂ₜuᶜ = Field(@at (Center, Center, Center) model.timestepper.Gⁿ.u)

    ∂ₜu = Field(∂ₜuᶜ, indices = (i₁:i₂, j₁:j₂, 1))

    p = model.pressures.pNHS

    ∫∂ₓp = Field(∂x(p), indices = (i₁:i₂, j₁:j₂, 1))

    a_local = Field(Integral(∂ₜu))

    a_flux = Field(Integral(uᵣ²)) - Field(Integral(uₗ²)) + Field(Integral(uvᵣ)) - Field(Integral(uvₗ))

    a_viscous_stress = 2ν * (Field(Integral(∂₁uᵣ)) - Field(Integral(∂₁uₗ)))

    a_pressure = Field(Integral(∫∂ₓp))

    return a_local + a_flux + a_pressure - a_viscous_stress
end

Re = 1000
#=
if Re <= 100
    Ny = 512 
    Nx = Ny
elseif Re <= 1000
    Ny = 2^11
    Nx = Ny
elseif Re == 10^4
    Ny = 2^12
    Nx = Ny
elseif Re == 10^5
    Ny = 2^13
    Nx = Ny
elseif Re == 10^6
    Ny = 3/2 * 2^13 |> Int
    Nx = Ny
end
=#

Ny = 2048 
Nx = Ny

prefix = "flow_around_cylinder_Re$(Re)_Ny$(Ny)"

ϵ = 0 # break up-down symmetry
x = (-6, 12) # 18
y = (-6 + ϵ, 6 + ϵ)  # 12
# TODO: temporatily use an iteration interval thingy with a fixed timestep!!!
kw = (; size=(Nx, Ny), x, y, halo=(6, 6), topology=(Bounded, Bounded, Flat))
grid = RectilinearGrid(arch; kw...)
reduced_precision_grid = RectilinearGrid(arch, Float32; kw...)

grid = ImmersedBoundaryGrid(grid, GridFittedBoundary(cylinder))

advection = Centered(order=2)
closure = ScalarDiffusivity(ν=1/Re)

no_slip = ValueBoundaryCondition(0)
u_bcs = FieldBoundaryConditions(immersed=no_slip, 
                                east = PerturbationAdvectionOpenBoundaryCondition(u∞; inflow_timescale = 1/4, outflow_timescale = Inf),
                                west = PerturbationAdvectionOpenBoundaryCondition(u∞; inflow_timescale = 0.1, outflow_timescale = Inf))
v_bcs = FieldBoundaryConditions(immersed=no_slip,
                                east = GradientBoundaryCondition(0),
                                west = ValueBoundaryCondition(0))
boundary_conditions = (u=u_bcs, v=v_bcs)

ddp = DiagonallyDominantPreconditioner()
preconditioner = FFTBasedPoissonSolver(reduced_precision_grid)
reltol = abstol = 1e-7
pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=10;
                                                 reltol, abstol, preconditioner)

model = NonhydrostaticModel(; grid, pressure_solver, closure,
                              advection, boundary_conditions)

@show model

uᵢ(x, y) = 1e-2 * randn()
vᵢ(x, y) = 1e-2 * randn()
set!(model, u=uᵢ, v=vᵢ)

Δx = minimum_xspacing(grid)
Δt = max_Δt = 0.002#0.2 * Δx^2 * Re

simulation = Simulation(model; Δt, stop_time)
#conjure_time_step_wizard!(simulation, cfl=1.0, IterationInterval(3); max_Δt)

u, v, w = model.velocities
#d = ∂x(u) + ∂y(v)

# Drag computation
drag_force = drag(model; ν=1/Re)
compute!(drag_force)

wall_time = Ref(time_ns())

function progress(sim)
    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    #compute!(drag_force)
    D = CUDA.@allowscalar drag_force[1, 1, 1]
    cᴰ = D / (u∞ * r) 
    vmax = maximum(model.velocities.v)
    dmax = 0#maximum(abs, d)

    msg = @sprintf("Iter: %d, time: %.2f, Δt: %.4f, Poisson iters: %d",
                   iteration(sim), time(sim), sim.Δt, pressure_iters)

    elapsed = 1e-9 * (time_ns() - wall_time[])

    msg *= @sprintf(", max d: %.2e, max v: %.2e, Cd: %0.2f, wall time: %s",
                    dmax, vmax, cᴰ, prettytime(elapsed))

    @info msg
    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

ζ = ∂x(v) - ∂y(u)

p = model.pressures.pNHS

outputs = (; u, v, p, ζ)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs,
                                                    schedule = IterationInterval(Int(2/Δt)),#TimeInterval(0.1),
                                                    filename = prefix * "_fields.jld2",
                                                    overwrite_existing = true,
                                                    with_halos = true)

simulation.output_writers[:drag] = JLD2OutputWriter(model, (; drag_force),
                                                    schedule = IterationInterval(Int(0.1/Δt)),#TimeInterval(0.1),
                                                    filename = prefix * "_drag.jld2",
                                                    overwrite_existing = true,
                                                    with_halos = true,
                                                    indices = (1, 1, 1))

run!(simulation)


# Re = 100
# with 12m ~0.33 Hz and Cd ~ 1.403
# with 6m  ~0.33 Hz and Cd ~ (higher, don't seem to have computed it!)
# with 18m ~0.32 Hz and Cd ~ 1.28
# with 30m ~0.34 Hz and Cd ~ 1.37
# the Strouhal number should be 0.16 to 0.18 which I think means we're pretty close as 1 drag osccilation cycle is 2 sheddings so we have 0.165 ish

# Re = 1000
# with 12m ~ HZ and Cd ~