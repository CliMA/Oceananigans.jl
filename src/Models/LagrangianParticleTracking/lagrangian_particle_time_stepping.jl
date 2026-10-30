#####
##### Time stepping Lagrangian particles
#####
##### Particles solve dX/dt = Ẋ(X, t), where Ẋ is the particle velocity in the coordinates of the
##### grid (see `particle_coordinate_velocity`). Particles are advanced with the scheme of the
##### model's time stepper, one stage at a time:
#####
#####   1. At the beginning of the first stage, particles are sorted, `dynamics` is called once
#####      with the full time step, and flags for absorbed particles are reset.
#####   2. At every stage, the particle velocity is evaluated at the current particle position with
#####      the fields at the beginning of the stage, the position is advanced by
#####      `advance_particle_position`, and boundary conditions are applied.
#####   3. At the end of the last stage, absorbed particles are removed.
#####
##### Because particles are sorted and removed only once per time step, per-particle storage
##### for the scheme (`particles.timestepper`) stays aligned with the particles within a time step.
#####

#####
##### Per-particle storage
#####

"""
    ParticleTimeStepper{S, B}

The `timestepper` property of `LagrangianParticles` that belong to a model whose time-stepping
scheme needs per-particle storage, such as the velocity of each particle at the previous stage.

- `state`: a `NamedTuple` of vectors, each with one element per particle.
- `buffers`: scratch vectors used to reorder `state`.

`state` is resized with the particles, permuted when particles are sorted, and compacted when
absorbed particles are removed, so that `state[name][p]` always belongs to particle `p`.
`state` is checkpointed.

`ParticleTimeStepper` is to `LagrangianParticles` what a time stepper's tendency storage is to
the prognostic fields of a model: it holds the history required by multi-stage and multi-step schemes.
"""
struct ParticleTimeStepper{S, B}
      state :: S
    buffers :: B
end

function ParticleTimeStepper(state::NamedTuple)
    buffers = map(array -> similar(array, 0), state)
    return ParticleTimeStepper(state, buffers)
end

particle_time_stepper(::Nothing) = nothing
particle_time_stepper(state::NamedTuple) = ParticleTimeStepper(state)

time_stepper_state(::Nothing) = nothing
time_stepper_state(timestepper::ParticleTimeStepper) = timestepper.state

Base.summary(timestepper::ParticleTimeStepper) = string("ParticleTimeStepper with state ", keys(timestepper.state))

"""
    particle_time_stepper_state(timestepper, x)

Return a `NamedTuple` of empty vectors (for example, `similar(x, 0)`) that store per-particle
state for advancing Lagrangian particles with `timestepper`, or `nothing` if `timestepper` needs
no per-particle state. The vectors are resized to the number of particles at the beginning of
every time step. `x` is the vector of particle `x`-positions.

The default is `nothing`. Both `RungeKutta3TimeStepper` and `QuasiAdamsBashforth2TimeStepper`
store the previous particle velocity `(u⁻, v⁻, w⁻)`, in the coordinates of the grid.
"""
particle_time_stepper_state(timestepper, x) = nothing

previous_particle_velocities(x) = (u⁻ = similar(x, 0), v⁻ = similar(x, 0), w⁻ = similar(x, 0))

particle_time_stepper_state(::RungeKutta3TimeStepper, x) = previous_particle_velocities(x)
particle_time_stepper_state(::QuasiAdamsBashforth2TimeStepper, x) = previous_particle_velocities(x)

@inline previous_particle_velocity(state, p) = @inbounds (state.u⁻[p], state.v⁻[p], state.w⁻[p])

@inline function store_particle_velocity!(state, p, (ẋ, ẏ, ż))
    @inbounds begin
        state.u⁻[p] = ẋ
        state.v⁻[p] = ẏ
        state.w⁻[p] = ż
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return `particles` with per-particle storage for advancing them with `timestepper`;
see `particle_time_stepper_state`. Called by model constructors.
"""
function materialize_lagrangian_particles(particles::LagrangianParticles, timestepper)
    state = particle_time_stepper_state(timestepper, particles.properties.x)

    return LagrangianParticles(particles.properties,
                               particles.boundary_conditions,
                               particles.tracked_fields,
                               particles.dynamics,
                               particles.parameters,
                               particles.sorting,
                               particles.removal,
                               particle_time_stepper(state))
end

# Returns `true` if the storage was resized, so that it holds no history from the previous time step.
resize_particle_time_stepper!(::Nothing, Np) = false

function resize_particle_time_stepper!(timestepper::ParticleTimeStepper, Np)
    resized = false
    for array in timestepper.state
        if length(array) != Np
            resize!(array, Np)
            resized = true
        end
    end
    return resized
end

permute_particle_time_stepper!(::Nothing, permutation, grid, Np) = nothing

function permute_particle_time_stepper!(timestepper::ParticleTimeStepper, permutation, grid, Np)
    arch = architecture(grid)
    for (array, buffer) in zip(timestepper.state, timestepper.buffers)
        resize!(buffer, Np)
        launch!(arch, grid, KernelParameters(1:Np), _gather!, buffer, array, permutation)
        copyto!(array, 1, buffer, 1, Np)
    end
    return nothing
end

compact_particle_time_stepper!(::Nothing, removal, grid, Np, Nkept) = nothing

function compact_particle_time_stepper!(timestepper::ParticleTimeStepper, removal, grid, Np, Nkept)
    arch = architecture(grid)
    for (array, buffer) in zip(timestepper.state, timestepper.buffers)
        resize!(buffer, Np)
        launch!(arch, grid, KernelParameters(1:Np), _compact!, buffer, array, removal.absorbed, removal.destinations)
        copyto!(array, 1, buffer, 1, Nkept)
        resize!(array, Nkept)
    end
    return nothing
end

prognostic_state(timestepper::ParticleTimeStepper) = timestepper.state

function restore_prognostic_state!(restored::ParticleTimeStepper, from)
    for name in keys(restored.state)
        array = restored.state[name]
        data = from[name]
        resize!(array, length(data))
        copyto!(array, data)
    end
    return restored
end

restore_prognostic_state!(::ParticleTimeStepper, ::Nothing) = nothing

#####
##### Stages
#####

"""
    particle_stages(timestepper)

Return the number of stages of `timestepper`, that is, the number of times that
`substep_lagrangian_particles!` is called per time step. The default is 1.
"""
particle_stages(timestepper) = 1
particle_stages(::RungeKutta3TimeStepper) = 3

"""
    particle_stage(timestepper, stage, restart)

Return an `isbits` object that describes stage `stage` of `timestepper`, such as the stage
coefficients, for `advance_particle_position`. `restart` is `true` when the per-particle
storage was (re)allocated at the beginning of the current time step, so that it holds no history
from previous time steps (for example, when particles are added). The default is a forward Euler step.
"""
particle_stage(timestepper, stage, restart) = ForwardEulerParticleStep()

"""
    advance_particle_position(stage, X, Ẋ, Δt, state, p)

Return the new position of particle `p`, before boundary conditions are applied, given its
current position `X = (x, y, z)`, its velocity `Ẋ = (ẋ, ẏ, ż)` in the coordinates of the grid
evaluated at `X` with the fields at the beginning of the stage, the full time step `Δt`,
and per-particle `state` (see `particle_time_stepper_state`), which may be updated.
`stage` is returned by `particle_stage`.
"""
function advance_particle_position end

struct ForwardEulerParticleStep end

@inline advance_particle_position(::ForwardEulerParticleStep, (x, y, z), (ẋ, ẏ, ż), Δt, state, p) =
    (x + Δt * ẋ, y + Δt * ẏ, z + Δt * ż)

# Low-storage third-order Runge-Kutta: x += Δt * (γ * Ẋ + ζ * Ẋ⁻), where Ẋ⁻ is the particle velocity
# at the beginning of the previous stage. ζ is `nothing` for the first stage.
struct RungeKutta3ParticleStage{FT, Z}
    γ :: FT
    ζ :: Z
end

function particle_stage(timestepper::RungeKutta3TimeStepper, stage, restart)
    stage == 1 && return RungeKutta3ParticleStage(timestepper.γ¹, nothing)
    stage == 2 && return RungeKutta3ParticleStage(timestepper.γ², timestepper.ζ²)
    return RungeKutta3ParticleStage(timestepper.γ³, timestepper.ζ³)
end

@inline rk3_particle_velocity(γ, ::Nothing, ẋ, ẋ⁻) = γ * ẋ
@inline rk3_particle_velocity(γ, ζ, ẋ, ẋ⁻) = γ * ẋ + ζ * ẋ⁻

@inline function advance_particle_position(stage::RungeKutta3ParticleStage, (x, y, z), (ẋ, ẏ, ż), Δt, state, p)
    γ, ζ = stage.γ, stage.ζ
    ẋ⁻, ẏ⁻, ż⁻ = previous_particle_velocity(state, p)
    store_particle_velocity!(state, p, (ẋ, ẏ, ż))

    x̃ = x + Δt * rk3_particle_velocity(γ, ζ, ẋ, ẋ⁻)
    ỹ = y + Δt * rk3_particle_velocity(γ, ζ, ẏ, ẏ⁻)
    z̃ = z + Δt * rk3_particle_velocity(γ, ζ, ż, ż⁻)

    return (x̃, ỹ, z̃)
end

# Without per-particle storage, each stage is a forward Euler step over the stage time step (γ + ζ) Δt
@inline rk3_stage_Δt(Δt, γ, ::Nothing) = γ * Δt
@inline rk3_stage_Δt(Δt, γ, ζ) = (γ + ζ) * Δt

@inline advance_particle_position(stage::RungeKutta3ParticleStage, X, Ẋ, Δt, ::Nothing, p) =
    advance_particle_position(ForwardEulerParticleStep(), X, Ẋ, rk3_stage_Δt(Δt, stage.γ, stage.ζ), nothing, p)

# Quasi-second-order Adams-Bashforth: x += Δt * ((3/2 + χ) * Ẋⁿ - (1/2 + χ) * Ẋⁿ⁻¹).
# A forward Euler step is taken when the model takes one (χ = -1/2), and when the
# per-particle storage holds no history from the previous time step.
struct QuasiAdamsBashforth2ParticleStep{FT}
        χ :: FT
    euler :: Bool
end

function particle_stage(timestepper::QuasiAdamsBashforth2TimeStepper, stage, restart)
    χ = timestepper.χ
    euler = restart | (χ == -1/2)
    return QuasiAdamsBashforth2ParticleStep(χ, euler)
end

@inline function ab2_particle_velocity(step, ẋ, ẋ⁻)
    FT = typeof(step.χ)
    α = convert(FT, 3/2) + step.χ
    β = convert(FT, 1/2) + step.χ
    # The previous velocity is not used for Euler steps, and may not be initialized
    return ifelse(step.euler, ẋ, α * ẋ - β * ẋ⁻)
end

@inline function advance_particle_position(step::QuasiAdamsBashforth2ParticleStep, (x, y, z), (ẋ, ẏ, ż), Δt, state, p)
    ẋ⁻, ẏ⁻, ż⁻ = previous_particle_velocity(state, p)
    store_particle_velocity!(state, p, (ẋ, ẏ, ż))

    x̃ = x + Δt * ab2_particle_velocity(step, ẋ, ẋ⁻)
    ỹ = y + Δt * ab2_particle_velocity(step, ẏ, ẏ⁻)
    z̃ = z + Δt * ab2_particle_velocity(step, ż, ż⁻)

    return (x̃, ỹ, z̃)
end

@inline advance_particle_position(::QuasiAdamsBashforth2ParticleStep, X, Ẋ, Δt, ::Nothing, p) =
    advance_particle_position(ForwardEulerParticleStep(), X, Ẋ, Δt, nothing, p)

#####
##### Stepping particles
#####

"""
$(TYPEDSIGNATURES)

Prepare `particles` for a new time step: resize per-particle storage, sort the particles,
call `particles.dynamics` with the full time step `Δt`, and reset the flags of absorbed particles.
Return `true` if the per-particle storage was resized, so that it holds no history from the previous time step.
"""
function begin_lagrangian_particle_step!(particles::LagrangianParticles, model, Δt)
    resized = resize_particle_time_stepper!(particles.timestepper, length(particles))
    sort_lagrangian_particles!(particles, model.grid)
    particles.dynamics(particles, model, Δt)

    # `dynamics` may add or remove particles
    Np = length(particles)
    resized = resize_particle_time_stepper!(particles.timestepper, Np) | resized
    reset_absorption!(particles.removal, Np)

    return resized
end

"""
$(TYPEDSIGNATURES)

Advance `particles` through stage `stage` of `timestepper`, where `Δt` is the full time step.
"""
function advance_lagrangian_particles!(particles::LagrangianParticles, model, Δt, timestepper, stage)
    restart = false

    if stage == 1
        restart = begin_lagrangian_particle_step!(particles, model, Δt)
    end

    advect_lagrangian_particles!(particles, model, Δt, particle_stage(timestepper, stage, restart))

    if stage == particle_stages(timestepper)
        remove_absorbed_particles!(particles, model.grid)
    end

    return nothing
end

step_lagrangian_particles!(::Nothing, model, Δt) = nothing
substep_lagrangian_particles!(::Nothing, model, Δt, stage) = nothing

"""
$(TYPEDSIGNATURES)

Advance `particles` by `Δt` with a first-order forward Euler step, using the velocities of `model`.
Particles are sorted, `particles.dynamics` is called, particles are advected, and absorbed
particles are removed. Used by `SplitRungeKuttaTimeStepper` and by models that only track particles.
"""
step_lagrangian_particles!(particles::LagrangianParticles, model, Δt) =
    advance_lagrangian_particles!(particles, model, Δt, nothing, 1)

"""
$(TYPEDSIGNATURES)

Advance `particles` through stage `stage` of `model.timestepper`, where `Δt` is the full time step.
Time steppers call `substep_lagrangian_particles!` at the beginning of each stage, before the
fields are advanced, so that particle velocities are evaluated with the fields at the beginning of the stage.

At the first stage, particles are sorted, `particles.dynamics(particles, model, Δt)` is called
with the full time step, and the flags of absorbed particles are reset. At every stage, particle
positions are advanced and boundary conditions are applied. Particles absorbed at an
[`AbsorbingBoundary`](@ref) stop moving, and are removed at the end of the last stage.

With `RungeKutta3TimeStepper`, particles are advanced with the same low-storage scheme as the fields,

```
x⁽ᵏ⁾ = x⁽ᵏ⁻¹⁾ + Δt * (γᵏ * U⁽ᵏ⁻¹⁾ + ζᵏ * U⁽ᵏ⁻²⁾),
```

where `U⁽ᵏ⁻¹⁾` is the particle velocity at the beginning of stage `k`. With
`QuasiAdamsBashforth2TimeStepper`, particles are advanced with

```
xⁿ⁺¹ = xⁿ + Δt * ((3/2 + χ) * Uⁿ - (1/2 + χ) * Uⁿ⁻¹).
```

## Extending to other time steppers

A time stepper of type `TS` advances particles by calling `substep_lagrangian_particles!(model, Δt, stage)`
at the beginning of each stage and extending

- `particle_time_stepper_state(::TS, x)`: per-particle storage (default: `nothing`);
- `particle_stages(::TS)`: number of stages (default: 1);
- `particle_stage(::TS, stage, restart)`: an `isbits` stage descriptor (default: forward Euler);
- `advance_particle_position(stage_descriptor, X, Ẋ, Δt, state, p)`: the new particle position.

For example, the strong-stability-preserving RK3 scheme can be written in terms of the
position `Xⁿ` at the beginning of the time step and the displacement `D` from `Xⁿ`, with
`D¹ = Δt U⁰`, `D² = (D¹ + Δt U¹) / 4`, `D³ = 2 (D² + Δt U²) / 3`, and `X⁽ᵏ⁾ = Xⁿ + Dᵏ`:

```julia
using Oceananigans.Models.LagrangianParticleTracking: LagrangianParticleTracking as LPT

struct SSPStage{K} end

LPT.particle_time_stepper_state(::SSPRK3, x) = (x⁰ = similar(x, 0), y⁰ = similar(x, 0), z⁰ = similar(x, 0),
                                                Dx = similar(x, 0), Dy = similar(x, 0), Dz = similar(x, 0))
LPT.particle_stages(::SSPRK3) = 3
LPT.particle_stage(::SSPRK3, stage, restart) = SSPStage{stage}()

@inline function LPT.advance_particle_position(::SSPStage{1}, (x, y, z), (ẋ, ẏ, ż), Δt, state, p)
    D = (Δt * ẋ, Δt * ẏ, Δt * ż)
    # store (x, y, z) in state.x⁰[p], ... and D in state.Dx[p], ...
    return (x + D[1], y + D[2], z + D[3])
end
```

and similarly for stages 2 and 3, which load `Xⁿ` and `D` from `state`.
The returned position is "unwrapped": boundary conditions are applied after `advance_particle_position`.
"""
substep_lagrangian_particles!(particles::LagrangianParticles, model, Δt, stage) =
    advance_lagrangian_particles!(particles, model, Δt, model.timestepper, stage)
