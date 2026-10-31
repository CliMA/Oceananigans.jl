"""
    DroguedParticleDynamics(depths)

`DroguedParticleDynamics` goes in the `dynamics` slot of `LagrangianParticles`
and modifies their behavior to mimic the behavior of buoys that are
drogued at `depths`. The particles remain at their `z` position
so the "measurement depth can be set", and then are advected in the `x` and `y`
directions according to the velocity field at `depths`.

`depths` should be an (abstract) array of length `length(particles)`.


Example
=======

```jldoctest
julia> using Oceananigans

julia> n = 10
10

julia> dynamics = DroguedParticleDynamics(-10:10/n:0)
DroguedParticleDynamics{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}(-10.0:1.0:0.0)

julia> particles = LagrangianParticles(; x = zeros(n), y = zeros(n), z = zeros(n), dynamics)
10 LagrangianParticles with eltype Particle:
├── 3 properties: (:x, :y, :z)
├── boundary conditions: ParticleBoundaryConditions(ReflectingBoundary(1.0))
├── 0 tracked fields: ()
├── sorting: nothing
└── dynamics: DroguedParticleDynamics{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}

```
"""
struct DroguedParticleDynamics{DD}
    depths :: DD
end

Adapt.adapt_structure(to, dbd::DroguedParticleDynamics) =
    DroguedParticleDynamics(adapt(to, dbd.depths))

@inline (::DroguedParticleDynamics)(args...) = nothing

const DroguedParticles = LagrangianParticles{<:Any, <:Any, <:Any, <:DroguedParticleDynamics}

# Sorting and removal reorder particle properties, but not the depths of drogued particles
validate_reordering(dynamics, sorting, removal) = nothing

function validate_reordering(::DroguedParticleDynamics, sorting, removal)
    isnothing(sorting) && isnothing(removal) && return nothing
    throw(ArgumentError("DroguedParticleDynamics does not support sorting or AbsorbingBoundary conditions."))
end

function advect_lagrangian_particles!(particles::DroguedParticles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    Np = length(particles)
    Np == 0 && return nothing

    launch!(arch, grid, KernelParameters(1:Np),
            _advect_drogued_particles!,
            particles.properties, particles.boundary_conditions, model.grid, Δt, total_velocities(model),
            particles.dynamics.depths)

    return nothing
end

@kernel function _advect_drogued_particles!(properties, boundary_conditions, grid, Δt, velocities, depths)
    p = @index(Global)

    @inbounds begin
        x = properties.x[p]
        y = properties.y[p]
        z = depths[p]
    end

    (x⁺, y⁺, _), _ = advect_particle((x, y, z), properties, p, boundary_conditions, grid, Δt, velocities)

    @inbounds begin
        properties.x[p] = x⁺
        properties.y[p] = y⁺
    end
end
