"""
    DroguedParticleDynamics(depths)

`DroguedParticleDynamics` goes in the `dynamics` slot of `LagrangianParticles` 
and modifies their behaviour to mimic the behaviour of bouys which are 
drogued at `depths`. The particles remain at the their `z` position
so the "measurment depth can be set", and then are advected in the `x` and `y`
directions according to the velocity field at `depths`.

`depths` should be an (abstract) array of length `length(particles)`.


Example
=======

```jldoctest
using Oceananigans

n = 10

dynamics = DroguedParticleDynamics(-10:10/(n-1):0)

particles = LagrangianParticles(; x = zeros(n), y = zeros(n), z = zeros(n), dynamics)
```
"""
struct DroguedParticleDynamics{DD}
    depths :: DD
end

Adapt.adapt_structure(to, dbd::DroguedParticleDynamics) = 
    DroguedParticleDynamics(adapt(to, dbd.depths))

@inline (::DroguedParticleDynamics)(args...) = nothing

const DroguedParticles = LagrangianParticles{<:Any, <:Any, <:Any, <:DroguedParticleDynamics}

function advect_lagrangian_particles!(particles::DroguedParticles, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    parameters = KernelParameters(1:length(particles))

    launch!(arch, grid, parameters,
            _advect_drogued_particles!,
            particles.properties, particles.restitution, model.grid, Δt, total_velocities(model), particles.dynamics.depths)

    return nothing
end

@kernel function _advect_drogued_particles!(properties, restitution, grid, Δt, velocities, depths)
    p = @index(Global)

    @inbounds begin
        x = properties.x[p]
        y = properties.y[p]
        z = depths[p]
    end

    x⁺, y⁺, _ = advect_particle((x, y, z), properties, p, restitution, grid, Δt, velocities)

    @inbounds begin
        properties.x[p] = x⁺
        properties.y[p] = y⁺
    end
end
