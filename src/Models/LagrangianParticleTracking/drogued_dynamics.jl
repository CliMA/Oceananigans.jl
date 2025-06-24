"""
    DroguedDynamics(depths)

`DroguedDynamics` goes in the `dynamics` slot of `LagrangianParticles` 
and modifies their behaviour to mimic the behaviour of bouys which are 
drogued at `depths`. The particles remain at the their `z` position
so the "measurment depth can be set", and then are advected in the `x` and `y`
directions according to the velocity field at `depths`.

`depths` should be an array of length `length(particles)`.
"""
struct DroguedDynamics{DD}
    depths :: DD
end

Adapt.adapt_structure(to, dbd::DroguedDynamics) = 
    DroguedDynamics(adapt(to, dbd.depths))

@inline (::DroguedDynamics)(args...) = nothing

const DroguedParticles = LagrangianParticles{<:Any, <:Any, <:Any, <:DroguedDynamics}

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
