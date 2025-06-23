"""
    DroguedBuoyDynamics(drogue_depths)

`DroguedBuoyDynamics` goes in the `dynamics` slot of `LagrangianParticles` 
and modifies their behaviour to mimic the behaviour of bouys which are 
drogued at `drogue_depths`. The particles remain at the their `z` position
so the "measurment depth can be set", and then are advected in the `x` and `y`
directions according to the velocity field at `drogue_depths`.

`drogue_depths` should be an array of length `length(particles)`.
"""
struct DroguedBuoyDynamics{DD}
    drogue_depths :: DD
end

@inline (::DroguedBuoyDynamics)(args...) = nothing

const DroguedBuoyParticle = LagrangianParticles{<:Any, <:Any, <:Any, <:DroguedBuoyDynamics}

function advect_lagrangian_particles!(particles::DroguedBuoyParticle, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    parameters = KernelParameters(1:length(particles))

    launch!(arch, grid, parameters,
            _advect_drogued_particles!,
            particles.properties, particles.restitution, model.grid, Δt, total_velocities(model), particles.dynamics.drogue_depths)

    return nothing
end

@kernel function _advect_drogued_particles!(properties, restitution, grid, Δt, velocities, drogue_depths)
    p = @index(Global)

    @inbounds begin
        x = properties.x[p]
        y = properties.y[p]
        z = drogue_depths[p]
    end

    x⁺, y⁺, _ = advect_particle((x, y, z), properties, p, restitution, grid, Δt, velocities)

    @inbounds begin
        properties.x[p] = x⁺
        properties.y[p] = y⁺
    end
end
