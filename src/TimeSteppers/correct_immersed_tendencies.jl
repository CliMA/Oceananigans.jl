using Oceananigans.Grids: xnode, ynode, znode, Cell, AbstractGrid

"""
    correct_immersed_tendencies!(model)
    
Correct the tendency terms to implement no-slip boundary conditions on an immersed boundary
 without the contribution from the non-hydrostatic pressure. 
Makes velocity vanish within the immersed surface.
"""

correct_immersed_tendencies!(model, Δt, γⁿ, ζⁿ) =
    correct_immersed_tendencies!(model, model.immersed_boundary, Δt, γⁿ, ζⁿ)

# if no immersed boundary, do nothing (no cost)
correct_immersed_tendencies!(model, ::Nothing, Δt, γⁿ, ζⁿ) = nothing

# otherwise, unpack the model
function correct_immersed_tendencies!(model, immersed_boundary, Δt, γⁿ, ζⁿ)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    correct_immersed_velocities_tendencies_kernel! = _correct_velocities_tendencies!(device(model.architecture), workgroup, worksize)
    correct_immersed_tracer_tendencies_kernel! = _correct_tracer_tendencies!(device(model.architecture), workgroup, worksize)
    
    # events we want to occur, evaluate using kernel function
    # first the event to correct velocities
    correct_velocities_event =
            correct_immersed_velocities_tendencies_kernel!(model.grid,
                                            immersed_boundary,
                                            model.timestepper.Gⁿ,
                                            model.timestepper.G⁻,
                                            model.velocities,
                                            Δt, γⁿ, ζⁿ,
                                            dependencies=barrier)
    events = [correct_velocities_event]
    
    # then the events to correct tracers (if any, will start on 4th index)
    for i in 1:length(model.tracers)
        @inbounds c = model.tracers[i]
        @inbounds Gcⁿ = model.timestepper.Gⁿ[i+3]
        @inbounds Gc⁻ = model.timestepper.G⁻[i+3]

        correct_tracer_event = correct_immersed_tracer_tendencies_kernel!(c, Δt, γⁿ, ζⁿ, Gcⁿ, Gc⁻, immersed_boundary, dependencies=barrier, model.grid)

        push!(events, correct_tracer_event)
    end
    
    # wait for these things to happen before continuing in calculations
    wait(device(model.architecture), MultiEvent(Tuple(events)))
    
    return nothing
end

"""
Correct the tendency terms in the velocity for the nth stage of the 3rd-order RK method
for the presence of an immersed boundary

    `G^{n+1} = (-u^{n} - Δt ζⁿ G^{n-1})/(Δt γⁿ)`,
    
where `n` denotes the substage.
"""

@kernel function _correct_velocities_tendencies!(Gⁿ, grid::AbstractGrid{FT}, immersed, G⁻, velocities, Δt, γⁿ, ζⁿ) where FT
    i, j, k = @index(Global, NTuple)
    
    # evaluating x,y,z at cell centers to determine if boundary or not
    x = xnode(Cell, i, grid)
    y = ynode(Cell, j, grid)
    z = znode(Cell, k, grid)

    @inbounds begin
        # correcting velocity tendency terms: if immersd boundary gives true then correct tednecy, otherwise don't (it's a fluid node)
        Gⁿ.u[i, j, k] = ifelse(immersed(x, y, z),
                               - (velocities.u[i, j, k] + ζⁿ * Δt * G⁻.u[i, j, k]) / (γⁿ * Δt),
                               Gⁿ.u[i, j, k])

        Gⁿ.v[i, j, k] = ifelse(immersed(x, y, z),
                               - (velocities.v[i, j, k] + ζⁿ * Δt * G⁻.v[i, j, k]) / (γⁿ * Δt),
                               Gⁿ.v[i, j, k])

        Gⁿ.w[i, j, k] = ifelse(immersed(x, y, z),
                               - (velocities.w[i, j, k] + ζⁿ * Δt * G⁻.w[i, j, k]) / (γⁿ * Δt),
                               Gⁿ.w[i, j, k])
    end
end

"""
Correct the tendency terms in the tracer equations for the nth stage of the 3rd-order RK method for the presence of an immersed boundary

    `Gc^{n+1} = (-c^{n} - Δt ζⁿ Gc^{n-1})/(Δt γⁿ)`,
    
where `n` denotes the substage.
"""

@kernel function _correct_velocities_tendencies!(Gcⁿ, grid::AbstractGrid{FT}, immersed, Gc⁻, c, Δt, γⁿ, ζⁿ) where FT
    i, j, k = @index(Global, NTuple)
    
    # evaluating x,y,z at cell centers to determine if boundary or not
    x = xnode(Cell, i, grid)
    y = ynode(Cell, j, grid)
    z = znode(Cell, k, grid)
    
    @inbounds Gcⁿ[i, j, k] = ifelse(immersed(x, y, z),- (c[i, j, k] + ζⁿ * Δt * Gc⁻[i, j, k]) / (γⁿ * Δt), Gcⁿ[i, j, k])
end

