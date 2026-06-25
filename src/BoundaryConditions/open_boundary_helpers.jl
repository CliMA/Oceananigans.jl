# Helper functions for open boundary conditions
# Simplifies user setup for common use cases

using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Oceananigans.Grids: immersed_peripheral_node, immersed_inactive_node

"""
    FlatherOpenBoundaries(grid; gravitational_acceleration=9.81)

Create Flather open boundary conditions for barotropic transport.

Returns a named tuple `(boundary_conditions, state_fields, update_callback)`:
- `boundary_conditions`: NamedTuple with `U` and `V` FieldBoundaryConditions
- `state_fields`: NamedTuple with internal state fields `(u, v, η)`
- `update_callback`: Function that returns a callback to update state fields

# Example

```julia
using Oceananigans
using Oceananigans.BoundaryConditions: FlatherOpenBoundaries

# Create Flather BCs
bcs, state, make_update_callback = FlatherOpenBoundaries(grid)

# Setup model
model = HydrostaticFreeSurfaceModel(grid;
    barotropic_transport_bcs = bcs)

# Create simulation and add update callback
simulation = Simulation(model; Δt=10, stop_time=1hour)
simulation.callbacks[:update_flather_bcs] = Callback(make_update_callback(simulation),
                                                      IterationInterval(1))

# Run
run!(simulation)
```

# Arguments
- `grid`: The ocean grid (must be `Bounded` in horizontal directions)
- `gravitational_acceleration`: Gravitational acceleration (default: 9.81 m/s²)

# Notes
- Automatically handles GPU compatibility (distributed or single GPU)
- State fields are updated each timestep via the callback
- Works transparently with immersed boundaries
"""
function FlatherOpenBoundaries(grid; gravitational_acceleration=9.81)

    # Create internal state fields (updated each timestep)
    u_state = Field{Face, Center, Center}(grid)
    v_state = Field{Center, Face, Center}(grid)
    η_state = Field{Center, Center, Nothing}(grid)

    # Wetcell check helper
    @inline function wetcell(i, j, k, grid, ℓx, ℓy, ℓz)
        return !immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) &
               !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)
    end

    # Vertical integration helper
    @inline function vertical_integral(i, j, grid, u_data, Δz_func, ℓx, ℓy, ℓz)
        Nz = size(grid, 3)
        integral = zero(eltype(grid))
        for k in 1:Nz
            wet = wetcell(i, j, k, grid, ℓx, ℓy, ℓz)
            @inbounds integral += ifelse(wet, u_data[i, j, k] * Δz_func(i, j, k, grid, ℓx(), ℓy(), ℓz()), zero(integral))
        end
        return integral
    end

    # Boundary condition functions (return (U, η) tuple for Flather)
    @inline function west_U_bc(j, k, grid, clock, fields, p)
        U = vertical_integral(1, j, grid, p.u, Δzᶠᶜᶜ, Face(), Center(), Center())
        η = @inbounds p.η[1, j, 1]
        return (U, η)
    end

    @inline function east_U_bc(j, k, grid, clock, fields, p)
        i = grid.Nx + 1
        U = vertical_integral(i, j, grid, p.u, Δzᶠᶜᶜ, Face(), Center(), Center())
        η = @inbounds p.η[i, j, 1]
        return (U, η)
    end

    @inline function south_V_bc(i, k, grid, clock, fields, p)
        V = vertical_integral(i, 1, grid, p.v, Δzᶜᶠᶜ, Center(), Face(), Center())
        η = @inbounds p.η[i, 1, 1]
        return (V, η)
    end

    @inline function north_V_bc(i, k, grid, clock, fields, p)
        j = grid.Ny + 1
        V = vertical_integral(i, j, grid, p.v, Δzᶜᶠᶜ, Center(), Face(), Center())
        η = @inbounds p.η[i, j, 1]
        return (V, η)
    end

    # Create boundary conditions (use .data for GPU compatibility)
    params = (u = u_state.data, v = v_state.data, η = η_state.data)

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
        west  = FlatherBoundaryCondition(west_U_bc, discrete_form=true,
                                         parameters=params,
                                         gravitational_acceleration=gravitational_acceleration),
        east  = FlatherBoundaryCondition(east_U_bc, discrete_form=true,
                                         parameters=params,
                                         gravitational_acceleration=gravitational_acceleration))

    V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
        south = FlatherBoundaryCondition(south_V_bc, discrete_form=true,
                                         parameters=params,
                                         gravitational_acceleration=gravitational_acceleration),
        north = FlatherBoundaryCondition(north_V_bc, discrete_form=true,
                                         parameters=params,
                                         gravitational_acceleration=gravitational_acceleration))

    boundary_conditions = (U = U_bcs, V = V_bcs)
    state_fields = (u = u_state, v = v_state, η = η_state)

    # Update callback factory (returns a callback that updates state fields)
    function make_update_callback(simulation)
        return function update_flather_state(sim)
            # Copy current model state to BC parameter fields
            u_state .= sim.model.velocities.u
            v_state .= sim.model.velocities.v
            η_state .= sim.model.free_surface.η
        end
    end

    return (boundary_conditions = boundary_conditions,
            state_fields = state_fields,
            update_callback = make_update_callback)
end
