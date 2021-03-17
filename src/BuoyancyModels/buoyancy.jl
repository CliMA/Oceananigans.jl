struct Buoyancy{M, G}
                        model :: M
    gravitational_unit_vector :: G
end

function Buoyancy(; model, gravitational_unit_vector=(0, 0, 1))
    length(gravitational_unit_vector) == 3 ||
        throw(ArgumentError("gravitational_unit_vector must have length 3"))

    gx, gy, gz = gravitational_unit_vector

    gx^2 + gy^2 + gz^2 ≈ 1 ||
        throw(ArgumentError("gravitational_unit_vector must be a unit vector with g[1]² + g[2]² + g[3]² = 1"))

    return Buoyancy(model, gravitational_unit_vector)
end

@inline ĝ_x(buoyancy) = @inbounds buoyancy.gravitational_unit_vector[1]
@inline ĝ_y(buoyancy) = @inbounds buoyancy.gravitational_unit_vector[2]
@inline ĝ_z(buoyancy) = @inbounds buoyancy.gravitational_unit_vector[3]

#####
##### For convinience
#####

@inline required_tracers(bm::Buoyancy) = required_tracers(bm.model)

@inline get_temperature_and_salinity(bm::Buoyancy, C) = get_temperature_and_salinity(bm.model, C)

@inline ∂x_b(i, j, k, grid, b::Buoyancy, C) = ∂x_b(i, j, k, grid, b.model, C)
@inline ∂y_b(i, j, k, grid, b::Buoyancy, C) = ∂y_b(i, j, k, grid, b.model, C)
@inline ∂z_b(i, j, k, grid, b::Buoyancy, C) = ∂z_b(i, j, k, grid, b.model, C)

regularize_buoyancy(b) = b
regularize_buoyancy(b::AbstractBuoyancyModel) = Buoyancy(model=b)
