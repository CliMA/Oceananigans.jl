struct Buoyancy{M, G}
                        model :: M
    gravitational_unit_vector :: G
end

function Buoyancy(; model, gravitational_unit_vector=(0, 0, 1))
    length(gravitational_unit_vector) == 3 ||
        throw(ArgumentError("gravitational_unit_vector must have length 3"))

    gx, gy, gz = gravitational_unit_vector

    gx^2 + gy^2 + gz^2 ≈ 1 ||
        throw(ArgumentError("gravitational_unit_vector must be a unit vector with gx² + gy² + gz² = 1"))

    return Buoyancy(model, gravitational_unit_vector)
end

@inline required_tracers(bm::Buoyancy) = required_tracers(bm.model)

@inline get_temperature_and_salinity(bm::Buoyancy, C) = get_temperature_and_salinity(bm.model, C)

@inline ĝ_x(buoyancy_model) = @inbounds buoyancy_model.gravitational_unit_vector[1]
@inline ĝ_y(buoyancy_model) = @inbounds buoyancy_model.gravitational_unit_vector[2]
@inline ĝ_z(buoyancy_model) = @inbounds buoyancy_model.gravitational_unit_vector[3]
