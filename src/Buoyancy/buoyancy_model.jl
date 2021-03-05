struct BuoyancyModel{M, G}
                  model :: M
    gravity_unit_vector :: G
end

function BuoyancyModel(model; gravity_unit_vector=(0, 0, 1))
    @assert length(gravity_unit_vector) == 3
    gx, gy, gz = gravity_unit_vector
    @assert gx^2 + gy^2 + gz^2 ≈ 1

    return BuoyancyModel(model, gravity_unit_vector)
end
