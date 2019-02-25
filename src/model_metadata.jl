struct ModelMetadata
    arch::Symbol
    float_type::DataType
end

# _ prefix given to avoid conflict with ModelMetadata(::Symbol, ::DataType)
# method defined by the struct itself. We want some checking of the inputs.
function _ModelMetadata(arch=:cpu, float_type=Float64)
    @assert arch in [:cpu, :gpu] "Only :cpu and :gpu architectures are currently supported."
    @assert float_type in [Float64, Float32, Float16] "Only 64-bit, 32-bit, and 16-bit floats are currently supported."

    ModelMetadata(arch, float_type)
end
