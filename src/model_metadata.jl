struct ModelMetadata
    arch::Symbol
    float_type::DataType

    function ModelMetadata(arch=:CPU, float_type=Float64)
        @assert arch in [:CPU, :GPU] "Only :CPU and :GPU architectures are currently supported."
        @assert float_type in [Float64, Float32, Float16] "Only 64-bit, 32-bit, and 16-bit floats are currently supported."
        new(arch, float_type)
    end
end
