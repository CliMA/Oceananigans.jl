struct ModelConfiguration
    κh # Horizontal Laplacian heat diffusion [m²/s]. diffKhT in MITgcm.
    κv # Vertical Laplacian heat diffusion [m²/s]. diffKzT in MITgcm.
    𝜈h # Horizontal eddy viscosity [m²/s]. viscAh in MITgcm.
    𝜈v # Vertical eddy viscosity [m²/s]. viscAz in MITgcm.

    function ModelConfiguration(κh, κv, 𝜈h, 𝜈v)
        @assert κh >= 0
        @assert κv >= 0
        @assert 𝜈h >= 0
        @assert 𝜈v >= 0
        new(κh, κv, 𝜈h, 𝜈v)
    end
end
