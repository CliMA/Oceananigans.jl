struct ModelConfiguration{T<:AbstractFloat}
    κh::T  # Horizontal Laplacian heat diffusion [m²/s]. diffKhT in MITgcm.
    κv::T  # Vertical Laplacian heat diffusion [m²/s]. diffKzT in MITgcm.
    𝜈h::T  # Horizontal eddy viscosity [m²/s]. viscAh in MITgcm.
    𝜈v::T  # Vertical eddy viscosity [m²/s]. viscAz in MITgcm.
    smag_coeff::T   # Smagorinsky coefficient
    Prandtl_num::T  # Prandtl number used with Smagorinsky closure

    function ModelConfiguration(κh, κv, 𝜈h, 𝜈v, smag_coeff, Prandtl_num)
        @assert κh >= 0
        @assert κv >= 0
        @assert 𝜈h >= 0
        @assert 𝜈v >= 0
        new{Float64}(κh, κv, 𝜈h, 𝜈v, smag_coeff, Prandtl_num)
    end
end
