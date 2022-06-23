#####
##### Stretched smoothness indicators gathered from precomputed values.
##### The stretched values for β coefficients are calculated from 
##### Shu, NASA/CR-97-206253, ICASE Report No. 97-65
##### by hardcoding that p(x) is a 2nd order polynomial
#####

@inline function biased_left_β(ψ, scheme, r, dir, i, location) 
    @inbounds begin
        stencil = retrieve_left_smooth(scheme, r, dir, i, location)
        wᵢᵢ = stencil[1]   
        wᵢⱼ = stencil[2]
        result = 0
        @unroll for j = 1:3
            result += ψ[j] * ( wᵢᵢ[j] * ψ[j] + wᵢⱼ[j] * dagger(ψ)[j] )
        end
    end
    return result
end

@inline function biased_right_β(ψ, scheme, r, dir, i, location) 
    @inbounds begin
        stencil = retrieve_right_smooth(scheme, r, dir, i, location)
        wᵢᵢ = stencil[1]   
        wᵢⱼ = stencil[2]
        result = 0
        @unroll for j = 1:3
            result += ψ[j] * ( wᵢᵢ[j] * ψ[j] + wᵢⱼ[j] * dagger(ψ)[j] )
        end
    end
    return result
end

@inline left_biased_β₀(FT, ψ, T, scheme, args...) = biased_left_β(ψ, scheme, 0, args...) 
@inline left_biased_β₁(FT, ψ, T, scheme, args...) = biased_left_β(ψ, scheme, 1, args...) 
@inline left_biased_β₂(FT, ψ, T, scheme, args...) = biased_left_β(ψ, scheme, 2, args...) 

@inline right_biased_β₀(FT, ψ, T, scheme, args...) = biased_right_β(ψ, scheme, 2, args...) 
@inline right_biased_β₁(FT, ψ, T, scheme, args...) = biased_right_β(ψ, scheme, 1, args...) 
@inline right_biased_β₂(FT, ψ, T, scheme, args...) = biased_right_β(ψ, scheme, 0, args...) 

@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+1][i] 

@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+4][i] 

@inline calc_smoothness_coefficients(FT, ::Val{false}, args...; kwargs...) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N; order) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::AbstractRange, arch, N; order) = nothing

function calc_smoothness_coefficients(FT, beta, coord, arch, N; order) 

    cpu_coord = arch_array(CPU(), coord)

    order == 3 || throw(ArgumentError("The stretched smoothness coefficients are only implemented for order == 3"))
    
    s1 = create_smoothness_coefficients(FT, 0, -, cpu_coord, arch, N; order)
    s2 = create_smoothness_coefficients(FT, 1, -, cpu_coord, arch, N; order)
    s3 = create_smoothness_coefficients(FT, 2, -, cpu_coord, arch, N; order)
    s4 = create_smoothness_coefficients(FT, 0, +, cpu_coord, arch, N; order)
    s5 = create_smoothness_coefficients(FT, 1, +, cpu_coord, arch, N; order)
    s6 = create_smoothness_coefficients(FT, 2, +, cpu_coord, arch, N; order)
    
    return (s1, s2, s3, s4, s5, s6)
end

function create_smoothness_coefficients(FT, r, op, cpu_coord, arch, N; order)

    # derivation written on overleaf
    stencil = NTuple{2, NTuple{order, FT}}[]   
    @inbounds begin
        for i = 0:N+1
       
            bias1 = Int(op == +)
            bias2 = bias1 - 1

            Δcᵢ = cpu_coord[i + bias1] - cpu_coord[i + bias2]
        
            Bᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1, der = Primitive())
            bᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1)
            bₓᵢ = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1, der = FirstDerivative())
            Aᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias2, der = Primitive())
            aᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias2)
            aₓᵢ = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias2, der = FirstDerivative())
            pₓₓ = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1, der = SecondDerivative())

            Pᵢ  =  (Bᵢ .- Aᵢ)

            wᵢᵢ = Δcᵢ  .* (bᵢ .* bₓᵢ .- aᵢ .* aₓᵢ .- pₓₓ .* Pᵢ)              .+ Δcᵢ^4 .* (pₓₓ .* pₓₓ)
            wᵢⱼ = Δcᵢ  .* (star(bᵢ, bₓᵢ)  .- star(aᵢ, aₓᵢ) .- star(pₓₓ, Pᵢ)) .+ Δcᵢ^4 .* star(pₓₓ, pₓₓ)

            push!(stencil, (wᵢᵢ, wᵢⱼ))
        end
    end

    return OffsetArray(arch_array(arch, stencil), -1)
end

@inline dagger(ψ)    = (ψ[2], ψ[3], ψ[1])
@inline star(ψ₁, ψ₂) = (ψ₁ .* dagger(ψ₂) .+ dagger(ψ₁) .* ψ₂)

