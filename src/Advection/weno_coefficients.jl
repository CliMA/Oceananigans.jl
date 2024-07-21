# Just change the coefficients and the smoothness

# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)
@inline smoothness_tuple(::Val{2}, ::Val{0}) = (1.0, -2.0, 1.0)
@inline smoothness_tuple(::Val{2}, ::Val{1}) = (1.0, -2.0, 1.0)

@inline smoothness_tuple(::Val{3}, ::Val{0}) = (10.0, -310.0, 11.0, 25.0, -19.0,  4.0)
@inline smoothness_tuple(::Val{3}, ::Val{1}) = (4.0,  -130.0,  5.0, 13.0, -13.0,  4.0)
@inline smoothness_tuple(::Val{3}, ::Val{2}) = (4.0,  -190.0, 11.0, 25.0, -31.0, 10.0)

@inline smoothness_tuple(::Val{4}, ::Val{0}) = (2.107,  -9.402, 7.042, -1.854, 11.003,  -17.246,  4.642,  7.043,  -3.882, 0.547)
@inline smoothness_tuple(::Val{4}, ::Val{1}) = (0.547,  -2.522, 1.922, -0.494,  3.443,  - 5.966,  1.602,  2.843,  -1.642, 0.267)
@inline smoothness_tuple(::Val{4}, ::Val{2}) = (0.267,  -1.642, 1.602, -0.494,  2.843,  - 5.966,  1.922,  3.443,  -2.522, 0.547)
@inline smoothness_tuple(::Val{4}, ::Val{3}) = (0.547,  -3.882, 4.642, -1.854,  7.043,  -17.246,  7.042, 11.003,  -9.402, 2.107)

@inline smoothness_tuple(::Val{5}, ::Val{0}) = (1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)
@inline smoothness_tuple(::Val{5}, ::Val{1}) = (0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)
@inline smoothness_tuple(::Val{5}, ::Val{2}) = (0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)
@inline smoothness_tuple(::Val{5}, ::Val{3}) = (0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)
@inline smoothness_tuple(::Val{5}, ::Val{4}) = (0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)

@inline reconstruction_tuple(::Val{S}, ::Val{B}) where {S, B} = stencil_coefficients(50, B, collect(1:100), collect(1:100); order = S)

@eval begin
    # Third order WENO -> no lower order possible
    @inline p_coefficients(order, ::Val{2}, ::Val{0}) = $(reconstruction_tuple(Val(2), Val(0)))
    @inline p_coefficients(order, ::Val{2}, ::Val{1}) = $(reconstruction_tuple(Val(2), Val(1)))
    
    # Fifth order WENO -> can li$coefficienmit to third order WENO
    @inline function p_coefficients(order, ::Val{3}, ::Val{0}) 
        βₙ   =  $(reconstruction_tuple(Val(3), Val(0)))
        βₙ₋₁ = $((reconstruction_tuple(Val(2), Val(0))..., 0))
        return ifelse(order == 3, βₙ, βₙ₋₁)
    end

    @inline function p_coefficients(order, ::Val{3}, ::Val{1}) 
        βₙ   =  $(reconstruction_tuple(Val(3), Val(1)))
        βₙ₋₁ = $((reconstruction_tuple(Val(2), Val(1))..., 0))
        return ifelse(order == 3, βₙ, βₙ₋₁)
    end

    @inline function p_coefficients(order, ::Val{3}, ::Val{2})
        βₙ   = $(reconstruction_tuple(Val(3), Val(2)))
        βₙ₋₁ = $(0 .* reconstruction_tuple(Val(3), Val(2)))
        return ifelse(order == 3, βₙ, βₙ₋₁)
    end

    # Seventh order WENO -> can limit to both fifth and third order WENO
    @inline function p_coefficients(order, ::Val{4}, ::Val{0}) 
        βₙ   =  $(reconstruction_tuple(Val(4), Val(0)))
        βₙ₋₁ = $((reconstruction_tuple(Val(3), Val(0))..., 0))
        βₙ₋₂ = $((reconstruction_tuple(Val(2), Val(0))..., 0, 0))
        return ifelse(order == 4, βₙ, ifelse(order == 3, βₙ₋₁, βₙ₋₂))
    end

    @inline function p_coefficients(order, ::Val{4}, ::Val{1}) 
        βₙ   =  $(reconstruction_tuple(Val(4), Val(1)))
        βₙ₋₁ = $((reconstruction_tuple(Val(3), Val(1))..., 0))
        βₙ₋₂ = $((reconstruction_tuple(Val(2), Val(1))..., 0, 0))
        return ifelse(order == 4, βₙ, ifelse(order == 3, βₙ₋₁, βₙ₋₂))
    end

    @inline function p_coefficients(order, ::Val{4}, ::Val{2})
        βₙ   =  $(reconstruction_tuple(Val(4), Val(2)))
        βₙ₋₁ = $((reconstruction_tuple(Val(3), Val(2))..., 0))
        βₙ₋₂ = $(0 .* reconstruction_tuple(Val(4), Val(2)))
        return ifelse(order == 4, βₙ, ifelse(order == 3, βₙ₋₁, βₙ₋₂))
    end

    @inline function p_coefficients(order, ::Val{4}, ::Val{3})
        βₙ   = $(reconstruction_tuple(Val(4), Val(3)))
        βₙ₋₁ = $(0 .* reconstruction_tuple(Val(4), Val(3)))
        return ifelse(order == 4, βₙ, βₙ₋₁)
    end

    # Ninth order WENO -> can limit to seventh, fifth and third order WENO
    @inline function p_coefficients(order, ::Val{5}, ::Val{0}) 
        βₙ   =  $(reconstruction_tuple(Val(5), Val(0)))
        βₙ₋₁ = $((reconstruction_tuple(Val(4), Val(0))..., 0))
        βₙ₋₂ = $((reconstruction_tuple(Val(3), Val(0))..., 0, 0))
        βₙ₋₃ = $((reconstruction_tuple(Val(2), Val(0))..., 0, 0, 0))
        return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, ifelse(order == 3, βₙ₋₂, βₙ₋₃)))
    end

    @inline function p_coefficients(order, ::Val{5}, ::Val{1}) 
        βₙ   =  $(reconstruction_tuple(Val(5), Val(1)))
        βₙ₋₁ = $((reconstruction_tuple(Val(4), Val(1))..., 0))
        βₙ₋₂ = $((reconstruction_tuple(Val(3), Val(1))..., 0, 0))
        βₙ₋₃ = $((reconstruction_tuple(Val(2), Val(0))..., 0, 0, 0))
        return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, ifelse(order == 3, βₙ₋₂, βₙ₋₃)))
    end

    @inline function p_coefficients(order, ::Val{5}, ::Val{2})
        βₙ   =  $(reconstruction_tuple(Val(5), Val(2)))
        βₙ₋₁ = $((reconstruction_tuple(Val(4), Val(2))..., 0))
        βₙ₋₂ = $((reconstruction_tuple(Val(3), Val(2))..., 0, 0))  
        βₙ₋₃ = $(0 .* reconstruction_tuple(Val(5),  Val(2)))
        return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, ifelse(order == 3, βₙ₋₂, βₙ₋₃)))
    end

    @inline function p_coefficients(order, ::Val{5}, ::Val{3})
        βₙ   =  $(reconstruction_tuple(Val(5), Val(3)))
        βₙ₋₁ = $((reconstruction_tuple(Val(4), Val(3))..., 0))
        βₙ₋₂ = $(0 .* reconstruction_tuple(Val(5),  Val(3)))
        return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, βₙ₋₂))
    end

    @inline function p_coefficients(order, ::Val{5}, ::Val{4})
        βₙ   = $(reconstruction_tuple(Val(5),  Val(4)))
        βₙ₋₁ = $(0 .* reconstruction_tuple(Val(5),  Val(4)))
        return ifelse(order == 5, βₙ, βₙ₋₁)
    end
end

for buffer in [2, 3, 4, 5]
    for stencil in collect(0:buffer - 1)
        @eval @inline β_coefficients(::Val{$buffer}, ::Val{$buffer}, ::Val{$stencil}) = $(smoothness_tuple(Val(buffer), Val(stencil)))
    end 
end

@eval begin
    @inline β_coefficients(::Val{3}, ::Val{2}, ::Val{0}) = $(smoothness_tuple(Val(2), Val(0))[1:2]..., 0.0, smoothness_tuple(Val(2), Val(0))[3], 0.0, 0.0)
    @inline β_coefficients(::Val{3}, ::Val{2}, ::Val{1}) = $(smoothness_tuple(Val(2), Val(1))[1:2]..., 0.0, smoothness_tuple(Val(2), Val(1))[3], 0.0, 0.0)
    @inline β_coefficients(::Val{3}, ::Val{2}, ::Val{2}) = $(0 .* smoothness_tuple(Val(3), Val(2)))

    @inline β_coefficients(::Val{4}, ::Val{2}, ::Val{0}) = $(smoothness_tuple(Val(2), Val(0))[1:2]..., 0.0, 0.0, smoothness_tuple(Val(2), Val(0))[3], zeros(5)...)
    @inline β_coefficients(::Val{4}, ::Val{2}, ::Val{1}) = $(smoothness_tuple(Val(2), Val(1))[1:2]..., 0.0, 0.0, smoothness_tuple(Val(2), Val(1))[3], zeros(5)...)
    @inline β_coefficients(::Val{4}, ::Val{2}, ::Val{2}) = $(0 .* smoothness_tuple(Val(4), Val(2)))
    @inline β_coefficients(::Val{4}, ::Val{2}, ::Val{3}) = $(0 .* smoothness_tuple(Val(4), Val(3)))

    @inline β_coefficients(::Val{4}, ::Val{3}, ::Val{0}) = $(smoothness_tuple(Val(3), Val(0))[1:3]..., 0.0, smoothness_tuple(Val(3), Val(0))[4:5]..., 0.0, smoothness_tuple(Val(3), Val(0))[6], 0.0, 0.0)
    @inline β_coefficients(::Val{4}, ::Val{3}, ::Val{1}) = $(smoothness_tuple(Val(3), Val(1))[1:3]..., 0.0, smoothness_tuple(Val(3), Val(1))[4:5]..., 0.0, smoothness_tuple(Val(3), Val(1))[6], 0.0, 0.0)
    @inline β_coefficients(::Val{4}, ::Val{3}, ::Val{2}) = $(smoothness_tuple(Val(3), Val(2))[1:3]..., 0.0, smoothness_tuple(Val(3), Val(2))[4:5]..., 0.0, smoothness_tuple(Val(3), Val(2))[6], 0.0, 0.0)
    @inline β_coefficients(::Val{4}, ::Val{3}, ::Val{3}) = $(0 .* smoothness_tuple(Val(4), Val(3)))

    @inline β_coefficients(::Val{5}, ::Val{2}, ::Val{0}) = $(smoothness_tuple(Val(2), Val(0))[1:2]..., 0.0, 0.0, 0.0, smoothness_tuple(Val(2), Val(0))[3], zeros(9)...)
    @inline β_coefficients(::Val{5}, ::Val{2}, ::Val{1}) = $(smoothness_tuple(Val(2), Val(1))[1:2]..., 0.0, 0.0, 0.0, smoothness_tuple(Val(2), Val(1))[3], zeros(9)...)
    @inline β_coefficients(::Val{5}, ::Val{2}, ::Val{2}) = $(0 .* smoothness_tuple(Val(5), Val(2)))
    @inline β_coefficients(::Val{5}, ::Val{2}, ::Val{3}) = $(0 .* smoothness_tuple(Val(5), Val(3)))
    @inline β_coefficients(::Val{5}, ::Val{2}, ::Val{4}) = $(0 .* smoothness_tuple(Val(5), Val(4)))

    @inline β_coefficients(::Val{5}, ::Val{3}, ::Val{0}) = $(smoothness_tuple(Val(3), Val(0))[1:3]..., 0.0, 0.0, smoothness_tuple(Val(3), Val(0))[4:5]..., 0.0, 0.0, smoothness_tuple(Val(3), Val(0))[6], zeros(5)...)
    @inline β_coefficients(::Val{5}, ::Val{3}, ::Val{1}) = $(smoothness_tuple(Val(3), Val(1))[1:3]..., 0.0, 0.0, smoothness_tuple(Val(3), Val(1))[4:5]..., 0.0, 0.0, smoothness_tuple(Val(3), Val(1))[6], zeros(5)...)
    @inline β_coefficients(::Val{5}, ::Val{3}, ::Val{2}) = $(smoothness_tuple(Val(3), Val(2))[1:3]..., 0.0, 0.0, smoothness_tuple(Val(3), Val(2))[4:5]..., 0.0, 0.0, smoothness_tuple(Val(3), Val(2))[6], zeros(5)...)
    @inline β_coefficients(::Val{5}, ::Val{3}, ::Val{3}) = $(0 .* smoothness_tuple(Val(5), Val(3)))
    @inline β_coefficients(::Val{5}, ::Val{3}, ::Val{4}) = $(0 .* smoothness_tuple(Val(5), Val(4)))

    @inline β_coefficients(::Val{5}, ::Val{4}, ::Val{0}) = $(smoothness_tuple(Val(4), Val(0))[1:4]..., 0.0, smoothness_tuple(Val(4), Val(0))[5:7]..., 0.0, smoothness_tuple(Val(4), Val(0))[8:9]..., 0.0, smoothness_tuple(Val(4), Val(0))[10], 0.0, 0.0)
    @inline β_coefficients(::Val{5}, ::Val{4}, ::Val{1}) = $(smoothness_tuple(Val(4), Val(1))[1:4]..., 0.0, smoothness_tuple(Val(4), Val(1))[5:7]..., 0.0, smoothness_tuple(Val(4), Val(1))[8:9]..., 0.0, smoothness_tuple(Val(4), Val(1))[10], 0.0, 0.0)
    @inline β_coefficients(::Val{5}, ::Val{4}, ::Val{2}) = $(smoothness_tuple(Val(4), Val(2))[1:4]..., 0.0, smoothness_tuple(Val(4), Val(2))[5:7]..., 0.0, smoothness_tuple(Val(4), Val(2))[8:9]..., 0.0, smoothness_tuple(Val(4), Val(2))[10], 0.0, 0.0)
    @inline β_coefficients(::Val{5}, ::Val{4}, ::Val{3}) = $(smoothness_tuple(Val(4), Val(3))[1:4]..., 0.0, smoothness_tuple(Val(4), Val(3))[5:7]..., 0.0, smoothness_tuple(Val(4), Val(3))[8:9]..., 0.0, smoothness_tuple(Val(4), Val(3))[10], 0.0, 0.0)
    @inline β_coefficients(::Val{5}, ::Val{4}, ::Val{4}) = $(0 .* smoothness_tuple(Val(5), Val(4)))
end

for buffer in [2, 3, 4, 5]
    for stencil in collect(0:1:buffer-1)
        @eval begin
            @inline  coeff_left_p(scheme::WENO{$buffer, FT}, order, ::Val{$stencil}, ::Type{Nothing}, args...) where FT = @inbounds FT.(p_coefficients(order, Val($buffer), Val($stencil)))
            @inline coeff_right_p(scheme::WENO{$buffer, FT}, order, ::Val{$stencil}, ::Type{Nothing}, args...) where FT = @inbounds FT.(p_coefficients(order, Val($buffer), Val($stencil)))

            # stretched coefficients are retrieved from precalculated coefficients
            # stretched coefficients are retrieved from precalculated coefficients
            @inline  coeff_left_p(scheme::WENO{$buffer}, order, ::Val{$stencil}, T, dir, i, loc) = @inbounds retrieve_coeff(scheme, $stencil,     dir, i, loc)
            @inline coeff_right_p(scheme::WENO{$buffer}, order, ::Val{$stencil}, T, dir, i, loc) = @inbounds retrieve_coeff(scheme, $(stencil-1), dir, i, loc)
        end
    end
end