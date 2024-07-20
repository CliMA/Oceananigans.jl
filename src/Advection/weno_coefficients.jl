# Just change the coefficients and the smoothness

# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)
@inline smoothness_tuple(::Val{2}, ::Val{0}) = (1, -2, 1)
@inline smoothness_tuple(::Val{2}, ::Val{1}) = (1, -2, 1)

@inline smoothness_tuple(::Val{3}, ::Val{0}) = (10, -31, 11, 25, -19,  4)
@inline smoothness_tuple(::Val{3}, ::Val{1}) = (4,  -13, 5,  13, -13,  4)
@inline smoothness_tuple(::Val{3}, ::Val{2}) = (4,  -19, 11, 25, -31, 10)

@inline smoothness_tuple(::Val{4}, ::Val{0}) = (2.107,  -9.402, 7.042, -1.854, 11.003,  -17.246,  4.642,  7.043,  -3.882, 0.547)
@inline smoothness_tuple(::Val{4}, ::Val{1}) = (0.547,  -2.522, 1.922, -0.494,  3.443,  - 5.966,  1.602,  2.843,  -1.642, 0.267)
@inline smoothness_tuple(::Val{4}, ::Val{2}) = (0.267,  -1.642, 1.602, -0.494,  2.843,  - 5.966,  1.922,  3.443,  -2.522, 0.547)
@inline smoothness_tuple(::Val{4}, ::Val{3}) = (0.547,  -3.882, 4.642, -1.854,  7.043,  -17.246,  7.042, 11.003,  -9.402, 2.107)

@inline smoothness_tuple(::Val{5}, ::Val{0}) = (1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)
@inline smoothness_tuple(::Val{5}, ::Val{1}) = (0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)
@inline smoothness_tuple(::Val{5}, ::Val{2}) = (0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)
@inline smoothness_tuple(::Val{5}, ::Val{3}) = (0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)
@inline smoothness_tuple(::Val{5}, ::Val{4}) = (0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)

@inline smoothness_tuple(::Val{6}, ::Val{0}) = (0.6150211, -4.7460464, 7.6206736, -6.3394124, 2.7060170, -0.4712740,  9.4851237, -31.1771244, 26.2901672, -11.3206788,  1.9834350, 26.0445372, -44.4003904, 19.2596472, -3.3918804, 19.0757572, -16.6461044, 2.9442256, 3.6480687, -1.2950184, 0.1152561)
@inline smoothness_tuple(::Val{6}, ::Val{1}) = (0.1152561, -0.9117992, 1.4742480, -1.2183636, 0.5134574, -0.0880548,  1.9365967,  -6.5224244,  5.5053752,  -2.3510468,  0.4067018,  5.6662212,  -9.7838784,  4.2405032, -0.7408908,  4.3093692,  -3.7913324, 0.6694608, 0.8449957, -0.3015728, 0.0271779)
@inline smoothness_tuple(::Val{6}, ::Val{2}) = (0.0271779, -0.2380800, 0.4086352, -0.3462252, 0.1458762, -0.0245620,  0.5653317,  -2.0427884,  1.7905032,  -0.7727988,  0.1325006,  1.9510972,  -3.5817664,  1.5929912, -0.2792660,  1.7195652,  -1.5880404, 0.2863984, 0.3824847, -0.1429976, 0.0139633)
@inline smoothness_tuple(::Val{6}, ::Val{3}) = (0.0139633, -0.1429976, 0.2863984, -0.2792660, 0.1325006, -0.0245620,  0.3824847,  -1.5880404,  1.5929912,  -0.7727988,  0.1458762,  1.7195652,  -3.5817664,  1.7905032, -0.3462252,  1.9510972,  -2.0427884, 0.4086352, 0.5653317, -0.2380800, 0.0271779)
@inline smoothness_tuple(::Val{6}, ::Val{4}) = (0.0271779, -0.3015728, 0.6694608, -0.7408908, 0.4067018, -0.0880548,  0.8449957,  -3.7913324,  4.2405032,  -2.3510468,  0.5134574,  4.3093692,  -9.7838784,  5.5053752, -1.2183636,  5.6662212,  -6.5224244, 1.4742480, 1.9365967, -0.9117992, 0.1152561)
@inline smoothness_tuple(::Val{6}, ::Val{5}) = (0.1152561, -1.2950184, 2.9442256, -3.3918804, 1.9834350, -0.4712740,  3.6480687, -16.6461044, 19.2596472, -11.3206788,  2.7060170, 19.0757572, -44.4003904, 26.2901672, -6.3394124, 26.0445372, -31.1771244, 7.6206736, 9.4851237, -4.7460464, 0.6150211)


@inline reconstruction_tuple(::Val{S}, ::Val{B}) where {S, B} = stencil_coefficients(50, B, collect(1:100), collect(1:100); order = S)

# uniform coefficients are independent on direction and location
for (coefficients, coefficient_tuple) in zip((:β_coefficients, :p_coefficients), 
                                             (smoothness_tuple, reconstruction_tuple))
    @eval begin
        # Third order WENO -> no lower order possible
        @inline $coefficients(order, ::Val{2}, ::Val{0}) = $(coefficient_tuple(Val(2), Val(0)))
        @inline $coefficients(order, ::Val{2}, ::Val{1}) = $(coefficient_tuple(Val(2), Val(1)))
        
        # Fifth order WENO -> can limit to third order WENO
        @inline function $coefficients(order, ::Val{3}, ::Val{0}) 
            βₙ   =  $(coefficient_tuple(Val(3), Val(0)))
            βₙ₋₁ = $((coefficient_tuple(Val(2), Val(0))..., 0))
            return ifelse(order == 3, βₙ, βₙ₋₁)
        end

        @inline function $coefficients(order, ::Val{3}, ::Val{1}) 
            βₙ   =  $(coefficient_tuple(Val(3), Val(1)))
            βₙ₋₁ = $((coefficient_tuple(Val(2), Val(1))..., 0))
            return ifelse(order == 3, βₙ, βₙ₋₁)
        end

        @inline function $coefficients(order, ::Val{3}, ::Val{2})
            βₙ   = $(coefficient_tuple(Val(3), Val(2)))
            βₙ₋₁ = $(0 .* coefficient_tuple(Val(3), Val(2)))
            return ifelse(order == 3, βₙ, βₙ₋₁)
        end

        # Seventh order WENO -> can limit to both fifth and third order WENO
        @inline function $coefficients(order, ::Val{4}, ::Val{0}) 
            βₙ   =  $(coefficient_tuple(Val(4), Val(0)))
            βₙ₋₁ = $((coefficient_tuple(Val(3), Val(0))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(2), Val(0))..., 0, 0))
            return ifelse(order == 4, βₙ, ifelse(order == 3, βₙ₋₁, βₙ₋₂))
        end

        @inline function $coefficients(order, ::Val{4}, ::Val{1}) 
            βₙ   =  $(coefficient_tuple(Val(4), Val(1)))
            βₙ₋₁ = $((coefficient_tuple(Val(3), Val(1))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(2), Val(1))..., 0, 0))
            return ifelse(order == 4, βₙ, ifelse(order == 3, βₙ₋₁, βₙ₋₂))
        end

        @inline function $coefficients(order, ::Val{4}, ::Val{2})
            βₙ   =  $(coefficient_tuple(Val(4), Val(2)))
            βₙ₋₁ = $((coefficient_tuple(Val(3), Val(2))..., 0))
            βₙ₋₂ = $(0 .* coefficient_tuple(Val(4), Val(2)))
            return ifelse(order == 4, βₙ, ifelse(order == 3, βₙ₋₁, βₙ₋₂))
        end

        @inline function $coefficients(order, ::Val{4}, ::Val{3})
            βₙ   = $(coefficient_tuple(Val(4), Val(3)))
            βₙ₋₁ = $(0 .* coefficient_tuple(Val(4), Val(3)))
            return ifelse(order == 4, βₙ, βₙ₋₁)
        end

        # Ninth order WENO -> can limit to seventh, fifth and third order WENO
        @inline function $coefficients(order, ::Val{5}, ::Val{0}) 
            βₙ   =  $(coefficient_tuple(Val(5), Val(0)))
            βₙ₋₁ = $((coefficient_tuple(Val(4), Val(0))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(3), Val(0))..., 0, 0))
            βₙ₋₃ = $((coefficient_tuple(Val(2), Val(0))..., 0, 0, 0))
            return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, ifelse(order == 3, βₙ₋₂, βₙ₋₃)))
        end

        @inline function $coefficients(order, ::Val{5}, ::Val{1}) 
            βₙ   =  $(coefficient_tuple(Val(5), Val(1)))
            βₙ₋₁ = $((coefficient_tuple(Val(4), Val(1))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(3), Val(1))..., 0, 0))
            βₙ₋₃ = $((coefficient_tuple(Val(2), Val(0))..., 0, 0, 0))
            return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, ifelse(order == 3, βₙ₋₂, βₙ₋₃)))
        end

        @inline function $coefficients(order, ::Val{5}, ::Val{2})
            βₙ   =  $(coefficient_tuple(Val(5), Val(2)))
            βₙ₋₁ = $((coefficient_tuple(Val(4), Val(2))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(3), Val(2))..., 0, 0))  
            βₙ₋₃ = $(0 .* coefficient_tuple(Val(5),  Val(2)))
            return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, ifelse(order == 3, βₙ₋₂, βₙ₋₃)))
        end

        @inline function $coefficients(order, ::Val{5}, ::Val{3})
            βₙ   =  $(coefficient_tuple(Val(5), Val(3)))
            βₙ₋₁ = $((coefficient_tuple(Val(4), Val(3))..., 0))
            βₙ₋₂ = $(0 .* coefficient_tuple(Val(5),  Val(3)))
            return ifelse(order == 5, βₙ, ifelse(order == 4, βₙ₋₁, βₙ₋₂))
        end

        @inline function $coefficients(order, ::Val{5}, ::Val{4})
            βₙ   = $(coefficient_tuple(Val(5),  Val(4)))
            βₙ₋₁ = $(0 .* coefficient_tuple(Val(5),  Val(4)))
            return ifelse(order == 5, βₙ, βₙ₋₁)
        end

        # Eleventh order WENO -> can limit to ninth, seventh, fifth and third order WENO
        @inline function $coefficients(order, ::Val{6}, ::Val{0}) 
            βₙ   =  $(coefficient_tuple(Val(6), Val(0)))
            βₙ₋₁ = $((coefficient_tuple(Val(5), Val(0))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(4), Val(0))..., 0, 0))
            βₙ₋₃ = $((coefficient_tuple(Val(3), Val(0))..., 0, 0, 0))
            βₙ₋₄ = $((coefficient_tuple(Val(2), Val(0))..., 0, 0, 0, 0))
            return ifelse(order == 6, βₙ, ifelse(order == 5, βₙ₋₁, ifelse(order == 4, βₙ₋₂, ifelse(order == 3, βₙ₋₃, βₙ₋₄))))
        end

        @inline function $coefficients(order, ::Val{6}, ::Val{1}) 
            βₙ   =  $(coefficient_tuple(Val(6), Val(1)))
            βₙ₋₁ = $((coefficient_tuple(Val(5), Val(1))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(4), Val(1))..., 0, 0))
            βₙ₋₃ = $((coefficient_tuple(Val(3), Val(1))..., 0, 0, 0))
            βₙ₋₄ = $((coefficient_tuple(Val(2), Val(1))..., 0, 0, 0, 0))
            return ifelse(order == 6, βₙ, ifelse(order == 5, βₙ₋₁, ifelse(order == 4, βₙ₋₂, ifelse(order == 3, βₙ₋₃, βₙ₋₄))))
        end

        @inline function $coefficients(order, ::Val{6}, ::Val{2})
            βₙ   =  $(coefficient_tuple(Val(6), Val(2)))
            βₙ₋₁ = $((coefficient_tuple(Val(5), Val(2))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(4), Val(2))..., 0, 0))
            βₙ₋₃ = $((coefficient_tuple(Val(3), Val(2))..., 0, 0, 0))
            βₙ₋₄ = $(0 .* coefficient_tuple(Val(6), Val(2)))
            return ifelse(order == 6, βₙ, ifelse(order == 5, βₙ₋₁, ifelse(order == 4, βₙ₋₂, ifelse(order == 3, βₙ₋₃, βₙ₋₄))))
        end

        @inline function $coefficients(order, ::Val{6}, ::Val{3})
            βₙ   =  $(coefficient_tuple(Val(6), Val(3)))
            βₙ₋₁ = $((coefficient_tuple(Val(5), Val(3))..., 0))
            βₙ₋₂ = $((coefficient_tuple(Val(4), Val(3))..., 0, 0))
            βₙ₋₃ = $(0 .* coefficient_tuple(Val(6), Val(3)))
            return ifelse(order == 6, βₙ, ifelse(order == 5, βₙ₋₁, ifelse(order == 4, βₙ₋₂, βₙ₋₃)))
        end

        @inline function $coefficients(order, ::Val{6}, ::Val{4})
            βₙ   =  $(coefficient_tuple(Val(6), Val(4)))
            βₙ₋₁ = $((coefficient_tuple(Val(5), Val(4))..., 0))
            βₙ₋₂ = $(0 .* coefficient_tuple(Val(6), Val(4)))
            return ifelse(order == 6, βₙ, ifelse(order == 5, βₙ₋₁, βₙ₋₂))
        end

        @inline function $coefficients(order, ::Val{6}, ::Val{5})
            βₙ   =  $(coefficient_tuple(Val(6), Val(5)))
            βₙ₋₁ = $(0 .* coefficient_tuple(Val(6), Val(4)))
            βₙ₋₂ = $(0 .* coefficient_tuple(Val(6), Val(4)))
            return ifelse(order == 6, βₙ, βₙ₋₁)
        end
    end
end

for buffer in [2, 3, 4, 5, 6]
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