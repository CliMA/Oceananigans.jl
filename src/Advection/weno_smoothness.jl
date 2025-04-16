
# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)

for FT in fully_supported_float_types
    @eval begin
        """
            smoothness_coefficients(::Val{FT}, ::Val{buffer}, ::Val{stencil})

        Return the coefficients used to calculate the smoothness indicators for the stencil 
        number `stencil` of a WENO reconstruction of order `buffer * 2 - 1`. The coefficients
        are ordered in such a way to calculate the smoothness in the following fashion:
        
        ```julia
        buffer  = 4
        stencil = 0
        
        ψ = # The stencil corresponding to S₀ with buffer 4 (7th order WENO)
        
        C = smoothness_coefficients(Val(buffer), Val(0))
        
        # The smoothness indicator
        β = ψ[1] * (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3] + C[4] * ψ[4]) + 
            ψ[2] * (C[5]  * ψ[2] + C[6] * ψ[3] + C[7] * ψ[4]) + 
            ψ[3] * (C[8]  * ψ[3] + C[9] * ψ[4])
            ψ[4] * (C[10] * ψ[4])
        ```
        
        This last operation is metaprogrammed in the function `metaprogrammed_smoothness_operation`
        """
        @inline smoothness_coefficients(::Val{$FT}, ::Val{1}, Nrest, ::Val{0}) = $(FT.((1, )))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{2}, Nrest, ::Val{0}) = $(FT.((1, -2, 1)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{2}, Nrest, ::Val{1}) = $(FT.((1, -2, 1)))

        # 5th order WENO, restricted to orders 3 and 1
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{1}, ::Val{0}) = $(FT.((1, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{1}, ::Val{1}) = $(FT.((0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{1}, ::Val{2}) = $(FT.((0, 0, 0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{2}, ::Val{0}) = $(FT.((1, -2, 0, 1, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{2}, ::Val{1}) = $(FT.((1, -2, 0, 1, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{2}, ::Val{2}) = $(FT.((0,  0, 0, 0, 0, 0)))
        
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{3}, ::Val{0}) = $(FT.((10, -31, 11, 25, -19,  4)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{3}, ::Val{1}) = $(FT.((4,  -13, 5,  13, -13,  4)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{3}, ::Val{2}) = $(FT.((4,  -19, 11, 25, -31, 10)))

        # 7th order WENO, restricted to orders 5, 3, and 1
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{1}, ::Val{0}) = $(FT.((1, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{1}, ::Val{1}) = $(FT.((0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{1}, ::Val{2}) = $(FT.((0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{1}, ::Val{3}) = $(FT.((0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{2}, ::Val{0}) = $(FT.((1, -2, 0, 0, 1, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{2}, ::Val{1}) = $(FT.((1, -2, 0, 0, 1, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{2}, ::Val{2}) = $(FT.((0,  0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{2}, ::Val{3}) = $(FT.((0,  0, 0, 0, 0, 0, 0, 0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{3}, ::Val{0}) = $(FT.((10, -31, 11, 0, 25, -19, 0,  4, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{3}, ::Val{1}) = $(FT.((4,  -13, 5,  0, 13, -13, 0,  4, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{3}, ::Val{2}) = $(FT.((4,  -19, 11, 0, 25, -31, 0, 10, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{3}, ::Val{3}) = $(FT.((0,    0,  0, 0,  0,   0, 0,  0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{4}, ::Val{0}) = $(FT.((2.107,  -9.402, 7.042, -1.854, 11.003,  -17.246,  4.642,  7.043,  -3.882, 0.547)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{4}, ::Val{1}) = $(FT.((0.547,  -2.522, 1.922, -0.494,  3.443,  - 5.966,  1.602,  2.843,  -1.642, 0.267)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{4}, ::Val{2}) = $(FT.((0.267,  -1.642, 1.602, -0.494,  2.843,  - 5.966,  1.922,  3.443,  -2.522, 0.547)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{4}, ::Val{3}) = $(FT.((0.547,  -3.882, 4.642, -1.854,  7.043,  -17.246,  7.042, 11.003,  -9.402, 2.107)))

        # 9th order WENO, restricted to orders 7, 5, 3, and 1
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{1}, ::Val{0}) = $(FT.((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{1}, ::Val{1}) = $(FT.((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{1}, ::Val{2}) = $(FT.((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{1}, ::Val{3}) = $(FT.((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{1}, ::Val{4}) = $(FT.((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{2}, ::Val{0}) = $(FT.((1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{2}, ::Val{1}) = $(FT.((1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{2}, ::Val{2}) = $(FT.((0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{2}, ::Val{3}) = $(FT.((0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{2}, ::Val{4}) = $(FT.((0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{3}, ::Val{0}) = $(FT.((10, -31, 11, 0, 0, 25, -19, 0, 0,  4, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{3}, ::Val{1}) = $(FT.((4,  -13, 5,  0, 0, 13, -13, 0, 0,  4, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{3}, ::Val{2}) = $(FT.((4,  -19, 11, 0, 0, 25, -31, 0, 0, 10, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{3}, ::Val{3}) = $(FT.((0,    0,  0, 0, 0,  0,   0, 0, 0,  0, 0, 0, 0, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{3}, ::Val{4}) = $(FT.((0,    0,  0, 0, 0,  0,   0, 0, 0,  0, 0, 0, 0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{4}, ::Val{0}) = $(FT.((2.107,  -9.402, 7.042, -1.854, 0, 11.003,  -17.246,  4.642, 0,  7.043,  -3.882, 0, 0.547, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{4}, ::Val{1}) = $(FT.((0.547,  -2.522, 1.922, -0.494, 0,  3.443,  - 5.966,  1.602, 0,  2.843,  -1.642, 0, 0.267, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{4}, ::Val{2}) = $(FT.((0.267,  -1.642, 1.602, -0.494, 0,  2.843,  - 5.966,  1.922, 0,  3.443,  -2.522, 0, 0.547, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{4}, ::Val{3}) = $(FT.((0.547,  -3.882, 4.642, -1.854, 0,  7.043,  -17.246,  7.042, 0, 11.003,  -9.402, 0, 2.107, 0, 0)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{4}, ::Val{4}) = $(FT.((0,           0,     0,      0, 0,      0,        0,      0, 0,      0,       0, 0,     0, 0, 0)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{5}, ::Val{0}) = $(FT.((1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{5}, ::Val{1}) = $(FT.((0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{5}, ::Val{2}) = $(FT.((0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{5}, ::Val{3}) = $(FT.((0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{5}, ::Val{4}) = $(FT.((0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)))
    end
end

@inline α_coefficients(α::NTuple{1, T}, ::Val{1}) where T = @inbounds α
@inline α_coefficients(α::NTuple{1, T}, ::Val{2}) where T = @inbounds (α[1], z(T))
@inline α_coefficients(α::NTuple{1, T}, ::Val{3}) where T = @inbounds (α[1], z(T), z(T))
@inline α_coefficients(α::NTuple{1, T}, ::Val{4}) where T = @inbounds (α[1], z(T), z(T), z(T))
@inline α_coefficients(α::NTuple{1, T}, ::Val{5}) where T = @inbounds (α[1], z(T), z(T), z(T), z(T))

@inline α_coefficients(α::NTuple{2, T}, ::Val{1}) where T = @inbounds (α[1], )
@inline α_coefficients(α::NTuple{2, T}, ::Val{2}) where T = @inbounds  α
@inline α_coefficients(α::NTuple{2, T}, ::Val{3}) where T = @inbounds (α[1], α[2], z(T))
@inline α_coefficients(α::NTuple{2, T}, ::Val{4}) where T = @inbounds (α[1], α[2], z(T), z(T))
@inline α_coefficients(α::NTuple{2, T}, ::Val{5}) where T = @inbounds (α[1], α[2], z(T), z(T), z(T))

@inline α_coefficients(α::NTuple{3, T}, ::Val{1}) where T = @inbounds (α[1], )
@inline α_coefficients(α::NTuple{3, T}, ::Val{2}) where T = @inbounds (α[1], α[2])
@inline α_coefficients(α::NTuple{3, T}, ::Val{3}) where T = @inbounds  α
@inline α_coefficients(α::NTuple{3, T}, ::Val{4}) where T = @inbounds (α[1], α[2], α[3], z(T))
@inline α_coefficients(α::NTuple{3, T}, ::Val{5}) where T = @inbounds (α[1], α[2], α[3], z(T), z(T))

@inline α_coefficients(α::NTuple{4, T}, ::Val{1}) where T = @inbounds (α[1], )
@inline α_coefficients(α::NTuple{4, T}, ::Val{2}) where T = @inbounds (α[1], α[2])
@inline α_coefficients(α::NTuple{4, T}, ::Val{3}) where T = @inbounds (α[1], α[2], α[3])
@inline α_coefficients(α::NTuple{4, T}, ::Val{4}) where T = @inbounds  α
@inline α_coefficients(α::NTuple{4, T}, ::Val{5}) where T = @inbounds (α[1], α[2], α[3], α[4], z(T))

@inline α_coefficients(α::NTuple{5, T}, ::Val{1}) where T = @inbounds (α[1], )
@inline α_coefficients(α::NTuple{5, T}, ::Val{2}) where T = @inbounds (α[1], α[2])
@inline α_coefficients(α::NTuple{5, T}, ::Val{3}) where T = @inbounds (α[1], α[2], α[3])
@inline α_coefficients(α::NTuple{5, T}, ::Val{4}) where T = @inbounds (α[1], α[2], α[3], α[4])
@inline α_coefficients(α::NTuple{5, T}, ::Val{5}) where T = @inbounds  α
