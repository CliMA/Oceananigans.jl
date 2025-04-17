
# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)

# Zero and one coefficient stencils
const ST20M = (0, 0, 0)
const ST210 = (1, 0, 0)
const ST30M = (0, 0, 0, 0, 0, 0)
const ST310 = (1, 0, 0, 0, 0, 0)
const ST40M = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const ST410 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const ST50M = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const ST510 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Coeffiecients for third order WENO
const ST220 = (1, -2, 1)
const ST221 = (1, -2, 1)

const ST320 = (1, -2, 0, 1, 0, 0)
const ST321 = (1, -2, 0, 1, 0, 0)

const ST420 = (1, -2, 0, 0, 1, 0, 0, 0, 0, 0)
const ST421 = (1, -2, 0, 0, 1, 0, 0, 0, 0, 0)

const ST520 = (1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const ST521 = (1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Cefficients for fifth order WENO
const ST330 = (10, -31, 11, 25, -19, 4)
const ST331 = (4,  -13,  5, 13, -13, 4)
const ST332 = (4,  -19, 11, 25, -31, 10)

const ST430 = (10, -31, 11, 0, 25, -19, 0, 4, 0, 0)
const ST431 = (4,  -13,  5, 0, 13, -13, 0, 4, 0, 0)
const ST432 = (10, -31, 11, 0, 25, -19, 0, 4, 0, 0)

const ST530 = (10, -31, 11, 0, 0, 25, -19, 0, 0, 4, 0, 0, 0, 0, 0)
const ST531 = (4,  -13,  5, 0, 0, 13, -13, 0, 0, 4, 0, 0, 0, 0, 0)
const ST532 = (10, -31, 11, 0, 0, 25, -19, 0, 0, 4, 0, 0, 0, 0, 0)

# Coefficients for seventh order WENO
const ST440 = (2.107, -9.402, 7.042, -1.854, 11.003, -17.246, 4.642,  7.043, -3.882, 0.547)
const ST441 = (0.547, -2.522, 1.922, -0.494,  3.443, -5.966,  1.602,  2.843, -1.642, 0.267)
const ST442 = (0.267, -1.642, 1.602, -0.494,  2.843, -5.966,  1.922,  3.443, -2.522, 0.547)
const ST443 = (0.547, -3.882, 4.642, -1.854,  7.043, -17.246, 7.042, 11.003, -9.402, 2.107)

const ST540 = (2.107, -9.402, 7.042, -1.854, 0, 11.003, -17.246, 4.642, 0,  7.043, -3.882, 0, 0.547, 0, 0)
const ST541 = (0.547, -2.522, 1.922, -0.494, 0,  3.443, -5.966,  1.602, 0,  2.843, -1.642, 0, 0.267, 0, 0)
const ST542 = (0.267, -1.642, 1.602, -0.494, 0,  2.843, -5.966,  1.922, 0,  3.443, -2.522, 0, 0.547, 0, 0)
const ST543 = (0.547, -3.882, 4.642, -1.854, 0,  7.043, -17.246, 7.042, 0, 11.003, -9.402, 0, 2.107, 0, 0)

# Coefficients for ninth order WENO
const ST550 = (1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)
const ST551 = (0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)
const ST552 = (0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)
const ST553 = (0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)
const ST554 = (0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)

for FT in fully_supported_float_types
    @eval begin
        """
            smoothness_coefficients(::Val{FT}, ::Val{buffer}, ::Val{red_order}, ::Val{stencil})

        Return the coefficients used to calculate the smoothness indicators for the stencil 
        number `stencil` of a WENO reconstruction of order `buffer * 2 - 1`. The actual order of 
        reconstruction is restricted by `red_order <= buffer`. The coefficients
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
        
        In the above case, if `red_order == 2`, then all the coefficients corresponding to ψ[i for i > 2]
        are zero. This last operation is metaprogrammed in the function `metaprogrammed_smoothness_operation`
        """
        # 3rd order WENO, restricted to order 1 (does not matter the restriction order here)
        @inline smoothness_coefficients(::Val{$FT}, ::Val{2}, red_order, ::Val{0}) = $(FT.(ST220))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{2}, red_order, ::Val{1}) = $(FT.(ST221))

        # 5th order WENO, restricted to orders 3 and 1
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, red_order, ::Val{0}) = ifelse(red_order == 1, $(FT.(ST310)),
                                                                                     ifelse(red_order == 2, $(FT.(ST320)),
                                                                                                            $(FT.(ST330)))) # Full order (3)

        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, red_order, ::Val{1}) = ifelse(red_order == 1, $(FT.(ST30M)),
                                                                                     ifelse(red_order == 2, $(FT.(ST321)),
                                                                                                            $(FT.(ST331)))) # Full order (3)

        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, red_order, ::Val{2}) = ifelse(red_order <  3, $(FT.(ST30M)),
                                                                                                            $(FT.(ST332))) # Full order (3)

        # 7th order WENO, restricted to orders 5, 3, and 1
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{0}) = ifelse(red_order == 1, $(FT.(ST410)),
                                                                                     ifelse(red_order == 2, $(FT.(ST420)), 
                                                                                     ifelse(red_order == 3, $(FT.(ST430)), 
                                                                                                            $(FT.(ST440))))) # Full order
        
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{1}) = ifelse(red_order == 1, $(FT.(ST40M)),
                                                                                     ifelse(red_order == 2, $(FT.(ST421)),
                                                                                     ifelse(red_order == 3, $(FT.(ST431)),
                                                                                                            $(FT.(ST441))))) # Full order

        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{2}) = ifelse(red_order  < 3, $(FT.(ST40M)),
                                                                                     ifelse(red_order == 3, $(FT.(ST432)),
                                                                                                            $(FT.(ST442)))) # Full order
        
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{3}) = ifelse(red_order <  4, $(FT.(ST40M)),
                                                                                                            $(FT.(ST443))) # Full order

        # 9th order WENO, restricted to orders 7, 5, 3, and 1
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{0}) = ifelse(red_order == 1, $(FT.(ST510)),
                                                                                     ifelse(red_order == 2, $(FT.(ST520)),
                                                                                     ifelse(red_order == 3, $(FT.(ST530)),
                                                                                     ifelse(red_order == 4, $(FT.(ST540)),
                                                                                                            $(FT.(ST550)))))) # Full order

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{1}) = ifelse(red_order == 1, $(FT.(ST50M)),
                                                                                     ifelse(red_order == 2, $(FT.(ST521)),
                                                                                     ifelse(red_order == 3, $(FT.(ST531)),
                                                                                     ifelse(red_order == 4, $(FT.(ST541)),
                                                                                                            $(FT.(ST551)))))) # Full order

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{2}) = ifelse(red_order <  3, $(FT.(ST50M)),
                                                                                     ifelse(red_order == 3, $(FT.(ST532)),
                                                                                     ifelse(red_order == 4, $(FT.(ST542)),
                                                                                                            $(FT.(ST552))))) # Full order
        
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{3}) = ifelse(red_order <  4, $(FT.(ST50M)),
                                                                                     ifelse(red_order == 4, $(FT.(ST543)),
                                                                                                            $(FT.(ST553)))) # Full order

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{4}) = ifelse(red_order <  5, $(FT.(ST50M)),
                                                                                                            $(FT.(ST554))) # Full order
    
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, AnyN, ::Val{0}) = $(FT.((0.6150211, -4.7460464, 7.6206736, -6.3394124, 2.7060170, -0.4712740,  9.4851237, -31.1771244, 26.2901672, -11.3206788,  1.9834350, 26.0445372, -44.4003904, 19.2596472, -3.3918804, 19.0757572, -16.6461044, 2.9442256, 3.6480687, -1.2950184, 0.1152561)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, AnyN, ::Val{1}) = $(FT.((0.1152561, -0.9117992, 1.4742480, -1.2183636, 0.5134574, -0.0880548,  1.9365967,  -6.5224244,  5.5053752,  -2.3510468,  0.4067018,  5.6662212,  -9.7838784,  4.2405032, -0.7408908,  4.3093692,  -3.7913324, 0.6694608, 0.8449957, -0.3015728, 0.0271779)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, AnyN, ::Val{2}) = $(FT.((0.0271779, -0.2380800, 0.4086352, -0.3462252, 0.1458762, -0.0245620,  0.5653317,  -2.0427884,  1.7905032,  -0.7727988,  0.1325006,  1.9510972,  -3.5817664,  1.5929912, -0.2792660,  1.7195652,  -1.5880404, 0.2863984, 0.3824847, -0.1429976, 0.0139633)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, AnyN, ::Val{3}) = $(FT.((0.0139633, -0.1429976, 0.2863984, -0.2792660, 0.1325006, -0.0245620,  0.3824847,  -1.5880404,  1.5929912,  -0.7727988,  0.1458762,  1.7195652,  -3.5817664,  1.7905032, -0.3462252,  1.9510972,  -2.0427884, 0.4086352, 0.5653317, -0.2380800, 0.0271779)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, AnyN, ::Val{4}) = $(FT.((0.0271779, -0.3015728, 0.6694608, -0.7408908, 0.4067018, -0.0880548,  0.8449957,  -3.7913324,  4.2405032,  -2.3510468,  0.5134574,  4.3093692,  -9.7838784,  5.5053752, -1.2183636,  5.6662212,  -6.5224244, 1.4742480, 1.9365967, -0.9117992, 0.1152561)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, AnyN, ::Val{5}) = $(FT.((0.1152561, -1.2950184, 2.9442256, -3.3918804, 1.9834350, -0.4712740,  3.6480687, -16.6461044, 19.2596472, -11.3206788,  2.7060170, 19.0757572, -44.4003904, 26.2901672, -6.3394124, 26.0445372, -31.1771244, 7.6206736, 9.4851237, -4.7460464, 0.6150211)))
    end
end