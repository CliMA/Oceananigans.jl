# _UNIFORM_ smoothness coefficients for WENO reconstructions
#
# The naming convention for the coefficients is as follows:
# SSXYZ
# SS -> smoothness stencil
# X  -> order of the WENO reconstruction
# Y  -> reduced order of the WENO reconstruction (with Y ≤ X)
# Z  -> stencil number  
# 
# The zero stencil for each order is names SSX0M

# Zero and one coefficient stencils
const SS20M = (0, 0, 0)
const SS210 = (1, 0, 0)

const SS30M = (0, 0, 0, 0, 0, 0)
const SS310 = (1, 0, 0, 0, 0, 0)

const SS40M = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS410 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

const SS50M = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS510 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

const SS60M = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS610 = (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Coefficients for third order WENO
const SS220 = (1, -2, 1)
const SS221 = (1, -2, 1)

const SS320 = (1, -2, 0, 1, 0, 0)
const SS321 = (1, -2, 0, 1, 0, 0)

const SS420 = (1, -2, 0, 0, 1, 0, 0, 0, 0, 0)
const SS421 = (1, -2, 0, 0, 1, 0, 0, 0, 0, 0)

const SS520 = (1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS521 = (1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

const SS620 = (1, -2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS621 = (1, -2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Coefficients for fifth order WENO
const SS330 = (10, -31, 11, 25, -19, 4)
const SS331 = (4,  -13,  5, 13, -13, 4)
const SS332 = (4,  -19, 11, 25, -31, 10)

const SS430 = (10, -31, 11, 0, 25, -19, 0,  4, 0, 0)
const SS431 = (4,  -13,  5, 0, 13, -13, 0,  4, 0, 0)
const SS432 = (4,  -19, 11, 0, 25, -31, 0, 10, 0, 0)

const SS530 = (10, -31, 11, 0, 0, 25, -19, 0, 0,  4, 0, 0, 0, 0, 0)
const SS531 = (4,  -13,  5, 0, 0, 13, -13, 0, 0,  4, 0, 0, 0, 0, 0)
const SS532 = (4,  -19, 11, 0, 0, 25, -31, 0, 0, 10, 0, 0, 0, 0, 0)

const SS630 = (10, -31, 11, 0, 0, 0, 25, -19, 0, 0, 0,  4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS631 = (4,  -13,  5, 0, 0, 0, 13, -13, 0, 0, 0,  4, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS632 = (4,  -19, 11, 0, 0, 0, 25, -31, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# Coefficients for seventh order WENO
const SS440 = (2107, -9402, 7042, -1854, 11003, -17246, 4642,  7043, -3882, 0547)
const SS441 = (0547, -2522, 1922, -0494,  3443,  -5966, 1602,  2843, -1642, 0267)
const SS442 = (0267, -1642, 1602, -0494,  2843,  -5966, 1922,  3443, -2522, 0547)
const SS443 = (0547, -3882, 4642, -1854,  7043, -17246, 7042, 11003, -9402, 2107)

const SS540 = (2107, -9402, 7042, -1854, 0, 11003, -17246, 4642, 0,  7043, -3882, 0, 0547, 0, 0)
const SS541 = (0547, -2522, 1922, -0494, 0,  3443,  -5966, 1602, 0,  2843, -1642, 0, 0267, 0, 0)
const SS542 = (0267, -1642, 1602, -0494, 0,  2843,  -5966, 1922, 0,  3443, -2522, 0, 0547, 0, 0)
const SS543 = (0547, -3882, 4642, -1854, 0,  7043, -17246, 7042, 0, 11003, -9402, 0, 2107, 0, 0)

const SS540 = (2107, -9402, 7042, -1854, 0, 0, 11003, -17246, 4642, 0, 0,  7043, -3882, 0, 0, 0547, 0, 0, 0, 0, 0)
const SS541 = (0547, -2522, 1922, -0494, 0, 0,  3443,  -5966, 1602, 0, 0,  2843, -1642, 0, 0, 0267, 0, 0, 0, 0, 0)
const SS542 = (0267, -1642, 1602, -0494, 0, 0,  2843,  -5966, 1922, 0, 0,  3443, -2522, 0, 0, 0547, 0, 0, 0, 0, 0)
const SS543 = (0547, -3882, 4642, -1854, 0, 0,  7043, -17246, 7042, 0, 0, 11003, -9402, 0, 0, 2107, 0, 0, 0, 0, 0)

# Coefficients for ninth order WENO
const SS550 = (107918,  -649501, 758823, -411487,  086329,  1020563, -2462076, 1358458, -288007, 1521393, -1704396, 364863,  482963, -208501, 022658)
const SS551 = (022658,  -140251, 165153, -088297,  018079,   242723,  -611976,  337018, -070237,  406293,  -464976, 099213,  138563, -060871, 006908)
const SS552 = (006908,  -051001, 067923, -038947,  008209,   104963,  -299076,  179098, -038947,  231153,  -299076, 067923,  104963, -051001, 006908)
const SS553 = (006908,  -060871, 099213, -070237,  018079,   138563,  -464976,  337018, -088297,  406293,  -611976, 165153,  242723, -140251, 022658)
const SS554 = (022658,  -208501, 364863, -288007,  086329,   482963, -1704396, 1358458, -411487, 1521393, -2462076, 758823, 1020563, -649501, 107918)

const SS650 = (107918,  -649501, 758823, -411487,  086329,  0, 1020563, -2462076, 1358458, -288007, 0, 1521393, -1704396, 364863, 0,  482963, -208501, 0, 022658, 0, 0)
const SS651 = (022658,  -140251, 165153, -088297,  018079,  0,  242723,  -611976,  337018, -070237, 0,  406293,  -464976, 099213, 0,  138563, -060871, 0, 006908, 0, 0)
const SS652 = (006908,  -051001, 067923, -038947,  008209,  0,  104963,  -299076,  179098, -038947, 0,  231153,  -299076, 067923, 0,  104963, -051001, 0, 006908, 0, 0)
const SS653 = (006908,  -060871, 099213, -070237,  018079,  0,  138563,  -464976,  337018, -088297, 0,  406293,  -611976, 165153, 0,  242723, -140251, 0, 022658, 0, 0)
const SS654 = (022658,  -208501, 364863, -288007,  086329,  0,  482963, -1704396, 1358458, -411487, 0, 1521393, -2462076, 758823, 0, 1020563, -649501, 0, 107918, 0, 0)

# Coefficients for eleventh order WENO
const SS660 = (06150211, -47460464, 76206736, -63394124, 27060170, -04712740,  94851237, -311771244, 262901672, -113206788,  19834350, 260445372, -444003904, 192596472, -33918804, 190757572, -166461044, 29442256, 36480687, -12950184, 01152561)
const SS661 = (01152561, -09117992, 14742480, -12183636, 05134574, -00880548,  19365967,  -65224244,  55053752,  -23510468,  04067018,  56662212,  -97838784,  42405032, -07408908,  43093692,  -37913324, 06694608, 08449957, -03015728, 00271779)
const SS662 = (00271779, -02380800, 04086352, -03462252, 01458762, -00245620,  05653317,  -20427884,  17905032,  -07727988,  01325006,  19510972,  -35817664,  15929912, -02792660,  17195652,  -15880404, 02863984, 03824847, -01429976, 00139633)
const SS663 = (00139633, -01429976, 02863984, -02792660, 01325006, -00245620,  03824847,  -15880404,  15929912,  -07727988,  01458762,  17195652,  -35817664,  17905032, -03462252,  19510972,  -20427884, 04086352, 05653317, -02380800, 00271779)
const SS664 = (00271779, -03015728, 06694608, -07408908, 04067018, -00880548,  08449957,  -37913324,  42405032,  -23510468,  05134574,  43093692,  -97838784,  55053752, -12183636,  56662212,  -65224244, 14742480, 19365967, -09117992, 01152561)
const SS665 = (01152561, -12950184, 29442256, -33918804, 19834350, -04712740,  36480687, -166461044, 192596472, -113206788,  27060170, 190757572, -444003904, 262901672, -63394124, 260445372, -311771244, 76206736, 94851237, -47460464, 06150211)

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
        @inline smoothness_coefficients(::WENO{2}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(SS210),    # Order 1
                                       $(SS220))    # Order 3

        @inline smoothness_coefficients(::WENO{2}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(SS20M),    # Order 1
                                       $(SS221))    # Order 3

        # 5th order WENO, restricted to orders 3 and 1
        @inline smoothness_coefficients(::WENO{3}, red_order, ::Val{0}) =
                ifelse(red_order == 1, $(SS310),     # Order 1
                ifelse(red_order == 2, $(SS320),     # Order 3                                           
                                       $(SS330)))    # Order 5

        @inline smoothness_coefficients(::WENO{3}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(SS30M),     # Order 1
                ifelse(red_order == 2, $(SS321),     # Order 3
                                       $(SS331)))    # Order 5

        @inline smoothness_coefficients(::WENO{3}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(SS30M),     # Order ≤ 3
                                       $(SS332))     # Order 5

        # 7th order WENO, restricted to orders 5, 3, and 1
        @inline smoothness_coefficients(::WENO{4}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(SS410),     # Order 1                              
                ifelse(red_order == 2, $(SS420),     # Order 3
                ifelse(red_order == 3, $(SS430),     # Order 5
                                       $(SS440))))   # Order 7
        
        @inline smoothness_coefficients(::WENO{4}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(SS40M),     # Order 1
                ifelse(red_order == 2, $(SS421),     # Order 3
                ifelse(red_order == 3, $(SS431),     # Order 5
                                       $(SS441))))   # Order 7

        @inline smoothness_coefficients(::WENO{4}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(SS40M),     # Order ≤ 3                    
                ifelse(red_order == 3, $(SS432),     # Order 5
                                       $(SS442)))    # Order 7
        
        @inline smoothness_coefficients(::WENO{4}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(SS40M),     # Order ≤ 5                                              
                                       $(SS443))     # Order 7

        # 9th order WENO, restricted to orders 7, 5, 3, and 1
        @inline smoothness_coefficients(::WENO{5}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(SS510),     # Order 1                             
                ifelse(red_order == 2, $(SS520),     # Order 3       
                ifelse(red_order == 3, $(SS530),     # Order 5
                ifelse(red_order == 4, $(SS540),     # Order 7
                                       $(SS550)))))  # Order 9

        @inline smoothness_coefficients(::WENO{5}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(SS50M),     # Order 1                             
                ifelse(red_order == 2, $(SS521),     # Order 3       
                ifelse(red_order == 3, $(SS531),     # Order 5
                ifelse(red_order == 4, $(SS541),     # Order 7
                                       $(SS551)))))  # Order 9

        @inline smoothness_coefficients(::WENO{5}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(SS50M),     # Order ≤ 3                            
                ifelse(red_order == 3, $(SS532),     # Order 5
                ifelse(red_order == 4, $(SS542),     # Order 7
                                       $(SS552))))   # Order 9

        @inline smoothness_coefficients(::WENO{5}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(SS50M),     # Order ≤ 5 
                ifelse(red_order == 4, $(SS543),     # Order 7
                                       $(SS553)))    # Order 9

        @inline smoothness_coefficients(::WENO{5}, red_order, ::Val{4}) = 
                ifelse(red_order <  5, $(SS50M),     # Order ≤ 7
                                       $(SS554))     # Order 9

        # 11th order WENO, restricted to orders 9, 7, 5, 3, and 1
        @inline smoothness_coefficients(::WENO{6}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(SS610),     # Order 1                             
                ifelse(red_order == 2, $(SS620),     # Order 3       
                ifelse(red_order == 3, $(SS630),     # Order 5
                ifelse(red_order == 4, $(SS640),     # Order 7
                ifelse(red_order == 5, $(SS650),     # Order 9
                                       $(SS660)))))) # Order 11

        @inline smoothness_coefficients(::WENO{6}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(SS60M),     # Order 1                             
                ifelse(red_order == 2, $(SS621),     # Order 3       
                ifelse(red_order == 3, $(SS631),     # Order 5
                ifelse(red_order == 4, $(SS641),     # Order 7
                ifelse(red_order == 5, $(SS651),     # Order 9
                                       $(SS661)))))) # Order 11

        @inline smoothness_coefficients(::WENO{6}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(SS60M),     # Order ≤ 3                            
                ifelse(red_order == 3, $(SS632),     # Order 5
                ifelse(red_order == 4, $(SS642),     # Order 7
                ifelse(red_order == 5, $(SS652),     # Order 9
                                       $(SS662)))))  # Order 11

        @inline smoothness_coefficients(::WENO{6}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(SS60M),     # Order ≤ 5 
                ifelse(red_order == 4, $(SS643),     # Order 7
                ifelse(red_order == 5, $(SS653),     # Order 9
                                       $(SS663))))   # Order 11

        @inline smoothness_coefficients(::WENO{6}, red_order, ::Val{4}) = 
                ifelse(red_order <  5, $(SS60M),     # Order ≤ 7
                ifelse(red_order == 5, $(SS654),     # Order 9
                                       $(SS664)))    # Order 11

        @inline smoothness_coefficients(::WENO{6}, red_order, ::Val{5}) = 
                ifelse(red_order <  6, $(SS60M),     # Order ≤ 9
                                       $(SS665))     # Order 11
end

# Global smoothness indicator τ₂ᵣ₋₁ from "Accuracy of the weighted essentially non-oscillatory 
# conservative finite difference schemes", Don & Borges, 2013
@inline function global_smoothness_indicator(β::NTuple{1}, R) 
    @inbounds abs(β[1])
end

@inline function global_smoothness_indicator(β::NTuple{2}, R) 
    τ = @inbounds @fastmath ifelse(R == 1, β[1], β[1] - β[2])
    return abs(τ)
end

@inline function global_smoothness_indicator(β::NTuple{3}, R) 
    τ = @inbounds @fastmath ifelse(R == 1, β[1],
                            ifelse(R == 2, β[1] - β[2],
                                           β[1] - β[3]))
    return abs(τ)
end

@inline function global_smoothness_indicator(β::NTuple{4}, R) 
    τ = @inbounds @fastmath ifelse(R == 1, β[1],
                            ifelse(R == 2, β[1] -  β[2],
                            ifelse(R == 3, β[1] -  β[3],
                                           β[1] + 3β[2] - 3β[3] - β[4])))
    return abs(τ)
end

@inline function global_smoothness_indicator(β::NTuple{5}, R) 
    τ = @inbounds @fastmath ifelse(R == 1, β[1],
                            ifelse(R == 2, β[1] - β[2],
                            ifelse(R == 3, β[1] - β[3],
                            ifelse(R == 4, β[1] + 3β[2] - 3β[3] -  β[4],
                                           β[1] + 2β[2] - 6β[3] + 2β[4] + β[5]))))
    return abs(τ)
end

# Otherwise we take the 11th order WENO smoothness indicator as a default
@inline function global_smoothness_indicator(β, R) 
    τ = @inbounds @fastmath ifelse(R == 1, β[1],
                            ifelse(R == 2, β[1] -   β[2],
                            ifelse(R == 3, β[1] -   β[3],
                            ifelse(R == 4, β[1] +  3β[2] -   3β[3] -    β[4],
                            ifelse(R == 5, β[1] +  2β[2] -   6β[3] +   2β[4] +   β[5],
                                           β[1] + 36β[2] + 135β[3] - 135β[4] - 36β[5] - β[6])))))
    return abs(τ)
end
