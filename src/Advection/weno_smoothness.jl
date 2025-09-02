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

# Coefficients for third order WENO
const SS220 = (1, -2, 1)
const SS221 = (1, -2, 1)

const SS320 = (1, -2, 0, 1, 0, 0)
const SS321 = (1, -2, 0, 1, 0, 0)

const SS420 = (1, -2, 0, 0, 1, 0, 0, 0, 0, 0)
const SS421 = (1, -2, 0, 0, 1, 0, 0, 0, 0, 0)

const SS520 = (1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
const SS521 = (1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)

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

# Coefficients for seventh order WENO
const SS440 = (2.107, -9.402, 7.042, -1.854, 11.003, -17.246, 4.642,  7.043, -3.882, 0.547)
const SS441 = (0.547, -2.522, 1.922, -0.494,  3.443, -5.966,  1.602,  2.843, -1.642, 0.267)
const SS442 = (0.267, -1.642, 1.602, -0.494,  2.843, -5.966,  1.922,  3.443, -2.522, 0.547)
const SS443 = (0.547, -3.882, 4.642, -1.854,  7.043, -17.246, 7.042, 11.003, -9.402, 2.107)

const SS540 = (2.107, -9.402, 7.042, -1.854, 0, 11.003, -17.246, 4.642, 0,  7.043, -3.882, 0, 0.547, 0, 0)
const SS541 = (0.547, -2.522, 1.922, -0.494, 0,  3.443, -5.966,  1.602, 0,  2.843, -1.642, 0, 0.267, 0, 0)
const SS542 = (0.267, -1.642, 1.602, -0.494, 0,  2.843, -5.966,  1.922, 0,  3.443, -2.522, 0, 0.547, 0, 0)
const SS543 = (0.547, -3.882, 4.642, -1.854, 0,  7.043, -17.246, 7.042, 0, 11.003, -9.402, 0, 2.107, 0, 0)

# Coefficients for ninth order WENO
const SS550 = (1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)
const SS551 = (0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)
const SS552 = (0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)
const SS553 = (0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)
const SS554 = (0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)

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
        @inline smoothness_coefficients(::WENO{2, <:Any, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(SS210)),   # Order 1
                                       $(FT.(SS220)))   # Order 3

        @inline smoothness_coefficients(::WENO{2, <:Any, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(SS20M)),   # Order 1
                                       $(FT.(SS221)))   # Order 3

        # 5th order WENO, restricted to orders 3 and 1
        @inline smoothness_coefficients(::WENO{3, <:Any, $FT}, red_order, ::Val{0}) =
                ifelse(red_order == 1, $(FT.(SS310)),    # Order 1
                ifelse(red_order == 2, $(FT.(SS320)),    # Order 3                                           
                                       $(FT.(SS330))))   # Order 5

        @inline smoothness_coefficients(::WENO{3, <:Any, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(SS30M)),    # Order 1
                ifelse(red_order == 2, $(FT.(SS321)),    # Order 3
                                       $(FT.(SS331))))   # Order 5

        @inline smoothness_coefficients(::WENO{3, <:Any, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(SS30M)),    # Order ≤ 3
                                       $(FT.(SS332)))    # Order 5

        # 7th order WENO, restricted to orders 5, 3, and 1
        @inline smoothness_coefficients(::WENO{4, <:Any, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(SS410)),    # Order 1                              
                ifelse(red_order == 2, $(FT.(SS420)),    # Order 3
                ifelse(red_order == 3, $(FT.(SS430)),    # Order 5
                                       $(FT.(SS440)))))  # Order 7
        
        @inline smoothness_coefficients(::WENO{4, <:Any, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(SS40M)),    # Order 1
                ifelse(red_order == 2, $(FT.(SS421)),    # Order 3
                ifelse(red_order == 3, $(FT.(SS431)),    # Order 5
                                       $(FT.(SS441)))))  # Order 7

        @inline smoothness_coefficients(::WENO{4, <:Any, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(SS40M)),    # Order ≤ 3                    
                ifelse(red_order == 3, $(FT.(SS432)),    # Order 5
                                       $(FT.(SS442))))   # Order 7
        
        @inline smoothness_coefficients(::WENO{4, <:Any, $FT}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(SS40M)),    # Order ≤ 5                                              
                                       $(FT.(SS443)))    # Order 7

        # 9th order WENO, restricted to orders 7, 5, 3, and 1
        @inline smoothness_coefficients(::WENO{5, <:Any, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(SS510)),    # Order 1                             
                ifelse(red_order == 2, $(FT.(SS520)),    # Order 3       
                ifelse(red_order == 3, $(FT.(SS530)),    # Order 5
                ifelse(red_order == 4, $(FT.(SS540)),    # Order 7
                                       $(FT.(SS550)))))) # Order 9

        @inline smoothness_coefficients(::WENO{5, <:Any, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(SS50M)),    # Order 1                             
                ifelse(red_order == 2, $(FT.(SS521)),    # Order 3       
                ifelse(red_order == 3, $(FT.(SS531)),    # Order 5
                ifelse(red_order == 4, $(FT.(SS541)),    # Order 7
                                       $(FT.(SS551)))))) # Order 9

        @inline smoothness_coefficients(::WENO{5, <:Any, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(SS50M)),    # Order ≤ 3                            
                ifelse(red_order == 3, $(FT.(SS532)),    # Order 5
                ifelse(red_order == 4, $(FT.(SS542)),    # Order 7
                                       $(FT.(SS552)))))  # Order 9

        @inline smoothness_coefficients(::WENO{5, <:Any, $FT}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(SS50M)),    # Order ≤ 5 
                ifelse(red_order == 4, $(FT.(SS543)),    # Order 7
                                       $(FT.(SS553))))   # Order 9

        @inline smoothness_coefficients(::WENO{5, <:Any, $FT}, red_order, ::Val{4}) = 
                ifelse(red_order <  5, $(FT.(SS50M)),    # Order ≤ 7
                                       $(FT.(SS554)))    # Order 9
    end
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

# Otherwise we take the 9th order WENO smoothness indicator as a default
@inline function global_smoothness_indicator(β, R) 
    τ = @inbounds @fastmath ifelse(R == 1, β[1],
                            ifelse(R == 2, β[1] - β[2],
                            ifelse(R == 3, β[1] - β[3],
                            ifelse(R == 4, β[1] + 3β[2] - 3β[3] -  β[4],
                                           β[1] + 2β[2] - 6β[3] + 2β[4] + β[5]))))
    return abs(τ)
end
