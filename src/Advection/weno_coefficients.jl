# _UNIFORM_ reconstruction coefficients

# The naming convention for the coefficients is as follows:
# RSXYZ
# RS -> reconstruction stencil
# X  -> order of the WENO reconstruction
# Y  -> reduced order of the WENO reconstruction (with Y ≤ X)
# Z  -> stencil number  
# 
# The zero stencil for each order is names RSX0M

# Zero and one coefficient stencils
const RS20M = (0, 0)
const RS210 = (1, 0)
const RS30M = (0, 0, 0)
const RS310 = (1, 0, 0)
const RS40M = (0, 0, 0, 0)
const RS410 = (1, 0, 0, 0)
const RS50M = (0, 0, 0, 0, 0)
const RS510 = (1, 0, 0, 0, 0)

# Coefficients for third order WENO reconstruction
const RS220 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=2)
const RS221 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=2)
const RS320 = (RS220..., 0)
const RS321 = (RS221..., 0)
const RS420 = (RS320..., 0)
const RS421 = (RS321..., 0)
const RS520 = (RS420..., 0)
const RS521 = (RS421..., 0)

# Coefficients for fifth order WENO
const RS330 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=3)
const RS331 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=3)
const RS332 = stencil_coefficients(BigFloat, 50, 2, collect(1:100), collect(1:100); order=3)
const RS430 = (RS330..., 0)
const RS431 = (RS331..., 0)
const RS432 = (RS332..., 0)
const RS530 = (RS430..., 0)
const RS531 = (RS431..., 0)
const RS532 = (RS432..., 0)

# Coefficients for seventh order WENO
const RS440 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=4)
const RS441 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=4)
const RS442 = stencil_coefficients(BigFloat, 50, 2, collect(1:100), collect(1:100); order=4)
const RS443 = stencil_coefficients(BigFloat, 50, 3, collect(1:100), collect(1:100); order=4)
const RS540 = (RS440..., 0)
const RS541 = (RS441..., 0)
const RS542 = (RS442..., 0)
const RS543 = (RS443..., 0)

# Coefficients for ninth order WENO
const RS550 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=5)
const RS551 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=5)
const RS552 = stencil_coefficients(BigFloat, 50, 2, collect(1:100), collect(1:100); order=5)
const RS553 = stencil_coefficients(BigFloat, 50, 3, collect(1:100), collect(1:100); order=5)
const RS554 = stencil_coefficients(BigFloat, 50, 4, collect(1:100), collect(1:100); order=5)

for FT in fully_supported_float_types
    @eval begin
        """
            reconstruction_coefficients(::Val{FT}, ::Val{buffer}, ::Val{red_order}, ::Val{stencil})

        Return the coefficients used to calculate the ....
        """
        # 3rd order WENO, restricted to order 1 (does not matter the restriction order here)
        @inline reconstruction_coefficients(::WENO{2, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS210)),    # Order 1
                                       $(FT.(RS220)))    # Order 3

        @inline reconstruction_coefficients(::WENO{2, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS20M)),    # Order 1
                                       $(FT.(RS221)))    # Order 3

        # 5th order WENO, restricted to orders 3 and 1
        @inline reconstruction_coefficients(::WENO{3, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS310)),    # Order 1                
                ifelse(red_order == 2, $(FT.(RS320)),    # Order 3                                           
                                       $(FT.(RS330))))   # Order 5

        @inline reconstruction_coefficients(::WENO{3, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS30M)),    # Order 1                
                ifelse(red_order == 2, $(FT.(RS321)),    # Order 3                                           
                                       $(FT.(RS331))))   # Order 5

        @inline reconstruction_coefficients(::WENO{3, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS30M)),    # Order ≤ 3                                          
                                       $(FT.(RS332)))    # Order 5

        # 7th order WENO, restricted to orders 5, 3, and 1
        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS410)),    # Order 1                                                
                ifelse(red_order == 2, $(FT.(RS420)),    # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS430)),    # Order 5  
                                       $(FT.(RS440)))))  # Order 7
        
        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS40M)),    # Order 1                                                
                ifelse(red_order == 2, $(FT.(RS421)),    # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS431)),    # Order 5  
                                       $(FT.(RS441)))))  # Order 7

        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS40M)),    # Order ≤ 3                                             
                ifelse(red_order == 3, $(FT.(RS432)),    # Order 5  
                                       $(FT.(RS442))))   # Order 7
        
        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(RS40M)),    # Order ≤ 5  
                                       $(FT.(RS443)))    # Order 7

        # 9th order WENO, restricted to orders 7, 5, 3, and 1
        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS510)),    # Order 1                                                
                ifelse(red_order == 2, $(FT.(RS520)),    # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS530)),    # Order 5  
                ifelse(red_order == 4, $(FT.(RS540)),    # Order 7
                                       $(FT.(RS550)))))) # Order 9

        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS50M)),    # Order 1                                                
                ifelse(red_order == 2, $(FT.(RS521)),    # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS531)),    # Order 5  
                ifelse(red_order == 4, $(FT.(RS541)),    # Order 7
                                       $(FT.(RS551)))))) # Order 9

        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS50M)),    # Order ≤ 3                                             
                ifelse(red_order == 3, $(FT.(RS532)),    # Order 5  
                ifelse(red_order == 4, $(FT.(RS542)),    # Order 7
                                       $(FT.(RS552)))))  # Order 9
        
        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(RS50M)),    # Order ≤ 5  
                ifelse(red_order == 4, $(FT.(RS543)),    # Order 7
                                       $(FT.(RS553))))   # Order 9

        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{4}) = 
                ifelse(red_order <  5, $(FT.(RS50M)),    # Order ≤ 7
                                       $(FT.(RS554)))    # Order 9
    end
end
