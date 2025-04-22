
# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)

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

# Cefficients for fifth order WENO
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
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{2}, red_order, ::Val{0}) = $(FT.(RS220))
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{2}, red_order, ::Val{1}) = $(FT.(RS221))

        # 5th order WENO, restricted to orders 3 and 1
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{3}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS310)),                  
                ifelse(red_order == 2, $(FT.(RS320)),                                             
                                       $(FT.(RS330)))) # Full order (3)

        @inline reconstruction_coefficients(::Val{$FT}, ::Val{3}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS30M)),
                ifelse(red_order == 2, $(FT.(RS321)),
                                       $(FT.(RS331)))) # Full order (3)

        @inline reconstruction_coefficients(::Val{$FT}, ::Val{3}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS30M)), 
                                       $(FT.(RS332))) # Full order (3)

        # 7th order WENO, restricted to orders 5, 3, and 1
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS410)),                               
                ifelse(red_order == 2, $(FT.(RS420)), 
                ifelse(red_order == 3, $(FT.(RS430)), 
                                       $(FT.(RS440))))) # Full order
        
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS40M)),                 
                ifelse(red_order == 2, $(FT.(RS421)),
                ifelse(red_order == 3, $(FT.(RS431)),
                                       $(FT.(RS441))))) # Full order

        @inline reconstruction_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{2}) = 
                ifelse(red_order  < 3, $(FT.(RS40M)),                        
                ifelse(red_order == 3, $(FT.(RS432)),
                                       $(FT.(RS442)))) # Full order
        
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{4}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(RS40M)),                                              
                                       $(FT.(RS443))) # Full order

        # 9th order WENO, restricted to orders 7, 5, 3, and 1
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS510)),                                
                ifelse(red_order == 2, $(FT.(RS520)),          
                ifelse(red_order == 3, $(FT.(RS530)),
                ifelse(red_order == 4, $(FT.(RS540)),
                                       $(FT.(RS550)))))) # Full order

        @inline reconstruction_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS50M)),
                ifelse(red_order == 2, $(FT.(RS521)),
                ifelse(red_order == 3, $(FT.(RS531)),
                ifelse(red_order == 4, $(FT.(RS541)),
                                       $(FT.(RS551)))))) # Full order

        @inline reconstruction_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS50M)),                                
                ifelse(red_order == 3, $(FT.(RS532)),                                    
                ifelse(red_order == 4, $(FT.(RS542)),
                                       $(FT.(RS552))))) # Full order
        
        @inline reconstruction_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(RS50M)),                                              
                ifelse(red_order == 4, $(FT.(RS543)),                                             
                                       $(FT.(RS553)))) # Full order

        @inline reconstruction_coefficients(::Val{$FT}, ::Val{5}, red_order, ::Val{4}) = 
                ifelse(red_order <  5, $(FT.(RS50M)),
                                       $(FT.(RS554))) # Full order
    end
end
