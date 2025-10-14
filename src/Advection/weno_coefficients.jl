# _UNIFORM_ reconstruction coefficients

# The naming convention for the coefficients is as follows:
# RSXYZ
# RS -> reconstruction stencil
# X  -> order of the WENO reconstruction
# Y  -> reduced order of the WENO reconstruction (with Y ≤ X)
# Z  -> stencil number  
# 
# The zero stencil for order X is named RSX0M

# Zero and one coefficient stencils
const RS20M = (0, 0)
const RS210 = (1, 1) .// 2 
const RS30M = (0, 0, 0)
const RS310 = (1, 1, 0) .// 2
const RS40M = (0, 0, 0, 0)
const RS410 = (1, 1, 0, 0) .// 2
const RS50M = (0, 0, 0, 0, 0)
const RS510 = (1, 1, 0, 0, 0) .// 2
const RS60M = (0, 0, 0, 0, 0, 0)
const RS610 = (1, 1, 0, 0, 0, 0) .// 2

# Coefficients for third order WENO reconstruction
const RS220 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=2)
const RS221 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=2)
const RS320 = (RS220..., 0)
const RS321 = (RS221..., 0)
const RS420 = (RS320..., 0)
const RS421 = (RS321..., 0)
const RS520 = (RS420..., 0)
const RS521 = (RS421..., 0) 
const RS620 = (RS520..., 0)
const RS621 = (RS521..., 0)

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
const RS630 = (RS530..., 0)
const RS631 = (RS531..., 0)
const RS632 = (RS532..., 0)

# Coefficients for seventh order WENO
const RS440 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=4)
const RS441 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=4)
const RS442 = stencil_coefficients(BigFloat, 50, 2, collect(1:100), collect(1:100); order=4)
const RS443 = stencil_coefficients(BigFloat, 50, 3, collect(1:100), collect(1:100); order=4)
const RS540 = (RS440..., 0)
const RS541 = (RS441..., 0)
const RS542 = (RS442..., 0)
const RS543 = (RS443..., 0)
const RS640 = (RS540..., 0)
const RS641 = (RS541..., 0)
const RS642 = (RS542..., 0)
const RS643 = (RS543..., 0)

# Coefficients for ninth order WENO
const RS550 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=5)
const RS551 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=5)
const RS552 = stencil_coefficients(BigFloat, 50, 2, collect(1:100), collect(1:100); order=5)
const RS553 = stencil_coefficients(BigFloat, 50, 3, collect(1:100), collect(1:100); order=5)
const RS554 = stencil_coefficients(BigFloat, 50, 4, collect(1:100), collect(1:100); order=5)
const RS650 = (RS550..., 0)
const RS651 = (RS551..., 0)
const RS652 = (RS552..., 0)
const RS653 = (RS553..., 0)
const RS654 = (RS554..., 0)

# Coefficients for eleventh order WENO
const RS660 = stencil_coefficients(BigFloat, 50, 0, collect(1:100), collect(1:100); order=6)
const RS661 = stencil_coefficients(BigFloat, 50, 1, collect(1:100), collect(1:100); order=6)
const RS662 = stencil_coefficients(BigFloat, 50, 2, collect(1:100), collect(1:100); order=6)
const RS663 = stencil_coefficients(BigFloat, 50, 3, collect(1:100), collect(1:100); order=6)
const RS664 = stencil_coefficients(BigFloat, 50, 4, collect(1:100), collect(1:100); order=6)
const RS665 = stencil_coefficients(BigFloat, 50, 5, collect(1:100), collect(1:100); order=6)

for FT in fully_supported_float_types
    @eval begin
        """
            reconstruction_coefficients(::Val{FT}, ::Val{buffer}, ::Val{red_order}, ::Val{stencil})

        Return the coefficients used to calculate the ....
        """
        # 3rd order WENO, restricted to order 1 (does not matter the restriction order here)
        @inline reconstruction_coefficients(::WENO{2, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS210)),     # Order 1 (Centered)
                                       $(FT.(RS220)))     # Order 3

        @inline reconstruction_coefficients(::WENO{2, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS20M)),     # Order 1 (Centered)
                                       $(FT.(RS221)))     # Order 3

        # 5th order WENO, restricted to orders 3 and 1
        @inline reconstruction_coefficients(::WENO{3, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS310)),     # Order 1 (Centered)               
                ifelse(red_order == 2, $(FT.(RS320)),     # Order 3                                           
                                       $(FT.(RS330))))    # Order 5

        @inline reconstruction_coefficients(::WENO{3, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS30M)),     # Order 1 (Centered)                
                ifelse(red_order == 2, $(FT.(RS321)),     # Order 3                                           
                                       $(FT.(RS331))))    # Order 5

        @inline reconstruction_coefficients(::WENO{3, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS30M)),     # Order ≤ 3                                          
                                       $(FT.(RS332)))     # Order 5

        # 7th order WENO, restricted to orders 5, 3, and 1
        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS410)),     # Order 1 (Centered)                                                
                ifelse(red_order == 2, $(FT.(RS420)),     # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS430)),     # Order 5  
                                       $(FT.(RS440)))))   # Order 7
        
        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS40M)),     # Order 1 (Centered)                                                
                ifelse(red_order == 2, $(FT.(RS421)),     # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS431)),     # Order 5  
                                       $(FT.(RS441)))))   # Order 7

        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS40M)),     # Order ≤ 3                                             
                ifelse(red_order == 3, $(FT.(RS432)),     # Order 5  
                                       $(FT.(RS442))))    # Order 7
        
        @inline reconstruction_coefficients(::WENO{4, $FT}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(RS40M)),     # Order ≤ 5  
                                       $(FT.(RS443)))     # Order 7

        # 9th order WENO, restricted to orders 7, 5, 3, and 1
        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS510)),     # Order 1 (Centered)                                                           
                ifelse(red_order == 2, $(FT.(RS520)),     # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS530)),     # Order 5  
                ifelse(red_order == 4, $(FT.(RS540)),     # Order 7
                                       $(FT.(RS550))))))  # Order 9

        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS50M)),     # Order 1 (Centered)                                                
                ifelse(red_order == 2, $(FT.(RS521)),     # Order 3                                             
                ifelse(red_order == 3, $(FT.(RS531)),     # Order 5  
                ifelse(red_order == 4, $(FT.(RS541)),     # Order 7
                                       $(FT.(RS551))))))  # Order 9

        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS50M)),     # Order ≤ 3                                             
                ifelse(red_order == 3, $(FT.(RS532)),     # Order 5  
                ifelse(red_order == 4, $(FT.(RS542)),     # Order 7
                                       $(FT.(RS552)))))   # Order 9
        
        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(RS50M)),     # Order ≤ 5  
                ifelse(red_order == 4, $(FT.(RS543)),     # Order 7
                                       $(FT.(RS553))))    # Order 9

        @inline reconstruction_coefficients(::WENO{5, $FT}, red_order, ::Val{4}) = 
                ifelse(red_order <  5, $(FT.(RS50M)),     # Order ≤ 7
                                       $(FT.(RS554)))     # Order 9

        # 11th order WENO, restricted to orders 9, 7, 5, 3, and 1
        @inline reconstruction_coefficients(::WENO{6, $FT}, red_order, ::Val{0}) = 
                ifelse(red_order == 1, $(FT.(RS610)),     # Order 1 (Centered)                             
                ifelse(red_order == 2, $(FT.(RS620)),     # Order 3       
                ifelse(red_order == 3, $(FT.(RS630)),     # Order 5
                ifelse(red_order == 4, $(FT.(RS640)),     # Order 7
                ifelse(red_order == 5, $(FT.(RS650)),     # Order 9
                                       $(FT.(RS660))))))) # Order 11

        @inline reconstruction_coefficients(::WENO{6, $FT}, red_order, ::Val{1}) = 
                ifelse(red_order == 1, $(FT.(RS60M)),     # Order 1 (Centered)                             
                ifelse(red_order == 2, $(FT.(RS621)),     # Order 3       
                ifelse(red_order == 3, $(FT.(RS631)),     # Order 5
                ifelse(red_order == 4, $(FT.(RS641)),     # Order 7
                ifelse(red_order == 5, $(FT.(RS651)),     # Order 9
                                       $(FT.(RS661))))))) # Order 11

        @inline reconstruction_coefficients(::WENO{6, $FT}, red_order, ::Val{2}) = 
                ifelse(red_order <  3, $(FT.(RS60M)),     # Order ≤ 3                            
                ifelse(red_order == 3, $(FT.(RS632)),     # Order 5
                ifelse(red_order == 4, $(FT.(RS642)),     # Order 7
                ifelse(red_order == 5, $(FT.(RS652)),     # Order 9
                                       $(FT.(RS662))))))  # Order 11

        @inline reconstruction_coefficients(::WENO{6, $FT}, red_order, ::Val{3}) = 
                ifelse(red_order <  4, $(FT.(RS60M)),     # Order ≤ 5 
                ifelse(red_order == 4, $(FT.(RS643)),     # Order 7
                ifelse(red_order == 5, $(FT.(RS653)),     # Order 9
                                       $(FT.(RS663)))))   # Order 11

        @inline reconstruction_coefficients(::WENO{6, $FT}, red_order, ::Val{4}) = 
                ifelse(red_order <  5, $(FT.(RS60M)),     # Order ≤ 7
                ifelse(red_order == 5, $(FT.(RS654)),     # Order 9
                                       $(FT.(RS664))))    # Order 11

        @inline reconstruction_coefficients(::WENO{6, $FT}, red_order, ::Val{5}) = 
                ifelse(red_order <  6, $(FT.(RS60M)),     # Order ≤ 9
                                       $(FT.(RS665)))     # Order 11
    end
end

# Optimal coefficients for WENO schemes
const C20 = 2//3
const C21 = 1//3

const C30 = 3//10
const C31 = 3//5
const C32 = 1//10

const C40 = 4//35
const C41 = 18//35
const C42 = 12//35
const C43 = 1//35

const C50 = 5//126
const C51 = 20//63
const C52 = 100//231
const C53 = 10//63
const C54 = 1//126

const C60 = 1//77
const C61 = 25//154
const C62 = 100//231
const C63 = 25//77
const C64 = 5//77
const C65 = 1//462

# Optimal values for finite volume reconstruction of order `WENO{order}` and stencil `Val{stencil}` from
# Balsara & Shu, "Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes with Inceasingly High Order of Accuracy"
for FT in fully_supported_float_types
    @eval begin
        # WENO 3rd order
        @inline C★(::WENO{2, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), $(FT(C20)))
        @inline C★(::WENO{2, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), $(FT(C21)))

        # WENO 5th order
        @inline C★(::WENO{3, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), ifelse(red_order==2, $(FT(C20)), $(FT(C30))))
        @inline C★(::WENO{3, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), ifelse(red_order==2, $(FT(C21)), $(FT(C31))))
        @inline C★(::WENO{3, $FT}, red_order, ::Val{2}) = ifelse(red_order <3, $(FT(0)), $(FT(C32)))

        # WENO 7th order
        @inline C★(::WENO{4, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), ifelse(red_order==2, $(FT(C20)), ifelse(red_order==3, $(FT(C30)), $(FT(C40)))))
        @inline C★(::WENO{4, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), ifelse(red_order==2, $(FT(C21)), ifelse(red_order==3, $(FT(C31)), $(FT(C41)))))
        @inline C★(::WENO{4, $FT}, red_order, ::Val{2}) = ifelse(red_order <3, $(FT(0)), ifelse(red_order==3, $(FT(C32)), $(FT(C42))))
        @inline C★(::WENO{4, $FT}, red_order, ::Val{3}) = ifelse(red_order <4, $(FT(0)), $(FT(C43)))

        # WENO 9th order
        @inline C★(::WENO{5, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), ifelse(red_order==2, $(FT(C20)), ifelse(red_order==3, $(FT(C30)), ifelse(red_order==4, $(FT(C40)), $(FT(C50))))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), ifelse(red_order==2, $(FT(C21)), ifelse(red_order==3, $(FT(C31)), ifelse(red_order==4, $(FT(C41)), $(FT(C51))))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{2}) = ifelse(red_order <3, $(FT(0)), ifelse(red_order==3, $(FT(C32)), ifelse(red_order==4, $(FT(C42)), $(FT(C52)))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{3}) = ifelse(red_order <4, $(FT(0)), ifelse(red_order==4, $(FT(C43)), $(FT(C53))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{4}) = ifelse(red_order <5, $(FT(0)), $(FT(C54)))

        # WENO 11th order
        @inline C★(::WENO{6, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), ifelse(red_order==2, $(FT(C20)), ifelse(red_order==3, $(FT(C30)), ifelse(red_order==4, $(FT(C40)), ifelse(red_order==5, $(FT(C50)), $(FT(C60)))))))
        @inline C★(::WENO{6, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), ifelse(red_order==2, $(FT(C21)), ifelse(red_order==3, $(FT(C31)), ifelse(red_order==4, $(FT(C41)), ifelse(red_order==5, $(FT(C51)), $(FT(C61)))))))
        @inline C★(::WENO{6, $FT}, red_order, ::Val{2}) = ifelse(red_order <3, $(FT(0)), ifelse(red_order==3, $(FT(C32)), ifelse(red_order==4, $(FT(C42)), ifelse(red_order==5, $(FT(C52)), $(FT(C62))))))
        @inline C★(::WENO{6, $FT}, red_order, ::Val{3}) = ifelse(red_order <4, $(FT(0)), ifelse(red_order==4, $(FT(C43)), ifelse(red_order==5, $(FT(C53)), $(FT(C63)))))
        @inline C★(::WENO{6, $FT}, red_order, ::Val{4}) = ifelse(red_order <5, $(FT(0)), ifelse(red_order==5, $(FT(C54)), $(FT(C64))))
        @inline C★(::WENO{6, $FT}, red_order, ::Val{5}) = ifelse(red_order <6, $(FT(0)), $(FT(C65)))
    end
end
