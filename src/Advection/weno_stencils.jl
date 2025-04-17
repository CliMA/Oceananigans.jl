
# WENO stencils
#
# This file contains the WENO stencils used in the WENO advection scheme.
#
# The stencils are defined for different orders of accuracy and for different ``reduced orders'', meaning that the
# stencils are limited to a ``reduced'' (lower) order of accuracy. where the order is reduced we place zeros instead
# of the values in the stencil, and zeroing out the last redundant WENO stencils.
#
# This strategy avoids having to define a new stencil for each order of accuracy.

# Utility function to get the zero value of a type
@inline z(T) = zero(T)

# WENO 3rd order limited to 1st order
@inline S₀₂(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (S[2], z(T)), (S[3], z(T))), 
                                                                                              ifelse(bias isa LeftBias, (S[2], S[3]), (S[3], S[2])))

@inline S₁₂(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (z(T), z(T)), (z(T), z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[1], S[2]), (S[4], S[3])))

# WENO 5th order limited to 3rd order, and 1st order
@inline S₀₃(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (S[3], z(T), z(T)), (S[4], z(T), z(T))),
                                                                       ifelse(red_order == 2, ifelse(bias isa LeftBias, (S[3], S[4], z(T)), (S[4], S[3], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[3], S[4], S[5]), (S[4], S[3], S[2]))))

@inline S₁₃(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (z(T), z(T), z(T)), (z(T), z(T), z(T))),
                                                                       ifelse(red_order == 2, ifelse(bias isa LeftBias, (S[2], S[3], z(T)), (S[5], S[4], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[2], S[3], S[4]), (S[5], S[4], S[3]))))

@inline S₂₃(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order <  3, ifelse(bias isa LeftBias, (z(T), z(T), z(T)), (z(T), z(T), z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[1], S[2], S[3]), (S[6], S[5], S[4])))

# WENO 7th order limited to 5th order, 3rd order, and 1st order
@inline S₀₄(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (S[4], z(T), z(T), z(T)), (S[5], z(T), z(T), z(T))),
                                                                       ifelse(red_order == 2, ifelse(bias isa LeftBias, (S[4], S[5], z(T), z(T)), (S[5], S[4], z(T), z(T))),
                                                                       ifelse(red_order == 3, ifelse(bias isa LeftBias, (S[4], S[5], S[6], z(T)), (S[5], S[4], S[3], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7]), (S[5], S[4], S[3], S[2])))))

@inline S₁₄(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T))),
                                                                       ifelse(red_order == 2, ifelse(bias isa LeftBias, (S[3], S[4], z(T), z(T)), (S[6], S[5], z(T), z(T))),
                                                                       ifelse(red_order == 3, ifelse(bias isa LeftBias, (S[3], S[4], S[5], z(T)), (S[6], S[5], S[4], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6]), (S[6], S[5], S[4], S[3])))))

@inline S₂₄(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order <  3, ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T))),
                                                                       ifelse(red_order == 3, ifelse(bias isa LeftBias, (S[2], S[3], S[4], z(T)), (S[7], S[6], S[5], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5]), (S[7], S[6], S[5], S[4]))))

@inline S₃₄(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order <  4, ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4]), (S[8], S[7], S[6], S[5])))

# WENO 9th order limited to 7th order, 5th order, 3rd order, and 1st order
@inline S₀₅(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (S[5], z(T), z(T), z(T), z(T)), (S[6], z(T), z(T), z(T), z(T))),
                                                                       ifelse(red_order == 2, ifelse(bias isa LeftBias, (S[5], S[6], z(T), z(T), z(T)), (S[6], S[5], z(T), z(T), z(T))),
                                                                       ifelse(red_order == 3, ifelse(bias isa LeftBias, (S[5], S[6], S[7], z(T), z(T)), (S[6], S[5], S[4], z(T), z(T))),
                                                                       ifelse(red_order == 4, ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], z(T)), (S[6], S[5], S[4], S[3], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], S[9]), (S[6], S[5], S[4], S[3], S[2]))))))

@inline S₁₅(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order == 1, ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T))),
                                                                       ifelse(red_order == 2, ifelse(bias isa LeftBias, (S[4], S[5], z(T), z(T), z(T)), (S[7], S[6], z(T), z(T), z(T))),
                                                                       ifelse(red_order == 3, ifelse(bias isa LeftBias, (S[4], S[5], S[6], z(T), z(T)), (S[7], S[6], S[5], z(T), z(T))),
                                                                       ifelse(red_order == 4, ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], z(T)), (S[7], S[6], S[5], S[4], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], S[8]), (S[7], S[6], S[5], S[4], S[3]))))))

@inline S₂₅(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order <  3, ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T))),
                                                                       ifelse(red_order == 3, ifelse(bias isa LeftBias, (S[3], S[4], S[5], z(T), z(T)), (S[8], S[7], S[6], z(T), z(T))),
                                                                       ifelse(red_order == 4, ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], z(T)), (S[8], S[7], S[6], S[5], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], S[7]), (S[8], S[7], S[6], S[5], S[4])))))

@inline S₃₅(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order <  4, ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T), z(T), z(T), z(T), z(T))),
                                                                       ifelse(red_order == 4, ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], z(T)), (S[9], S[8], S[7], S[6], z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], S[6]), (S[9], S[8], S[7], S[6], S[5]))))

@inline S₄₅(S::NTuple{N, T}, red_order, bias) where {N, T} = @inbounds ifelse(red_order <  5, ifelse(bias isa LeftBias, (z(T), z(T), z(T), z(T), z(T)), (z(T),  z(T), z(T), z(T), z(T))),
                                                                                              ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4], S[5]), (S[10], S[9], S[8], S[7], S[6])))

