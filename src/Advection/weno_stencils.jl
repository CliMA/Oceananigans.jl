
# WENO stencils
#
# This file contains the WENO stencils used in the WENO advection scheme.
#
# The stencils are defined for different orders of accuracy and for different ``reduced orders'', meaning that the
# stencils are limited to a ``reduced'' (lower) order of accuracy. where the order is reduced we place zeros instead
# of the values in the stencil, and zeroing out the last redundant WENO stencils.
#
# This strategy avoids having to define a new stencil for each order of accuracy.

# WENO 3rd order limited to 1st order
@inline S₀₂(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[2], S[3]), (S[3], S[2]))
@inline S₁₂(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[1], S[2]), (S[4], S[3]))

# WENO 5th order limited to 3rd order, and 1st Order
@inline S₀₃(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5]), (S[4], S[3], S[2]))
@inline S₁₃(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4]), (S[5], S[4], S[3]))
@inline S₂₃(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3]), (S[6], S[5], S[4]))

# WENO 7th order limited to 5th order, 3rd order, and 1st order
@inline S₀₄(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7]), (S[5], S[4], S[3], S[2]))
@inline S₁₄(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6]), (S[6], S[5], S[4], S[3]))
@inline S₂₄(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5]), (S[7], S[6], S[5], S[4]))
@inline S₃₄(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4]), (S[8], S[7], S[6], S[5]))

# WENO 9th order limited to 7th order, 5th order, 3rd order, and 1st order
@inline S₀₅(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], S[9]), (S[6],  S[5], S[4], S[3], S[2]))
@inline S₁₅(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], S[8]), (S[7],  S[6], S[5], S[4], S[3]))
@inline S₂₅(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], S[7]), (S[8],  S[7], S[6], S[5], S[4]))
@inline S₃₅(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], S[6]), (S[9],  S[8], S[7], S[6], S[5]))
@inline S₄₅(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4], S[5]), (S[10], S[9], S[8], S[7], S[6]))

# WENO 11th order limited to 9th order, 7th order, 5th order, 3rd order, and 1st order
@inline S₀₆(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[6], S[7], S[8], S[9], S[10], S[11]), (S[7],  S[6],  S[5],  S[4], S[3], S[2]))
@inline S₁₆(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[5], S[6], S[7], S[8], S[9],  S[10]), (S[8],  S[7],  S[6],  S[5], S[4], S[3]))
@inline S₂₆(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[4], S[5], S[6], S[7], S[8],  S[9]),  (S[9],  S[8],  S[7],  S[6], S[5], S[4]))
@inline S₃₆(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[3], S[4], S[5], S[6], S[7],  S[8]),  (S[10], S[9],  S[8],  S[7], S[6], S[5]))
@inline S₄₆(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[2], S[3], S[4], S[5], S[6],  S[7]),  (S[11], S[10], S[9],  S[8], S[7], S[6]))
@inline S₅₆(S, bias) = @inbounds ifelse(bias isa LeftBias, (S[1], S[2], S[3], S[4], S[5],  S[6]),  (S[12], S[11], S[10], S[9], S[8], S[7]))
