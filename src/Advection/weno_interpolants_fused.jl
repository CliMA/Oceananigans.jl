#####
##### Low-register WENO interpolation using sliding window approach
#####
##### Instead of loading all stencils at once (high register pressure), we:
##### 1. Load initial window of `buffer` values
##### 2. For each stencil: compute β and biased_p, then slide window by one value
##### 3. Compute weights from all β values
##### 4. Return weighted sum of biased_p values
#####
##### This reduces register usage from O(buffer²) to O(buffer)
#####

#####
##### Helper function to unify array and function access
#####

@inline getvalue(a, i, j, k, grid, args...) = @inbounds a[i, j, k]
@inline getvalue(a::Function, i, j, k, grid, args...) = a(i, j, k, grid, args...)

#####
##### Helper function to compute index offset based on bias, stencil, and position
#####

@inline stencil_offset(::LeftBias, s, p)  = p - s - 1
@inline stencil_offset(::RightBias, s, p) = s - p

#####
##### Code generation function to create WENO interpolation methods for all three directions
#####
function generate_weno_body(N, dir_name, dir_suffix)
    num_stencils = N + 1
    num_positions = N + 1
    
    # Build the function body step by step
    body_lines = Expr[]
    
    # Stencil 0: load initial window
    push!(body_lines, Expr(:comment, "Stencil 0: load initial window"))
    for p in 0:(num_positions-1)
        w_idx = p + 1
        if dir_name == :x
            push!(body_lines, :(w$w_idx = getvalue(ψ, i + stencil_offset(bias, 0, $p), j, k, grid, args...)))
        elseif dir_name == :y
            push!(body_lines, :(w$w_idx = getvalue(ψ, i, j + stencil_offset(bias, 0, $p), k, grid, args...)))
        else  # z
            push!(body_lines, :(w$w_idx = getvalue(ψ, i, j, k + stencil_offset(bias, 0, $p), grid, args...)))
        end
    end
    
    # Build w tuple for stencil 0
    w_vars = [Symbol("w$i") for i in 1:num_positions]
    w_tuple = Expr(:tuple, w_vars...)
    push!(body_lines, :(β1 = smoothness_indicator($w_tuple, scheme, Val(0))))
    push!(body_lines, :(p1 = biased_p(scheme, bias, Val(0), $w_tuple)))
    
    # Subsequent stencils
    for s in 1:(num_stencils-1)
        push!(body_lines, Expr(:comment, "Slide for stencil $s"))
        
        # Slide operations - combine with semicolons
        slide_ops = Expr[]
        for i in num_positions:-1:2
            push!(slide_ops, :(w$i = w$(i-1)))
        end
        # Combine slide operations with semicolons
        if length(slide_ops) > 1
            combined_slide = slide_ops[1]
            for i in 2:length(slide_ops)
                combined_slide = Expr(:(;), combined_slide, slide_ops[i])
            end
            push!(body_lines, combined_slide)
        elseif length(slide_ops) == 1
            push!(body_lines, slide_ops[1])
        end
        
        # Load new w1
        if dir_name == :x
            push!(body_lines, :(w1 = getvalue(ψ, i + stencil_offset(bias, $s, 0), j, k, grid, args...)))
        elseif dir_name == :y
            push!(body_lines, :(w1 = getvalue(ψ, i, j + stencil_offset(bias, $s, 0), k, grid, args...)))
        else  # z
            push!(body_lines, :(w1 = getvalue(ψ, i, j, k + stencil_offset(bias, $s, 0), grid, args...)))
        end
        
        # Compute β and p (w_tuple is still valid after slide)
        push!(body_lines, :(β$(s+1) = smoothness_indicator($w_tuple, scheme, Val($s))))
        push!(body_lines, :(p$(s+1) = biased_p(scheme, bias, Val($s), $w_tuple)))
    end
    
    # Compute weights
    push!(body_lines, Expr(:comment, "Compute weights"))
    β_vars = [Symbol("β$i") for i in 1:num_stencils]
    β_tuple = Expr(:tuple, β_vars...)
    push!(body_lines, :(τ = global_smoothness_indicator(Val($N), $β_tuple)))
    push!(body_lines, :(α = zweno_alpha_loop(scheme, $β_tuple, τ)))
    
    # Sum of alphas
    α_sum = Expr(:call, :+, [:(α[$i]) for i in 1:num_stencils]...)
    push!(body_lines, :(Σα = $α_sum))
    
    # Return statement
    p_vars = [Symbol("p$i") for i in 1:num_stencils]
    p_sum = Expr(:call, :+, [:(α[$i] * $(p_vars[i])) for i in 1:num_stencils]...)
    push!(body_lines, :(return @muladd ($p_sum) / Σα))
    
    return Expr(:block, body_lines...)
end

# Generate all methods
for N in [2, 3, 4, 5, 6]
    for (dir_name, dir_suffix) in [(:x, :xᶠᵃᵃ), (:y, :yᵃᶠᵃ), (:z, :zᵃᵃᶠ)]
        func_name = Symbol("fused_biased_interpolate_$dir_suffix")
        body = generate_weno_body(N, dir_name, dir_suffix)
        
        @eval @inline function $func_name(i, j, k, grid,
                                          scheme::WENO{$N, FT}, bias,
                                          ψ, args...) where FT
            $body
        end
    end
end
