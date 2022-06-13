using OffsetArrays
using Oceananigans.Grids: with_halo, return_metrics
using Oceananigans.Architectures: arch_array, architecture
using KernelAbstractions.Extras.LoopInfo: @unroll
using Adapt
import Base: show

#####
##### biased pₖ for û calculation
#####

@inline left_biased_p₀(scheme, ψ, args...) = @inbounds sum(coeff_left_p₀(scheme, args...) .* ψ)
@inline left_biased_p₁(scheme, ψ, args...) = @inbounds sum(coeff_left_p₁(scheme, args...) .* ψ)
@inline left_biased_p₂(scheme, ψ, args...) = @inbounds sum(coeff_left_p₂(scheme, args...) .* ψ)

@inline right_biased_p₀(scheme, ψ, args...) = @inbounds sum(coeff_right_p₀(scheme, args...) .* ψ)
@inline right_biased_p₁(scheme, ψ, args...) = @inbounds sum(coeff_right_p₁(scheme, args...) .* ψ)
@inline right_biased_p₂(scheme, ψ, args...) = @inbounds sum(coeff_right_p₂(scheme, args...) .* ψ)

#####
##### Coefficients for stretched (and uniform) ENO schemes (see Shu NASA/CR-97-206253, ICASE Report No. 97-65)
#####

@inline coeff_right_p₀(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₂(scheme, Nothing, args...)) 
@inline coeff_right_p₁(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₁(scheme, Nothing, args...)) 
@inline coeff_right_p₂(scheme, ::Type{Nothing}, args...) = reverse(coeff_left_p₀(scheme, Nothing, args...)) 

@inline coeff_left_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 0, dir, i ,loc)
@inline coeff_left_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 1, dir, i ,loc)
@inline coeff_left_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme, 2, dir, i ,loc)

@inline coeff_right_p₀(scheme, T, dir, i, loc) = retrieve_coeff(scheme, -1, dir, i ,loc)
@inline coeff_right_p₁(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  0, dir, i ,loc)
@inline coeff_right_p₂(scheme, T, dir, i, loc) = retrieve_coeff(scheme,  1, dir, i ,loc)

@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.coeff_xᶠᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.coeff_xᶜᵃᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.coeff_yᵃᶠᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.coeff_yᵃᶜᵃ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.coeff_zᵃᵃᶠ[r+2][i] 
@inline retrieve_coeff(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.coeff_zᵃᵃᶜ[r+2][i] 

@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+1][i] 
@inline retrieve_left_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+1][i] 

@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Face})   = scheme.smooth_xᶠᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{1}, i, ::Type{Center}) = scheme.smooth_xᶜᵃᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Face})   = scheme.smooth_yᵃᶠᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{2}, i, ::Type{Center}) = scheme.smooth_yᵃᶜᵃ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Face})   = scheme.smooth_zᵃᵃᶠ[r+4][i] 
@inline retrieve_right_smooth(scheme, r, ::Val{3}, i, ::Type{Center}) = scheme.smooth_zᵃᵃᶜ[r+4][i] 



function compute_stretched_weno_coefficients(grid, stretched_smoothness, FT; order = 3)
    
    rect_metrics = (:xᶠᵃᵃ, :xᶜᵃᵃ, :yᵃᶠᵃ, :yᵃᶜᵃ, :zᵃᵃᶠ, :zᵃᵃᶜ)

    if grid isa Nothing
        @warn "defaulting to uniform WENO scheme with $(FT) precision, use WENO5(grid = grid) if this was not intended"
        for metric in rect_metrics
            @eval $(Symbol(:coeff_ , metric)) = nothing
            @eval $(Symbol(:smooth_, metric)) = nothing
        end
    else
        !(grid isa RectilinearGrid) && (@warn "WENO on a curvilinear stretched coordinate is not validated, use at your own risk!!")

        metrics = return_metrics(grid)
        dirsize = (:Nx, :Nx, :Ny, :Ny, :Nz, :Nz)

        arch     = architecture(grid)
        new_grid = with_halo((4, 4, 4), grid)

        for (dir, metric, rect_metric) in zip(dirsize, metrics, rect_metrics)
            @eval $(Symbol(:coeff_ , rect_metric)) = calc_interpolating_coefficients($FT, $new_grid.$metric, $arch, $new_grid.$dir; order)
            @eval $(Symbol(:smooth_, rect_metric)) = calc_smoothness_coefficients($FT, $Val($stretched_smoothness), $new_grid.$metric, $arch, $new_grid.$dir; order) 
        end
    end

    return (coeff_xᶠᵃᵃ , coeff_xᶜᵃᵃ , coeff_yᵃᶠᵃ , coeff_yᵃᶜᵃ , coeff_zᵃᵃᶠ , coeff_zᵃᵃᶜ ,
            smooth_xᶠᵃᵃ, smooth_xᶜᵃᵃ, smooth_yᵃᶠᵃ, smooth_yᵃᶜᵃ, smooth_zᵃᵃᶠ, smooth_zᵃᵃᶜ)
end


@inline calc_interpolating_coefficients(FT, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N; order) = nothing
@inline calc_interpolating_coefficients(FT, coord::AbstractRange, arch, N; order)                              = nothing

@inline calc_smoothness_coefficients(FT, ::Val{false}, args...; kwargs...) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::OffsetArray{<:Any, <:Any, <:AbstractRange}, arch, N; order) = nothing
@inline calc_smoothness_coefficients(FT, ::Val{true}, coord::AbstractRange, arch, N; order) = nothing

function calc_interpolating_coefficients(FT, coord, arch, N; order = 3) 

    cpu_coord = Array(parent(coord))
    cpu_coord = OffsetArray(cpu_coord, coord.offsets[1])

    s1 = create_interp_coefficients(FT,-1, cpu_coord, arch, N; order)
    s2 = create_interp_coefficients(FT, 0, cpu_coord, arch, N; order)
    s3 = create_interp_coefficients(FT, 1, cpu_coord, arch, N; order)
    s4 = create_interp_coefficients(FT, 2, cpu_coord, arch, N; order)

    return (s1, s2, s3, s4)
end

function create_interp_coefficients(FT, r, cpu_coord, arch, N; order = 3)

    stencil = NTuple{3, FT}[]
    @inbounds begin
        for i = 0:N+1
            push!(stencil, stencil_coefficients(i, r, cpu_coord, cpu_coord; order))     
        end
    end
    return OffsetArray(arch_array(arch, stencil), -1)
end

function calc_smoothness_coefficients(FT, beta, coord, arch, N; order = 3) 

    cpu_coord = arch_array(CPU(), coord)

    s1 = create_smoothness_coefficients(FT, 0, -, cpu_coord, arch, N; order)
    s2 = create_smoothness_coefficients(FT, 1, -, cpu_coord, arch, N; order)
    s3 = create_smoothness_coefficients(FT, 2, -, cpu_coord, arch, N; order)
    s4 = create_smoothness_coefficients(FT, 0, +, cpu_coord, arch, N; order)
    s5 = create_smoothness_coefficients(FT, 1, +, cpu_coord, arch, N; order)
    s6 = create_smoothness_coefficients(FT, 2, +, cpu_coord, arch, N; order)
    
    return (s1, s2, s3, s4, s5, s6)
end

struct FirstDerivative end
struct SecondDerivative end
struct Primitive end

function create_smoothness_coefficients(FT, r, op, cpu_coord, arch, N; order)

    # derivation written on overleaf
    stencil = NTuple{2, NTuple{3, FT}}[]   
    @inbounds begin
        for i = 0:N+1
       
            bias1 = Int(op == +)
            bias2 = bias1 - 1

            Δcᵢ = cpu_coord[i + bias1] - cpu_coord[i + bias2]
        
            Bᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1, der = Primitive())
            bᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1)
            bₓᵢ = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1, der = FirstDerivative())
            Aᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias2, der = Primitive())
            aᵢ  = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias2)
            aₓᵢ = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias2, der = FirstDerivative())

            pₓₓ = stencil_coefficients(i, r, cpu_coord, cpu_coord; order, op, shift = bias1, der = SecondDerivative())
            Pᵢ  =  (Bᵢ .- Aᵢ)

            wᵢᵢ = Δcᵢ  .* (bᵢ .* bₓᵢ .- aᵢ .* aₓᵢ .- pₓₓ .* Pᵢ)  .+ Δcᵢ^4 .* (pₓₓ .* pₓₓ)
            wᵢⱼ = Δcᵢ  .* (star(bᵢ, bₓᵢ)  .- star(aᵢ, aₓᵢ) .- star(pₓₓ, Pᵢ)) .+
                                                 Δcᵢ^4 .* star(pₓₓ, pₓₓ)

            push!(stencil, (wᵢᵢ, wᵢⱼ))
        end
    end

    return OffsetArray(arch_array(arch, stencil), -1)
end

@inline dagger(ψ)    = (ψ[2], ψ[3], ψ[1])
@inline star(ψ₁, ψ₂) = (ψ₁ .* dagger(ψ₂) .+ dagger(ψ₁) .* ψ₂)

num_prod(i, m, l, r, xr, xi, shift, op, order, args...)            = prod(xr[i+shift] - xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::FirstDerivative)  = 2*xr[i+shift] - sum(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))
num_prod(i, m, l, r, xr, xi, shift, op, order, ::SecondDerivative) = 2

function num_prod(i, m, l, r, xr, xi, shift, op, order, ::Primitive) 
    s = sum(xi[op(i, r-q+1)]  for q=0:order if (q != m && q != l))
    p = prod(xi[op(i, r-q+1)] for q=0:order if (q != m && q != l))

    return xr[i+shift]^3 / 3 - sum * xr[i+shift]^2 / 2 + prod * xr[i+shift]
end

# Coefficients for (order-1)/2 finite-volume polynomial reconstruction.
function stencil_coefficients(i, r, xr, xi; shift = 0, op = Base.:(-), order = 3, der = nothing)
    coeffs = zeros(order)
    for j in 0:order-1
        for m in j+1:order
            numerator   = sum(num_prod(i, m, l, r, xr, xi, shift, op, order, der) for l=0:order if l != m)
            denominator = prod(xi[op(i, r-m+1)] - xi[op(i, r-l+1)] for l=0:order if l != m)
            coeffs[j+1] += numerator / denominator * (xi[op(i, r-j)] - xi[op(i, r-j+1)])
        end
    end

    return tuple(coeffs...)
end