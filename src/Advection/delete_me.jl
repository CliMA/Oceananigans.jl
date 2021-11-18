
using OffsetArrays

@inline dagger(ψ)    = (ψ[2:3]..., ψ[1])
@inline star(ψ₁, ψ₂) = (ψ₁ .* dagger(ψ₂) .+ dagger(ψ₁) .* ψ₂)

# Integral of ENO coefficients for 2nd order polynomial reconstruction at the face
function prim_interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        prod = 1
                        sum  = 0 
                        for q = 0:3
                            if q != m && q != l 
                                prod *= coord[op(i, r-q+1)]
                                sum  += coord[op(i, r-q+1)]
                            end
                        end
                        num += coord[i+bias]^3 / 3 - sum * coord[i+bias]^2 / 2 + prod * coord[i+bias]
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i,r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i,r-j)] - coord[op(i,r-j+1)]))
    end

    return coeff
end

# Second derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der2_interp_weights(r, coord, i, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        num += 2 
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i,r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i,r-j)] - coord[op(i,r-j+1)]))
    end

    return coeff
end

# first derivative of ENO coefficients for 2nd order polynomial reconstruction at the face
function der1_interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        sum = 0
                        for q = 0:3
                            if q != m && q != l 
                                sum += coord[op(i, r-q+1)]
                            end
                        end
                        num += 2 * coord[i+bias] - sum
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i,r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i,r-j)] - coord[op(i,r-j+1)]))
    end

    return coeff
end

# ENO coefficients for 2nd order polynomial reconstruction at the face
function interp_weights(r, coord, i, bias, op)

    coeff = ()
    for j = 0:2
        c = 0
        @inbounds begin
            for m = j+1:3
                num = 0
                for l = 0:3
                    if l != m
                        prod = 1
                        for q = 0:3
                            if q != m && q != l 
                                prod *= (coord[i+bias] - coord[op(i, r-q+1)])
                            end
                        end
                        num += prod
                    end
                end
                den = 1
                for l = 0:3
                    if l!= m
                        den *= (coord[op(i, r-m+1)] - coord[op(i,r-l+1)])
                    end
                end
                c += num / den
            end 
        end
        coeff = (coeff..., c * (coord[op(i,r-j)] - coord[op(i,r-j+1)]))
    end

    return coeff
end


N = 3
cpu_coord = [-4, -3, -2, -1, 0, 2, 4, 5, 6, 7, 8] .* 1.0
# cpu_coord = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6] .* 1.0
cpu_coord = OffsetArray(cpu_coord, -4)

FT = Float64
# written all on overleaf

global allstencils = ()
for op in (-, +)
    for r = 0:2

        stencil = NTuple{6, FT}[]   
        @inbounds begin
            for i = 0:N+1
            
                bias1 = Int(op == +)
                bias2 = Int(op == +) - 1

                Δcᵢ = cpu_coord[i + bias1] - cpu_coord[i + bias2]
                
                Bᵢ  = prim_interp_weights(r, cpu_coord, i, bias1, op)
                bᵢ  =      interp_weights(r, cpu_coord, i, bias1, op)
                bₓᵢ = der1_interp_weights(r, cpu_coord, i, bias1, op)
                Aᵢ  = prim_interp_weights(r, cpu_coord, i, bias2, op)
                aᵢ  =      interp_weights(r, cpu_coord, i, bias2, op)
                aₓᵢ = der1_interp_weights(r, cpu_coord, i, bias2, op)

                pₓₓ = der2_interp_weights(r, cpu_coord, i, op)
                Pᵢ  =  (Bᵢ .- Aᵢ)

                wᵢᵢ = Δcᵢ  .* (bᵢ .* bₓᵢ .- aᵢ .* aₓᵢ .- pₓₓ .* Pᵢ)  .+ Δcᵢ^4 .* (pₓₓ .* pₓₓ)
                wᵢⱼ = Δcᵢ  .* (star(bᵢ, bₓᵢ)  .- star(aᵢ, aₓᵢ) .- star(pₓₓ, Pᵢ)) .+
                                                        Δcᵢ^4 .* star(pₓₓ, pₓₓ)

                push!(stencil, (wᵢᵢ..., wᵢⱼ...))
            end
        end

        stencil = OffsetArray(stencil, -1)

        global allstencils = (allstencils..., stencil)
    end
end

b0(x, y, z) = FT(13/12) .* (x, -2y , z).^2 .+ FT(1/4) .* (3x,  - 4y,  +  z).^2
b1(x, y, z) = FT(13/12) .* (x, -2y , z).^2 .+ FT(1/4) .* ( x,    0y,  -  z).^2
b2(x, y, z) = FT(13/12) .* (x, -2y , z).^2 .+ FT(1/4) .* ( x,  - 4y,  + 3z).^2

b0s(x, y, z) = FT(13/12) .* star((x, -2y , z), (x, -2y , z)) .+ FT(1/4) .* star((3x,  - 4y,  +  z), (3x,  - 4y,  +  z))
b1s(x, y, z) = FT(13/12) .* star((x, -2y , z), (x, -2y , z)) .+ FT(1/4) .* star(( x,    0y,  -  z), ( x,    0y,  -  z))
b2s(x, y, z) = FT(13/12) .* star((x, -2y , z), (x, -2y , z)) .+ FT(1/4) .* star(( x,  - 4y,  + 3z), ( x,  - 4y,  + 3z))

real_stencil_0 = (b0(1, 1, 1)..., b0s(1, 1, 1)...)
real_stencil_1 = (b1(1, 1, 1)..., b1s(1, 1, 1)...)
real_stencil_2 = (b2(1, 1, 1)..., b2s(1, 1, 1)...)

real_stencil_left = [real_stencil_0, real_stencil_1, real_stencil_2]
real_stencil_right = reverse(real_stencil_left)

my_stencil_left  = [adv.smooth_xᶠᵃᵃ[1][2], adv.smooth_xᶠᵃᵃ[2][2], adv.smooth_xᶠᵃᵃ[3][2]]
my_stencil_right = [adv.smooth_xᶠᵃᵃ[6][2], adv.smooth_xᶠᵃᵃ[5][2], adv.smooth_xᶠᵃᵃ[4][2]]

