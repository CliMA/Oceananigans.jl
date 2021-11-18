## actual weights on right_β₀ == left_β₂


wᵢᵢ = (13/12 + 1/4, 4*13/12 + 4, 9/4 + 13/12)  

wᵢⱼ = (-4*13/12 - 2, -4*13/12 - 6, 2*13/12 + 6/4)

@inline dagger(ψ)    = (ψ[2:3]..., ψ[1])
@inline star(ψ₁, ψ₂) = (ψ₁ .* dagger(ψ₂) .+ dagger(ψ₁) .* ψ₂)

b0(x, y, z) = FT(13/12) .* (x, -2y , z).^2 .+ FT(1/4) .* (3x,  - 4y,  +  z).^2
b1(x, y, z) = FT(13/12) .* (x, -2y , z).^2 .+ FT(1/4) .* ( x,    0y,  -  z).^2
b2(x, y, z) = FT(13/12) .* (x, -2y , z).^2 .+ FT(1/4) .* ( x,  - 4y,  + 3z).^2

b0s(x, y, z) = FT(13/12) .* star((x, -2y , z), (x, -2y , z)) .+ FT(1/4) .* star((3x,  - 4y,  +  z), (3x,  - 4y,  +  z))
b1s(x, y, z) = FT(13/12) .* star((x, -2y , z), (x, -2y , z)) .+ FT(1/4) .* star(( x,    0y,  -  z), ( x,    0y,  -  z))
b2s(x, y, z) = FT(13/12) .* star((x, -2y , z), (x, -2y , z)) .+ FT(1/4) .* star(( x,  - 4y,  + 3z), ( x,  - 4y,  + 3z))

real_stencil_0 = (b0(1, 1, 1)..., b0s(1, 1, 1)...)
real_stencil_1 = (b1(1, 1, 1)..., b1s(1, 1, 1)...)
real_stencil_2 = (b2(1, 1, 1)..., b2s(1, 1, 1)...)

real_stencil = [real_stencil_0, real_stencil_1, real_stencil_2]


my_stencil = [allstencils[1][1], allstencils[2][1], allstencils[3][1], allstencils[4][1]]

