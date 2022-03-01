#####
##### Operators of the form A * q where A is an area and q is some quantity.
#####

for metric in (:Δ, :A), dir in (:x, :y, :z), LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    
    operator   = Symbol(metric, dir, :_q, LX, LY, LZ)
    area       = Symbol(metric, dir, LX, LY, LZ)

    @eval begin
        $operator(i, j, k, grid, q) = $area(i, j, k, grid) * q[i, j, k]
        $operator(i, j, k, grid, f::Function, args...) = $area(i, j, k, grid) * f(i, j, k, grid, args...)
    end   
end
