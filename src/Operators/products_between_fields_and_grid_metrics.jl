#####
##### Operators of the form grid_metric * q
#####

for metric in (:Δ, :A), dir in (:x, :y, :z), LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    
    operator    = Symbol(metric, dir, :_q, LX, LY, LZ)
    grid_metric = Symbol(metric, dir, LX, LY, LZ)

    @eval begin
        $operator(i, j, k, grid, q) = $grid_metric(i, j, k, grid) * q[i, j, k]
        $operator(i, j, k, grid, f::F, args...) where F<:Function = $grid_metric(i, j, k, grid) * f(i, j, k, grid, args...)
    end   
end
