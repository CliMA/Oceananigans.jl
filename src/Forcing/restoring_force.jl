@inline zerofunction(args...) = 0
@inline onefunction(args...) = 1

"""
    struct RestoringFunction{R, M, T}

Callable object for restoring fields to a `target` at
some `rate` and within a `mask`ed region in `x, y, z`. 
"""
struct RestoringFunction{R, M, T}
      rate :: R
      mask :: M
    target :: T
end

"""
    RestoringForce(; rate, mask=onefunction, target=zerofunction)

Returns a `SimpleForcing` that restores a field to `target(x, y, z, t)`
at the specified `rate`, in the region `mask(x, y, z)`.

Example
=======

* Restore a field to a linear z-gradient everywhere on a timescale of one minute.

```julia
julia> restore_stratification = RestoringForce(; rate = 1/60, target = (x, y, z, t) -> z)
```
"""
RestoringForce(; rate, mask=onefunction, target=zerofunction) =
    SimpleForcing(RestoringFunction(rate, mask, target); multiplicative=true)

@inline (f::RestoringFunction)(x, y, z, t, field) =
    f.rate * f.mask(x, y, z) * (f.target(x, y, z, t) - field)

#####
##### Sponge layer functions
#####

"""
    GaussianMask{D}(center, width)

Callable object that returns a Gaussian masking function centered on
`center`, with `width`, and varying along direction `D`.

Examples
========

* Create a Gaussian mask centered on `z=0` with width `1` meter.

```julia
julia> mask = GaussianMask{:z}(0, 1)
```
"""
struct GaussianMask{D, T}
    center :: T
     width :: T

    function GaussianMask{D}(center, width)
        T = promote_type(typeof(center), typeof(width))
        return new{D, T}(center, width)
    end
end    

@inline (g::GaussianMask{:x})(x, y, z) = exp(-(x - g.center)^2 / (2 * g.width^2))
@inline (g::GaussianMask{:y})(x, y, z) = exp(-(y - g.center)^2 / (2 * g.width^2))
@inline (g::GaussianMask{:z})(x, y, z) = exp(-(z - g.center)^2 / (2 * g.width^2))

#####
##### Linear target functions
#####

"""
    LinearTarget{D}(intercept, gradient)

Callable object that returns a Linear target function
with `intercept` and `gradient`, and varying along direction `D`.

Examples
========

* Create a linear target function varying in `z`, equal to `0` at 
  `z=0` and with gradient 10⁻⁶:

```julia
julia> target = LinearTarget{:z}(0, 1e-6)
```
"""
struct LinearTarget{D, T}
    intercept :: T
     gradient :: T

    function LinearTarget{D}(intercept, gradient)
        T = promote_type(typeof(gradient), typeof(intercept))
        return new{D, T}(intercept, gradient)
    end
end

@inline (p::LinearTarget{:x})(x, y, z, t) = p.intercept + p.gradient * x
@inline (p::LinearTarget{:y})(x, y, z, t) = p.intercept + p.gradient * y
@inline (p::LinearTarget{:z})(x, y, z, t) = p.intercept + p.gradient * z
