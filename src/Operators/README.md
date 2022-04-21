## Differentiation and interpolation operators

The geometry of the staggerd grid used by Oceananigans.jl (sometimes called the "C grid")
is (in one dimension) shown below
```
face   cell   face   cell   face

        i-1            i
         ↓             ↓
  |      ×      |      ×      |
  ↑             ↑             ↑
 i-1            i            i+1
```
Difference operators are denoted by a `δ` (`\delta`). Calculating the difference
of a cell-centered quantity `c` at cell `i` returns the difference at face `i`
```
δcᵢ = cᵢ - cᵢ₋₁
```
and so this operation, if applied along the x-dimension, is denoted by `δxᶠᵃᵃ`.

The difference of a face-centered quantity `u` at face `i` returns the difference at cell `i`
```
δuᵢ = uᵢ₊₁ - uᵢ
```
and is thus denoted `δxᶜᵃᵃ` when applied along the x-dimension.

The three characters at the end of the function name, `faa` for example, indicates that the
output lies on the cell faces in the x-dimension but remains at their original positions in 
the y- and z-dimensions. Thus we further identify this operator by the superscript `ᶠᵃᵃ`, where
the `a` stands for "any" as the location is unchanged by the operator and is determined by
the input.

As a result, the interpolation of a quantity `c` from a cell `i` to face `i` (which is denoted
"`ℑxᶠᵃᵃ`" in the code below) is
```
ℑxᶠᵃᵃ(c)ᵢ = (cᵢ + cᵢ₋₁) / 2
```
Conversely, the interpolation of a quantity `u` from a face `i` to cell `i` is given by
```
ℑxᶠᵃᵃ(u)ᵢ = (uᵢ₊₁ + uᵢ) / 2
```
The `ℑ` (`\Im`) symbol indicates that an interpolation is being performed. For example, `ℑx`
indicates that the interpolation is performed along the x-dimension.
