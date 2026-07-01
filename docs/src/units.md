# Units

`Oceananigans.Units` provides `Float64` constants for expressing physical quantities
as plain numeric products — no special types, just multiplication to SI units:

```@meta
DocTestSetup = quote
    using Oceananigans.Units
end
```

```jldoctest
julia> using Oceananigans.Units

julia> 5minutes
300.0

julia> 10days
864000.0

julia> 500kilometers
500000.0
```

Anywhere Oceananigans expects a number in SI units, you can write expressions like
`Δt = 5minutes` or `stop_time = 10days` for readability.

## Available units

| Quantity   | Constants                         | Value (SI)  |
|------------|-----------------------------------|-------------|
| Time       | `second`, `seconds`               | 1 s         |
| Time       | `minute`, `minutes`               | 60 s        |
| Time       | `hour`, `hours`                   | 3600 s      |
| Time       | `day`, `days`                     | 86400 s     |
| Length     | `meter`, `meters`                 | 1 m         |
| Length     | `kilometer`, `kilometers`         | 1000 m      |
| File size  | `KiB`, `MiB`, `GiB`, `TiB`       | powers of 1024 bytes |

Singular and plural forms are identical (`1day == 1days`).
The file-size constants are useful with the `max_filesize` keyword of output writers.
