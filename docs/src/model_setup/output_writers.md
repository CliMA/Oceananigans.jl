# Output writers

`AbstractOutputWriter`s save data to disk.
`Oceananigans` provides three ways to write output:

1. `NetCDFOutputWriter` for output of arrays and scalars that uses [NCDatasets.jl](https://github.com/Alexander-Barth/NCDatasets.jl)
2. `JLD2OutputWriter` for arbitrary julia data structures that uses [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)
3. `Checkpointer` that automatically saves as much model data as possible, using [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)

The `Checkpointer` is discussed on a separate documentation page.

## Basic usage

`NetCDFOutputWriter` and `JLD2OutputWriter` require four inputs:

1. The `model` from which output data is sourced (required to initialize the `OutputWriter`).
2. A key-value pairing of output "names" and "output" objects. `JLD2OutputWriter` accepts `NamedTuple`s and `Dict`s;
   `NetCDFOutputWriter` accepts `Dict`s with string-valued keys. Output objects are either `AbstractField`s or
   functions that return data when called via `func(model)`.
3. A `schedule` on which output is written. `TimeInterval`, `IterationInterval`, `WallTimeInterval` schedule
   periodic output according to the simulation time, simulation interval, or "wall time" (the physical time
   according to a clock on your wall). A fourth `schedule` called `AveragedTimeInterval` specifies
   periodic output that is time-averaged over a `window` prior to being written.
4. The filename and directory. Currently `NetCDFOutputWriter` accepts one `filepath` argument, while
   `JLD2OutputWriter` accepts a filename `prefix` and `dir`ectory.

Other important keyword arguments are

* `field_slicer::FieldSlicer` for outputting subregions, two- and one-dimensional slices of fields.
  By default a `FieldSlicer` is used to remove halo regions from fields so that only the physical
  portion of model data is saved to disk.

* `array_type` for specifying the type of the array that holds outputted field data. The default is
  `Array{Float32}`, or arrays of single-precision floating point numbers.

Once an `OutputWriter` is created, it can be used to write output by adding it the
ordered dictionary `simulation.output_writers`. prior to calling `run!(simulation)`.

More specific detail about the `NetCDFOutputWriter` and `JLD2OutputWriter` is given below.

## NetCDF output writer

Model data can be saved to NetCDF files along with associated metadata. The NetCDF output writer is generally used by
passing it a dictionary of (label, field) pairs and any indices for slicing if you don't want to save the full 3D field.

```@docs
NetCDFOutputWriter
```

## JLD2 output writer

JLD2 is a fast HDF5 compatible file format written in pure Julia.
JLD2 files can be opened in julia and in Python with the [h5py](https://www.h5py.org/)
package.

The `JLD2OutputWriter` receives either a `Dict`ionary or `NamedTuple` containing
`name, output` pairs. The `name` can be a symbol or string. The `output` must either be
an `AbstractField` or a function called with `func(model)` that returns arbitrary output.
Whenever output needs to be written, the functions will be called and the output
of the function will be saved to the JLD2 file.

```@docs
JLD2OutputWriter
```

## Time-averaged output

Time-averaged output is specified by setting the `schedule` keyword argument for either `NetCDFOutputWriter` or
`JLD2OutputWriter` to `AveragedTimeInterval`:

```@docs
AveragedTimeInterval
```
