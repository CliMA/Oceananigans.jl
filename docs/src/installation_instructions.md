# Getting started

This section covers the basics of installing Julia and Oceananigans, and then describes best practices when running
Oceananigans. If you're new to Julia, we also recommend reading [Julia's "getting started" page](https://julialang.org/learning/getting-started/).


## Installation instructions

Usually the hardest part of installing Oceananigans is installing Julia. You can first check if your
system already has Julia installed. If that's not the case, installation is necessary and you can
[download](https://julialang.org/downloads/) Julia from its official website, where also you can
find [platform-specific installation instructions](https://julialang.org/downloads/platform/).
Alternatively, you can also manually [compile Julia from source code](https://github.com/JuliaLang/julia#building-julia).

With Julia installed, you can install Oceananigans with a single line of code within Julia itself: `] add Oceananigans`. 
However, best practices for Julia suggest a couple extra commands to ensure that each project has
a separate and easily reproducible environment (we suggest that users read [this page](https://pkgdocs.julialang.org/v1/environments/)
to get a better understanding of working with environments). Best practices when starting a new Oceananigans 
project is to 

1. Start with a empty directory (in this example called `new_project`).
2. Inside the empty directory, start Julia with the command `julia --project`.
3. Access the built-in package manager by pressing `]` in the Julia command prompt.
4. Add the Oceananigans package with `add Oceananigans`.
5. Instantiate/build all dependencies `instantiate`.

To do so, open a Julia REPL from the terminal and then start the package manager by hitting `]`. Then:

```julia
(@v1.7) pkg> activate .
  Activating new environment at `~/new_project/Project.toml`

(new_project) pkg> add Oceananigans
(new_project) pkg> instantiate
```

We recommend installing Oceananigans with this way (using the built-in Julia package manager), because this installs a stable, tagged
release. Oceananigans.jl can be updated to the latest tagged release from the package manager by typing

```julia
(new_project) pkg> update Oceananigans
```

At this time, updating should be done with care, as Oceananigans is under rapid development and breaking 
changes to the user API occur often. But if anything does happen, please open [an issue on github](https://github.com/CliMA/Oceananigans.jl/issues)!
We're more than happy to help with getting your simulations up and running.

!!! compat "Julia 1.7 or newer"
    The latest version of Oceananigans requires at least Julia v1.7 to run.
    Installing Oceananigans with an older version of Julia will install an older version of Oceananigans (the latest version compatible with your version of Julia).

## Running Oceananigans

Whenever you run Oceananigans for a project, it is recommended that you activate the project's environment first.
This ensures that you will always use the package versions of you project and that any changes there will not affect
other projects.. To activate the project's environment you either start Julia using `julia --project` or from Julia's
package manager you call:

```julia
(@v1.7) pkg> activate .
  Activating new project at `~/new_project`

(new_project) pkg>
```

For initial explorations and getting used to the code, you can enter the commands directly in the
[Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) (which can be started with the command
`julia --project`). The REPL is an interactive command-line tool (similar to Python's IPython) that
has a searchable history, tab-completion, helpful keybindings, and many other features to help users
interact with packages. If you already have a script, you can call `julia --project my_oceananigans_script.jl`
from the terminal. This will execute all the commands in that script and subsequently close Julia.
This is equivalent to opening the Julia REPL inside your project directory and typing each of those
commands by hand (or simply calling `include("my_oceananigans_script.jl")` in the REPL).
