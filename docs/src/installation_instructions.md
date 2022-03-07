# Getting started

## Installation instructions

The hardest part of installing Oceananigans is actually installing Julia (and even that should be pretty easy). 
If you want to use Oceananigans on a laptop you can generally install Julia easily via the package manager of whatever
operational system you use (some quick googling should tell you how). If you're running on a cluster, you should check
if it already has Julia installed (most systems these days do). If that's the case, then the hardest part is already done!

If those options above are not possible, one alternative is to simply download the Julia pre-compiled binaries from the
[Julia website](https://julialang.org/downloads/), where you can simply run the Julia executable that's contained inside
(this should work out of the box!). Another option is to compile Julia manually from source code. This is definitely the hardest
option, and you can find instructions on how to do that [here](https://github.com/JuliaLang/julia#building-julia).

With Julia installed, you can install Oceananigans with a single line of code within Julia itself: `] add Oceananigans`. 
However, best practices for Julia suggest a couple extra commands to ensure that each project has
a separate and easily reproducible environment (we suggest that users read [this page](https://pkgdocs.julialang.org/v1/environments/)
to get a better understanding of working with environments). Best practices when starting a new Oceananigans 
project is to 

1. Start with a empty directory (in this example called `new_project`)
2. Inside the empty directory, start Julia with the command `julia --project`
3. Access the built-in package manager by pressing `]` in the Julia command prompt
4. Add the Oceananigans package with `add Oceananigans`
5. Instantiate/build all dependencies `instantiate`

This process looks similar to this on a terminal (note that there may be many messages between some of these commands):

```
user@system:~$ julia --project
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.6.3 (2021-09-23)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

(@v1.6) pkg> activate .
  Activating new environment at `~/Dropbox/tests/new_project/Project.toml`

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

!!! compat "Julia 1.6 or newer"
    The latest version of Oceananigans requires at least Julia v1.6 to run.
    Installing Oceananigans with an older version of Julia will install an older version of Oceananigans (the latest version compatible with your version of Julia).

## Running Oceananigans

Whenever you run Oceananigans for a project, it's recommended that you start Julia not with the simple command `julia`,
but with the command `julia --project`. This ensures that you will always use the package versions of you project
and that any changes there will not affect other projects. That said, running Oceananigans can be done a couple of ways. 


For initial explorations and getting used to the code, you can
enter the commands directly in the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) (which can be started with
the command `julia --project`). This is an interactive
command-line tool (similar to Python's IPython) that has a searchable history, tab-completion, helpful keybindings, and many
other features to help users interact with packages. If you already have a file with commands, you can issue `julia --project my_oceananigans_script.jl`
from the terminal, which will run all the commands in that script and subsequently close Julia. This is equivalent to opening
the Julia REPL inside your project directory and typing each of those commands by hand (or simply issuing
`include("my_oceananigans_script.jl")` in the REPL), with the exception that the REPL will remain open after the commands in this case.

If you are using a HPC system and you want to make use of a GPU, then an extra step generally has to be added: submitting the
job to request GPU resources. This is usually done with the `slurm` or `PBS` job schedulers and the details usually change depending
on which cluster you are using, but the process should be similar (if not the same) to submitting other jobs in that same cluster. We
recommend you check the cluster documentation for information on how to submit jobs that require GPU resources.
