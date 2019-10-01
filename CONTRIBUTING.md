# Contributors Guide

Thank you for considering contributing to Oceananigans! This short guide will
give you ideas on how you can contribute and help you make a contribution.

Please feel free to ask us questions and chat with us at any time if you're
unsure about anything.

## What can I do?

* Tackle an existing issue. We keep a list of [good first issues](https://github.com/climate-machine/Oceananigans.jl/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  that are self-contained and suitable for a newcomer to try and work on.

* Try to run Oceananigans and play around with it to simulate your favorite
  fluids and ocean physics. If you run into any problems or find it difficult
  to use or understand, please open an issue!

* Write up an example or tutorial on how to do something useful with
  Oceananigans, like how to set up a new physical configuration.

* Improve documentation or comments if you found something hard to use.

* Implement a new feature if you need it to use Oceananigans.

If you're interested in working on something, let us know by commenting on
existing issues or by opening a new issue if. This is to make sure no one else
is working on the same issue and so we can help and guide you in case there
is anything you need to know beforehand.

## Ground Rules

* Each pull request should consist of a logical collection of changes. You can
  include multiple bug fixes in a single pull request, but they should be related.
  For unrelated changes, please submit multiple pull requests.
* Do not commit changes to files that are irrelevant to your feature or bugfix
  (eg: .gitignore).
* Be willing to accept criticism and work on improving your code; we don't want
  to break other users' code, so care must be taken not to introduce bugs. We
  discuss pull requests and keep working on them until we believe we've done a
  good job.
* Be aware that the pull request review process is not immediate, and is
  generally proportional to the size of the pull request.

## Reporting a bug

The easiest way to get involved is to report issues you encounter when using
Oceananigans or by requesting something you think is missing.

* Head over to the [issues](https://github.com/climate-machine/Oceananigans.jl/issues) page.
* Search to see if your issue already exists or has even been solved previously.
* If you indeed have a new issue or request, click the "New Issue" button.
* Please be as specific as possible. Include the version of the code you were using, as
  well as what operating system you are running. The output of Julia's `versioninfo()`
  and `] status` is helpful to include. If possible, include complete, minimal example
  code that reproduces the problem.

## Setting up your development environment

* Install [Julia](https://julialang.org/) on your system.
* Install git on your system if it is not already there (install XCode command line tools on
  a Mac or git bash on Windows).
* Login to your GitHub account and make a fork of the
  [Oceananigans repository](https://github.com/climate-machine/Oceananigans.jl) by
  clicking the "Fork" button.
* Clone your fork of the Oceananigans repository (in terminal on Mac/Linux or git shell/
  GUI on Windows) in the location you'd like to keep it.
  ``git clone https://github.com/your-user-name/Oceananigans.jl.git``
* Navigate to that folder in the terminal or in Anaconda Prompt if you're on Windows.
* Connect your repository to the upstream (main project).
  ``git remote add oceananigans https://github.com/climate-machine/Oceananigans.jl.git``
* Create the development environment by opening Julia via ``julia --project`` then
  typing in ``] instantiate``. This will install all the dependencies in the ``Project.toml``
  file.
* You can test to make sure Oceananigans works by typing in ``] test``.

Your development environment is now ready!

## Pull Requests

Changes and contributions should be made via GitHub pull requests against the ``master`` branch.

When you're done making changes, commit the changes you made. Chris Beams has
written a [guide](https://chris.beams.io/posts/git-commit/) on how to write
good commit messages.

When you think your changes are ready to be merged into the main repository,
push to your fork and [submit a pull request](https://github.com/climate-machine/Oceananigans.jl/compare/).

**Working on your first Pull Request?** You can learn how from this _free_ video series
[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github), Aaron Meurer's [tutorial on the git workflow](https://www.asmeurer.com/git-workflow/), or the guide [“How to Contribute to Open Source"](https://opensource.guide/how-to-contribute/).

## Documentation

Now that you've made your awesome contribution, it's time to tell the world how to use it.
Writing documentation strings is really important to make sure others use your functionality
properly. Didn't write new functions? That's fine, but be sure that the documentation for
the code you touched is still in great shape. It is not uncommon to find some strange wording
or clarification that you can take care of while you are here.

## Credits

This contributor's guide is heavily based on the excellent [MetPy contributor's guide](https://github.com/Unidata/MetPy/blob/master/CONTRIBUTING.md).
