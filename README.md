
# Experiments in exploring CSC 2.

Explore a web service that provides access to a **small** fraction
of the CSC 2.0 database. This is **purely** an experiment, and so
is not guaranteed to be performant (there will be long page downloads),
be correct (so be careful with any data or image you see, and do not
assume that all data is being shown), be secure (you aren't putting
in any passwords, but there has been no attempt to avoid information
exfiltration), and it could delete or fill up your disk.

## Installation

This should be possible with a `ciao-install` installed version of
CIAO, but it has only been tested with a `conda` installed version.
I stringly suggest creating a separate environment, **just in case**:

```
% conda create -n=cscexplore -c https://cxc.cfa.harvard.edu/conda/ciao -c conda-forge ciao sherpa ciao-contrib
% conda activate cscexplore
```

If you have downloaded the code (e.g. with `git` or via
https://gitlab.com/dburke/csc-explore) you can then install
the viewer (if you are in the directory containing this file
and `setup.py`) with

```
% pip install .
```

After which, move to a different directory and then set up the
database (which only needs to be done once):

```
% export FLASK_APP=explore
% flask init-db
Cleaning up: /home/braveuser/anaconda/envs/cscexplore/var/explore-instance
Initialized the database.
```

To run the service:

```
% flask run
* Serving Flask app "explore" (lazy loading)
* Environment: development
* Debug mode: on
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
* Restarting with stat
* Debugger is active!
* Debugger PIN: 233-389-275
```

At this point you can go to http://127.0.0.1:5000/ with your
favourite web browser.

## Things you can do

At the moment all you can do is

 - search around a location (given by name, not RA,Dec) using a radius of 1 arcminutes
 - view previous searches

When a search has been completed you can view the results of that
search as a list of sources: each source can then be viewed, which
will display some basic source properties and the individual
observations which make up the source (broken down by stack). These
individual observations can be viewed, which will show some basic
data:

 - the counts, exposure map, and PSF, along with PSF and source and background regions
 - the lightcurve of the source and the background
 - the spectrum

## Things that are known to break

An incomplete list

 - repeated searches of the same location are not recognised as
   already being done
 - HRC observations
 - extended sources (the `2CXO J...` sources that end in `X`)
 - if you stop a search (e.g. by exiting the `flask run` command)
   there's no guarantee it will work well

## Things I'd like to add or improve

We should be able to use https://js9.si.edu/ to view the image data
interactively.

There should be some ability to fit models to the spectal data.

More data can be included (e.g. the spectral properties).

This is a very old-school app, and it could do with some attempt to
provide a more-meaningful user experience.

## Note

This is being done on a "best effort" basis, and at the moment the time
I have for this is rather limited.
