# Newton ADMM • [![Build Status][travis-image]][travis] [![License][license-image]][license]

*An implementation of the Newton ADMM algorithm for solving cone problems.*

[travis-image]: https://travis-ci.org/locuslab/newton_admm.png?branch=master
[travis]: http://travis-ci.org/locuslab/newton_admm

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

---

## What is Newton ADMM? 

Newton ADMM is a general ADMM algorithm described in our paper 
[here](https://arxiv.org/abs/1705.00772) 
which uses a semi-smooth Newton's method to minimize the ADMM residuals. 

## What is in this repository? 

1. This repository contains an implementation of Newton ADMM for the 
[SCS](https://github.com/cvxgrp/scs) ADMM iterations. In essence, this is a 
second order ADMM solver for generic cone programming. The signature for the
solver is almost identical to that of SCS, differing only in optional keyword
arguments, and so the two can be used interchangably.  

2. Examples of running Newton ADMM on various cone problems can be found in the
examples directory.

3. This repository also contains implementations of cone projections and their
respective Jacobians. See `newton_admm/cones.py`. 

## Todo

+ Add RPCA cone example
+ Add specialized solver examples

# Issues and Contributions

+ [file an issue](https://github.com/locuslab/newton_admm/issues)
+ [send in a PR](https://github.com/locuslab/newton_admm/pulls).

# Licensing

This repository is
[Apache-licensed](https://github.com/locuslab/newton_admm/blob/master/LICENSE).
