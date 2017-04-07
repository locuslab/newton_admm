# Newton ADMM • [![Build Status][travis-image]][travis] [![License][license-image]][license]

*An implementation of the Newton ADMM algorithm for solving cone problems.*

[travis-image]: https://travis-ci.com/locuslab/newton_admm.png?branch=master
[travis]: http://travis-ci.com/locuslab/newton_admm

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

---

## What is Newton ADMM? 

Newton ADMM is a general ADMM algorithm described in our paper [here]() which 
uses a semi-smooth Newton's method to minimize the ADMM residuals. 

## What is in this repository? 

This repository contains an implementation of Newton ADMM for the 
[SCS](https://github.com/cvxgrp/scs) ADMM iterations. In essence, this is a 
second order ADMM solver for generic cone programming. The signature for the
solver is almost identical to that of SCS, and so the two can be used 
interchangably.  

## Todo

+ Add examples of solvers for specific ADMM problems. 

# Issues and Contributions

+ [file an issue](https://github.com/locuslab/newton_admm/issues)
+ [send in a PR](https://github.com/locuslab/newton_admm/pulls).

# Licensing

This repository is
[Apache-licensed](https://github.com/locuslab/newton_admm/blob/master/LICENSE).
