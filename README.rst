.. image:: https://github.com/jschueller/otklsens/actions/workflows/build.yml/badge.svg?branch=master
    :target: https://github.com/jschueller/otklsens/actions/workflows/build.yml

otklsens
========

Run tests::

    PYTHONPATH=$PWD pytest-3 test/ -s

Notes
-----
- enumerateFunction with weights/variance
- with Field->Point, PCE output dim is dimension of the output
  but with Field->Field, PCD output dim is the dimension of KL output modes! -> optimize LSM.solve(Matrix)
