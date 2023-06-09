Prototype to compute distinguishing Hennessy-Milner formulas. For real world application one better uses the mCRL2 tool [ltscompare](https://www.mcrl2.org/web/user_manual/tools/release/ltscompare.html)

usage: python distinguish.py [-h] [-c] [-v] infile1 infile2

Compute a distinguishing HML formula

positional arguments:

  infile1           Input LTS 1, input should be Aldebaran format (.aut)

  infile2           Input LTS 2, input should be Aldebaran format (.aut)

options:

  -h, --help        show this help message and exit

  -c, --cleaveland

  -v, --verbose
