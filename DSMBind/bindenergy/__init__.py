"""DSMBind package.

The upstream repository did not include this file even though its applications
import ``bindenergy`` as a package.  Keep imports explicit in new code so that
optional drug/antibody dependencies are not loaded for TCR-pMHC training.
"""

