"""Auto-generated symbolic distillation equation."""

import numpy as np

def delta_beta(PoD, VoS, PSoSS, KoU2):
    return (KoU2 - (KoU2 + KoU2))*(-0.009034034)

def beta_fiomega(PoD, VoS, PSoSS, KoU2):
    return 1.0 + delta_beta(PoD, VoS, PSoSS, KoU2)
