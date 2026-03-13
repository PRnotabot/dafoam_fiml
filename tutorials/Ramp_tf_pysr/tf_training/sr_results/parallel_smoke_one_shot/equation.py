"""Auto-generated symbolic distillation equation."""

import numpy as np

def delta_beta(PoD, VoS, PSoSS, KoU2):
    return PoD*VoS*(-0.16967559)

def beta_fiomega(PoD, VoS, PSoSS, KoU2):
    return 1.0 + delta_beta(PoD, VoS, PSoSS, KoU2)
