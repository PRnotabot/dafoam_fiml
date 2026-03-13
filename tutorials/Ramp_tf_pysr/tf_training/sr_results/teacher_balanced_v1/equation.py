"""Auto-generated symbolic distillation equation."""

import numpy as np

def delta_beta(PoD, VoS, PSoSS, KoU2):
    return KoU2*0.026193405

def beta_fiomega(PoD, VoS, PSoSS, KoU2):
    return 1.0 + delta_beta(PoD, VoS, PSoSS, KoU2)
