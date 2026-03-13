"""Auto-generated sign-aware gated symbolic distillation equation."""

import numpy as np

def gate_raw(PoD, VoS, PSoSS, KoU2):
    return np.tanh((PoD - 1*0.47087985)*106.54328)

def gate_active(PoD, VoS, PSoSS, KoU2):
    return np.clip(0.5 * (1.0 + gate_raw(PoD, VoS, PSoSS, KoU2)), 0.0, 1.0)

def sign_active(PoD, VoS, PSoSS, KoU2):
    return np.clip(PoD - (-KoU2 + PSoSS) + np.tanh(8.600054 + PoD*(-15.715495)), -1.0, 1.0)

def amplitude_active(PoD, VoS, PSoSS, KoU2):
    return np.maximum(0.0, (0.045928564 - 0.04675482*PSoSS)*(KoU2 + PoD))

def delta_beta(PoD, VoS, PSoSS, KoU2):
    return gate_active(PoD, VoS, PSoSS, KoU2) * sign_active(PoD, VoS, PSoSS, KoU2) * amplitude_active(PoD, VoS, PSoSS, KoU2)

def beta_fiomega(PoD, VoS, PSoSS, KoU2):
    return 1.0 + delta_beta(PoD, VoS, PSoSS, KoU2)
