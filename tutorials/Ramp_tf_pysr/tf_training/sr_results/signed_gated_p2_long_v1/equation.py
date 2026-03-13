"""Auto-generated sign-aware gated symbolic distillation equation."""

import numpy as np

def gate_raw(PoD, VoS, PSoSS, KoU2):
    return np.tanh((PoD - 0.45792148)*26.6618)

def gate_active(PoD, VoS, PSoSS, KoU2):
    return np.clip(0.5 * (1.0 + gate_raw(PoD, VoS, PSoSS, KoU2)), 0.0, 1.0)

def sign_active(PoD, VoS, PSoSS, KoU2):
    return np.clip(KoU2 - np.tanh(KoU2*29.206163) + 0.93770653, -1.0, 1.0)

def amplitude_active(PoD, VoS, PSoSS, KoU2):
    return np.maximum(0.0, 0.04555343 + PSoSS*(-0.042618006))

def delta_beta(PoD, VoS, PSoSS, KoU2):
    return gate_active(PoD, VoS, PSoSS, KoU2) * sign_active(PoD, VoS, PSoSS, KoU2) * amplitude_active(PoD, VoS, PSoSS, KoU2)

def beta_fiomega(PoD, VoS, PSoSS, KoU2):
    return 1.0 + delta_beta(PoD, VoS, PSoSS, KoU2)
