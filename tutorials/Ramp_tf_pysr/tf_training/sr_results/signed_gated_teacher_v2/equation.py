"""Auto-generated sign-aware gated symbolic distillation equation."""

import numpy as np

def gate_raw(PoD, VoS, PSoSS, KoU2):
    return np.tanh((PoD - 1*0.4413447)*11.853484)

def gate_active(PoD, VoS, PSoSS, KoU2):
    return np.clip(0.5 * (1.0 + gate_raw(PoD, VoS, PSoSS, KoU2)), 0.0, 1.0)

def sign_active(PoD, VoS, PSoSS, KoU2):
    return np.clip(KoU2 + 1.4019966 + np.tanh(-2.9110258*KoU2)/(KoU2 + 0.037450533), -1.0, 1.0)

def amplitude_active(PoD, VoS, PSoSS, KoU2):
    return np.maximum(0.0, (PSoSS - 1*1.0799632)*(-0.041860998))

def delta_beta(PoD, VoS, PSoSS, KoU2):
    return gate_active(PoD, VoS, PSoSS, KoU2) * sign_active(PoD, VoS, PSoSS, KoU2) * amplitude_active(PoD, VoS, PSoSS, KoU2)

def beta_fiomega(PoD, VoS, PSoSS, KoU2):
    return 1.0 + delta_beta(PoD, VoS, PSoSS, KoU2)
