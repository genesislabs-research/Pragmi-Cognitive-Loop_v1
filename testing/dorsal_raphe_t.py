"""
dorsal_raphe_t.py
Loop Stage: Dorsal Raphe Nucleus Serotonin Source

BIOLOGICAL GROUNDING
The dorsal raphe nucleus is the primary midbrain source of forebrain
serotonin (5-HT). Its functional role is multifaceted: serotonin
encodes patience for delayed reward (Miyazaki et al. 2014), modulates
risk and aversion (Cools, Roberts, Robbins 2008), and provides a
slow opponent signal to dopamine in reinforcement learning (Daw,
Kakade, Dayan 2002).

The Daw et al. (2002) opponent-process account is most directly
implementable: where dopamine signals positive prediction error,
serotonin signals an inverse, scaled by an aversion-sensitivity
parameter. The two systems together produce balanced exploitation
of positive and negative outcomes, with depression understood as a
serotonin deficit producing impaired punishment learning.

Cohen, Amoroso, Uchida (2015) document that DRN 5-HT neurons fire
in response to both rewards and punishments, supporting a salience-
based account in addition to the opponent-process account. Boureau
and Dayan (2011) review the broader integration with motivation
and decision-making.

This file implements a DRN module emitting:
1. Tonic 5-HT level (slow drift toward an aversion-sensitivity scaled
   running average of negative TD errors)
2. Phasic 5-HT (instantaneous opponent of dopamine RPE)
3. Patience scalar suitable for biasing temporal discounting toward
   longer horizons when 5-HT is high

Primary grounding papers:

Daw ND, Kakade S, Dayan P (2002). "Opponent interactions between
serotonin and dopamine." Neural Networks, 15(4-6), 603-616.
DOI: 10.1016/S0893-6080(02)00052-7

Miyazaki KW, Miyazaki K, Tanaka KF, Yamanaka A, Takahashi A,
Tabuchi S, Doya K (2014). "Optogenetic activation of dorsal raphe
serotonin neurons enhances patience for future rewards." Current
Biology, 24(17), 2033-2040. DOI: 10.1016/j.cub.2014.07.041

Cohen JY, Amoroso MW, Uchida N (2015). "Serotonergic neurons signal
reward and punishment on multiple timescales." eLife, 4, e06346.
DOI: 10.7554/eLife.06346

Cools R, Roberts AC, Robbins TW (2008). "Serotoninergic regulation
of emotional and behavioural control processes." Trends in Cognitive
Sciences, 12(1), 31-40. DOI: 10.1016/j.tics.2007.10.011

Boureau YL, Dayan P (2011). "Opponency revisited: competition and
cooperation between dopamine and serotonin." Neuropsychopharmacology,
36(1), 74-97. DOI: 10.1038/npp.2010.151

Genesis Labs Research
Authored for the PRAGMI loop assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DorsalRapheConfig:
    """Configuration for the DRN serotonin module.

    Master flag is first per Genesis Labs standard. NOT a biological
    quantity.
    """

    enable_dorsal_raphe: bool = True
    enable_tonic_5ht: bool = True
    enable_phasic_5ht: bool = True
    enable_patience_signal: bool = True

    # Aversion sensitivity. Daw et al. (2002) parameter. NOT a
    # biological quantity in the strict sense.
    aversion_sensitivity: float = 1.0

    # Tonic decay constant. NOT a biological quantity.
    tonic_decay: float = 0.95

    # Mapping from tonic 5-HT to discount factor. Higher 5-HT means
    # longer patience, modeled as adding to the gamma in TD. NOT a
    # biological quantity, training artifact.
    patience_scale: float = 0.05


class DorsalRaphe(nn.Module):
    """Dorsal raphe nucleus serotonin source.

    BIOLOGICAL STRUCTURE: Dorsal raphe nucleus, a midbrain
    serotonergic nucleus projecting diffusely to forebrain.

    BIOLOGICAL FUNCTION: Computes opponent signal to dopamine for
    aversion learning, plus a tonic patience signal that biases
    temporal discounting toward longer horizons when 5-HT is high.

    Daw ND, Kakade S, Dayan P (2002).
    DOI: 10.1016/S0893-6080(02)00052-7
    Miyazaki KW et al. (2014). DOI: 10.1016/j.cub.2014.07.041
    Cohen JY, Amoroso MW, Uchida N (2015). DOI: 10.7554/eLife.06346
    Cools R, Roberts AC, Robbins TW (2008).
    DOI: 10.1016/j.tics.2007.10.011
    Boureau YL, Dayan P (2011). DOI: 10.1038/npp.2010.151

    ANATOMICAL INTERFACE (input):
        Sending structures: VTA dopamine TD error, ACC conflict,
        amygdala valence.
        Receiving structure: DRN (this module).
        Connection: convergent projections onto DRN core.

    ANATOMICAL INTERFACE (output):
        Sending structure: DRN serotonergic neurons.
        Receiving structures: forebrain broadly, including PFC, VTA,
        BG.
        Connection: serotonergic projections.
    """

    def __init__(self, cfg: Optional[DorsalRapheConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DorsalRapheConfig()
        # Persistent tonic 5-HT level.
        self.register_buffer("tonic_level", torch.tensor(0.0))

    def forward(
        self,
        dopamine: Optional[torch.Tensor] = None,
        aversion: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute tonic 5-HT, phasic 5-HT, and patience.

        Args:
            dopamine: optional scalar TD error from VTA.
            aversion: optional scalar aversion signal from amygdala.

        Returns:
            tonic_5ht: scalar tonic 5-HT level.
            phasic_5ht: scalar phasic 5-HT (opponent of dopamine).
            patience: scalar patience-bias for temporal discounting.
        """
        if not self.cfg.enable_dorsal_raphe:
            zero = torch.tensor(0.0)
            return zero, zero.clone(), zero.clone()

        d = dopamine if dopamine is not None else torch.tensor(0.0)
        a = aversion if aversion is not None else torch.tensor(0.0)

        # Phasic 5-HT: opponent of dopamine, scaled by aversion sensitivity.
        if self.cfg.enable_phasic_5ht:
            phasic = self.cfg.aversion_sensitivity * (-d + a)
        else:
            phasic = torch.tensor(0.0)

        # Tonic 5-HT: slow leaky integration of phasic.
        if self.cfg.enable_tonic_5ht:
            with torch.no_grad():
                self.tonic_level.copy_(
                    self.cfg.tonic_decay * self.tonic_level
                    + (1.0 - self.cfg.tonic_decay) * phasic.detach()
                )
            tonic = self.tonic_level.clone()
        else:
            tonic = torch.tensor(0.0)

        # Patience signal: monotone in tonic level.
        if self.cfg.enable_patience_signal:
            patience = self.cfg.patience_scale * tonic
        else:
            patience = torch.tensor(0.0)

        return tonic, phasic, patience

    def reset(self) -> None:
        """Reset tonic level."""
        with torch.no_grad():
            self.tonic_level.zero_()
