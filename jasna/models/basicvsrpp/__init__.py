# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

def register_all_modules():
    from jasna.models.basicvsrpp.mmagic import register_all_modules
    register_all_modules()
    from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet, BasicVSRPlusPlusGan