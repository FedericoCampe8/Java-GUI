/*********************************************************************
 * Authors: Ferdinando Fioretto (ffiorett@cs.nmsu.edu)               *
 *          Federico Campeotto (campe8@cs.nmsu.edu)                  *
 *          Alessandro Dal Palu', Enrico Pontelli, Agostino Dovier   *
 * (C) Copyright 2010-2011                                           *
 *                                                                   *
 * This file is part of FIASCO.                                      *
 *                                                                   *
 * FIASCO is free software; you can redistribute it and/or           *
 * modify it under the terms of the GNU General Public License       *
 * as published by the Free Software Foundation;                     *
 *                                                                   *
 * FIASCO is distributed WITHOUT ANY WARRANTY; without even the      *
 * implied  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR  *
 * PURPOSE. See the GNU General Public License for more details.     *
 *                                                                   *
 * You should have received a copy of the GNU General Public License *
 * along with this program; if not, see http://www.gnu.org/licenses. *
 *                                                                   *
 *********************************************************************/
#ifndef FIASCO_ENERGY__
#define FIASCO_ENERGY__

#include "typedefs.h"
#include "globals.h"

real get_energy ( ss_type* secondary_s_info,
                  real* h_distances, real* h_angles,
                  real * contact_params, aminoacid * aa_seq,
                  real * tors, real * tors_corr,
                  real hydrogen_w, real contact_w, real correlation_w,
                  int bb_start, int bb_end, int n_res, int scope_start, int scope_end,
                  int n_bytes, int n_blocks, int n_threads );

void hydrogen_energy ( real * structure, real * h_values,
                       real * h_distances, real * h_angles,
                       ss_type* secondary_s_info,
                       int bb_start, int bb_end, int n_res, int threadIdx );

void contact_energy ( real * structure, real * con_values,
                      real * contact_params, aminoacid * aa_seq,
                      int bb_start, int bb_end, int n_res, int threadIdx );

void contact_energy_cg ( real * structure, real * contact_params, aminoacid * aa_seq,
                         int first_cg_idx, int second_cg_idx, real* c_energy );

void correlation_energy ( real * structure, real * corr_val,
                          real * tors, real * tors_corr, aminoacid * aa_seq,
                          int bb_start, int bb_end, int n_res, int v_id=0 );

int get_h_distance_bin ( real distance );
int get_h_angle_bin    ( real angle );
int get_corr_aa_type   ( aminoacid );

#endif
