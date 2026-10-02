//
//                  ***GRADE_v1.00***
//  MyFunctions.hpp
//
//  Created by Farbod Mahmoudinobar and Cristiano L. Dias on 12/6/18.
//  Copyright © 2018 Farbod Mahmoudinobar and Cristiano L. Dias.
//
//  This file is part of GRADE.
//
//  GRADE is a free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  GRADE is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with GRADE.  If not, see <https://www.gnu.org/licenses/>.


#ifndef MyFunctions_hpp
#define MyFunctions_hpp

#include <stdio.h>
#include <vector>
#include <string>

using namespace std;

// Define the Ring struct and its type
enum RingType { SQUARE, PENTAGON, HEXAGON };

struct Ring {
    int index;  // Index of the ring
    RingType type;  // Type of ring (Square, Pentagon, Hexagon)
};

// Function declarations
void calc_Distance(int count_solvent, int count_solute, vector<vector<int>>& neigh_list, vector<vector<double>>& atom_Pos, double& boxX, double& boxY, double& boxZ, vector<int>& Nneigh, int Natoms, int topSolute, string time, double HBOND_DIST );

void ring_Finder(int count_solute, int Natoms, vector<int>& Nneigh, vector<vector<int>>& My_neigh, vector<vector<int>>& ring4_temp, vector<vector<int>>& ring5_temp, vector<vector<int>>& ring6_temp, int topSolute, int count_solvent, vector<vector<double>>& atom_Pos, double boxX, double boxY, double boxZ, double HBOND_DIST, double delta_s, double delta_p, double delta_h);

void find_shared_edges_ring5(int count_ring5, vector<vector<int>>& ring5, vector<vector<int>>& My_neigh_ring5, vector<unsigned long int>& N_ring5_neigh);

void find_shared_edges_ring6(int count_ring6, vector<vector<int>>& ring6, vector<vector<int>>& My_neigh_ring6, vector<unsigned long int>& N_ring6_neigh);

void find_shared_edges_ring6_ring5(int count_ring6, int count_ring5, vector<vector<int>>& ring5, vector<vector<int>>& ring6, vector<vector<int>>& My_neigh_ring6_ring5, vector<unsigned long int>& N_ring6_ring5_neigh );

void find_shared_edges_ring5_ring6(int count_ring5, int count_ring6, vector<vector<int>>& ring6, vector<vector<int>>& ring5, vector<vector<int>>& My_neigh_ring5_ring6, vector<unsigned long int>& N_ring5_ring6_neigh );

void find_shared_edges_ring4_ring5(int count_ring4, int count_ring5, vector<vector<int>>& ring4, vector<vector<int>>& ring5, vector<vector<int>>& My_neigh_ring4_ring5, vector<unsigned long int>& N_ring4_ring5_neigh );
    
void find_shared_edges_ring6_ring4(int count_ring6, int count_ring4, vector<vector<int>>& ring6, vector<vector<int>>& ring4, vector<vector<int>>& My_neigh_ring6_ring4, vector<unsigned long int>& N_ring6_ring4_neigh );

bool  compare(vector<int>& ringA, vector<int>& ringB, int ringA_N, int ringB_N );

bool  compare_adjacant(vector<int>& ringA, vector<int>& ringB, int ringA_N, int ringB_N , vector<int>& base);

int remove_duplicates_map(vector<vector<Ring>>& cup512); //TODO idk why this is defined for cup512

int remove_duplicates_map_cage(vector<vector<Ring>>& cups);

int remove_duplicates_map_cage_rings(vector<vector<int>>& cups);

int remove_duplicates_map_rings(vector<vector<int>>& cups);

void coplanar_Points(vector<vector<int>>& ring , vector<vector<double>>& atoms, string time, vector<vector<int>>& ring_New, int THETA);

void cup_512_Finder(vector<vector<int>>& ring5,int count_ring5, vector<unsigned long int>& N_ring5_neigh, vector<vector<int>>& My_neigh_ring5, vector<vector<Ring>>& cup512 );

void cup_62512_Finder(vector<vector<int>>& ring6, int count_ring6, vector<vector<int>>& My_neigh_ring6_ring5, vector<unsigned long int> N_ring6_ring5_neigh, vector<vector<int>>& ring5, vector<unsigned long int> N_ring5_neigh, vector<vector<int>>& My_neigh_ring5, vector<vector<Ring>>& cup62512);

void cup_415462_Finder(vector<vector<int>>& ring6, int count_ring6, vector<vector<int>>& My_neigh_ring6_ring5, vector<vector<int>>& My_neigh_ring5_ring6, vector<vector<int>>& My_neigh_ring6_ring4,vector<vector<int>>& My_neigh_ring4_ring5, vector<unsigned long int> N_ring6_ring5_neigh, vector<unsigned long int> N_ring5_ring6_neigh, vector<unsigned long int> N_ring6_ring4_neigh, vector<unsigned long int> N_ring4_ring5_neigh, vector<vector<int>>& ring5, vector<unsigned long int> N_ring5_neigh, vector<vector<int>>& My_neigh_ring5, vector<vector<int>>& ring4, int count_ring4, vector<vector<int>>& My_neigh_ring4, vector<vector<Ring>>& cup62512);

int cage_Finder(std::string cagetype, vector<vector<Ring>> cups, unsigned long count_cups, vector<vector<int>>& neighour_rings, vector<vector<int>>& cage, vector<vector<int>>& cage_rings, string time);

int cage_Finder_different_cups(std::string cagetype, vector<vector<Ring>> cups1, unsigned long count_cups1, vector<vector<Ring>> cups2, unsigned long count_cups2, vector<vector<int>> ring5, vector<vector<int>>& My_neigh_ring5,  vector<vector<int>>& My_neigh_ring6, vector<vector<int>>& My_neigh_ring6_ring5, vector<vector<int>>& cage, vector<vector<Ring>>& cage_rings, string time );

// void print_vmd_cage_frings(std::string cagetype, int cage_count, vector<vector<Ring>> cage_rings, vector<vector<int>> ring4, vector<vector<int>> ring5, vector<vector<int>> ring6, vector<vector<double>> atom_Pos, string time, string rawFilename , string box_size_xyz, vector<vector<double>> solutes, size_t & meth_counter, string solute1, int topSolute, string solute2, int count_solute2, int frameCounter);

double calc_F4(int count_solvent, int count_solute, vector<vector<int>>& My_neigh, vector<vector<double>>& atom_Pos, double& boxX, double& boxY, double& boxZ, vector<int>& Nneigh, int Natoms, int topSolute, string time, double HBOND_DIST );

int cage_Finder_64512(vector<vector<Ring>> cup62512, int count_62512_cups, vector<vector<int>>& cage_64512_rings);

void print_vmd_cage_frings(std::string cagetype, int cage_count, vector<vector<Ring>> cage_rings, vector<vector<int>> ring4, vector<vector<int>> ring5, vector<vector<int>> ring6, vector<vector<double>> atom_Pos, string time, string rawFilename , string box_size_xyz, vector<vector<double>> solutes, size_t & meth_counter, string solute1, int topSolute, string solute2, int count_solute2, int frameCounter);

void print_data(std::ofstream& frameDataFile, int frameCounter, int cage_512_count, int cage_62512_count, int cage_64512_count, double F4_value);

void print_usage();

#endif /* MyFunctions_hpp */
