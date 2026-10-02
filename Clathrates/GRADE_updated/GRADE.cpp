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


#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <unistd.h>


#include "MyFunctions.hpp"

using namespace std;


// Main function ----------------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    double const Version=1.00 ;
    
    //output program information.
    cout << std::fixed ;
    cout << std::setprecision(3);
    cout << "\n\n\t\t** GRADE - VERSION " << Version << " **\n\n" ;
    
    string inputFilename, s1, s2 ;
    string outputFilename, rawFilename;
    int DT=1, FR=1, THETA=45;
    int in_theta=0, in_fr=0, in_dt=0, in_r=0, in_F4=0, in_d1=0, in_d2=0, in_s1=0, in_s2=0;         //This parameter is for whether theta is given as input parameter (1) or taken as default value (0).
    double HBOND_DIST = 0.35;   //Command line input parameter for Hbond_distance cutoff, taken from '-r' flag.
    double delta_p = 0.18, delta_h = 0.26, delta_s = 0.3;//18;   // find delta s by varying it and checking for cage count
    // Command line input parameter for delta constraints for pentagon and hexagons.
    string F4 = "no";
    
    //Parsing command line parameters (input taken from "-i" and output taken from "-o")
    std::string arg;
    std::string inName, outName;
    
    
    
    //-------------------------------------------------------------------------------------
    
    int argi=1;
    
    while ( argi < argc )
    {
        const char* args = argv[ argi++ ] ;
        
        switch ( *args )
        {
            case '-':
                if( strcmp(args, "-i") == 0 )
                {
                    inputFilename = argv[argi];
                }
                else if (strcmp(args, "-o") == 0 )
                {
                    outputFilename = argv[argi];
                }
                else if (strcmp(args, "-dt") == 0)
                {
                    DT = atoi(argv[argi]);
                    in_dt = 1;
                }
                else if (strcmp(args, "-fr") == 0)
                {
                    FR = atoi(argv[argi]);
                    in_fr = 1 ;
                }
                else if (strcmp(args, "-theta") == 0)
                {
                    THETA = atoi(argv[argi]);
                    in_theta=1;
                }
                else if (strcmp(args, "-r") == 0)
                {
                    HBOND_DIST = atof(argv[argi]);
                    in_r = 1 ;
                }
                else if (strcmp(args, "-f4") == 0 || strcmp(args, "-F4") == 0)
                {
                    in_F4 = 1;
                    F4 = "YES" ;
                    if(argi < argc )F4 =argv[argi];
                    if(F4 == "no" || F4 == "No" || F4 == "NO") {in_F4=0; F4 = "NO";}
                }
                else if (strcmp(args, "-d1") == 0)
                {
                    delta_p = atof(argv[argi]);
                    in_d1 = 1 ;
                }
                else if (strcmp(args, "-d2") == 0)
                {
                    delta_h = atof(argv[argi]);
                    in_d2 = 1 ;
                }
                
                else if (strcmp(args, "-s1") == 0)
                {
                    s1 = argv[argi];
                    in_s1 = 1 ;
                }
                else if (strcmp(args, "-s2") == 0)
                {
                    s2 = argv[argi];
                    in_s2 = 1 ;
                }
                
                else cout << "Skipped unknown option(s): '" << args << " " <<argv[argi] << "'" << "\n\n";
                break;
                
        }
    }
    if(inputFilename.empty())
    {
        cout << "**No Input Provided!**" << "\n\n";
        print_usage();
        cout << "**No Input Provided!**" << "\n\n";
        return 0 ;
    }

    else{print_usage();}
    
    if (outputFilename.empty())
    {
        // size_t pos = inputFilename.find_last_of("/\\");
    
        // // Extract the file name after the last slash
        // std::string fileName = inputFilename.substr(pos + 1);
        outputFilename = inputFilename;
    }
    
    cout << "Command line:\n" ;
    cout << argv[0] << " -i " << inputFilename << " -o " << outputFilename ;
    if (in_dt == 1)  {cout << " -dt " << DT ;}
    if (in_fr == 1)  {cout << " -fr " << FR ;}
    if (in_theta==1) {cout << " -theta " << THETA;}
    if (in_r == 1)   {cout << " -r " << HBOND_DIST ;}
    if (in_F4 == 1)  {cout << " -f4 " << "YES" ;}
    if (in_d1 == 1)  {cout << " -d1 " << delta_p ;}
    if (in_d2 == 1)  {cout << " -d2 " << delta_h ;}
    if (in_s1 == 1)  {cout << " -s1 " << s1 ;}
    if (in_s2 == 1)  {cout << " -s1 " << s2 ;}
    
    cout << "\n\n" ;
    //-------------------------------------------------------------------------------------
    
    vector<vector<int>> My_neigh ;
    int Natoms, count_solute=0 , count_solute2=0;
    vector<int> Nneigh;
    vector<vector<double>> atom_Pos;
    double F4_value=0;                                  //Value of F4 order parameter for each frame.
    vector<double> current_F4_line;
    vector<vector<double>> time_vs_F4;                 //This variable holds Time/frame_counter in first column and value of F4 order parameter in second column.
    
    
    
    string line="NONE", str1, str2, str3;
    int int1;
    double x , y , z ;
    double boxX = 0.0, boxY = 0.0, boxZ = 0.0;
    vector<double> temp_vect;
    vector<int> temp_vect2;
    int lineNumber=0;
    int firstSOL=0;
    int frameCounter=0;
    int topSolute =0;
    size_t methane_512 = 0, methane_62512 = 0, methane_64512 = 0, methane_4151062;
    string time;
    Natoms=0;
    string solute1="AAA";
    string solute2="BBB";

    size_t found_ext = outputFilename.find_last_of(".");
    rawFilename = outputFilename.substr(0,found_ext);
    outputFilename = rawFilename + "_grade" + ".txt";

    std::cout << rawFilename << endl;
    

    remove(outputFilename.c_str());
    
    ofstream outFile;


    string temp1, temp2, solute1_atom_finder, solute2_atom_finder;
    temp1 = rawFilename + "_cage62512.gro" ;
    temp2 = rawFilename + "_cage512.gro" ;
    remove(temp1.c_str());
    remove(temp2.c_str());


    
    
    //Create the header for outputfile.
    outFile.open(outputFilename, ofstream::app);
    // outFile << "#|(Frame) Time(ps)\t\t|cage\t|filled_cage\t|cage\t|filled_cage\t|cage\t|filled_cage\t|" << endl ;
    // outFile << "#|\t\t\t\t|5¹²\t|5¹²\t\t|6²5¹²\t|6²5¹²\t\t|6⁴5¹²\t|6⁴5¹²\t\t|" << endl ;

    outFile << std::left    
                            << std::setw(15) << "Time (ps)"
                            << std::setw(20) << "Frame"
                            << std::setw(15) << "512_Cages"
                            << std::setw(15) << "62512_Cages"
                            << std::setw(15) << "64512_Cages"
                            << std::setw(15) << "4151062_Cages"
                            << std::setw(10) << "F4"
                            << "\n";


    ifstream fileIN;
    fileIN.open(inputFilename);
    
    //Error Check
    if(fileIN.fail()){
        cerr << "Error Reading File" << "\n";
        exit (1);}
    
    ofstream outFile_F4;
    if(in_F4 == 1)                          //If F4 flag option is on, open a file for F4 as a function of time.
    {
        remove("F4.xvg");       //Remove any existing F4.xvg file and create a new one. 
        outFile_F4.open("F4.xvg", ofstream::app);
        outFile_F4 << "# --------------------------------------- \n" ;
        outFile_F4 << "#|Frame\t|F4\t\t|Time(ps)\t|" << endl;
        outFile_F4 << "# --------------------------------------- \n" ;

    }
    
    //Start reading the input file.
    while (!fileIN.eof())
    {
        getline(fileIN, line);
        lineNumber++;
        //Following if statement makes sure the "Natoms" always has the correct number, even if the code skips the first frame due to FR being greater than 1. (Added in v1.18)
        if(lineNumber == 2 || (lineNumber == (2 + 3*frameCounter + frameCounter*Natoms) && !fileIN.eof()))
        {
            istringstream streamA(line);
            
            streamA >> Natoms ;
        }
        size_t found =0;
        
        size_t found_time=0;
        if( line.find("t=") ) {found_time = line.find("t=");}
        if (lineNumber == 1 || (lineNumber == (1 + 3*frameCounter + frameCounter*Natoms) && !fileIN.eof()) )        //This is to find the first line of gro file.
        {
            frameCounter++;
            if(( (frameCounter-1) % FR) != 0 ) continue;        //If frameCounter-1 is not a multiple of FR, continue to next frame.
                                                                //"-1" is to ignore the frame with t=0.000. This ensures that FR=20 reads frames with t=0,20,40,... .
           if (found_time != string::npos) {
	    time = line.substr(found_time + 3);
	    size_t stepPos = time.find("step=");

	    // Extract the first number (before "step=")
	    std::string firstNumberStr = time.substr(0, stepPos);

	    // Convert the time string to double
	    double time_print = std::stod(firstNumberStr);  // Convert to double

	    // Default frame value
	    long frame = 0;
	    bool valid_frame = false;

	    // Check if "step=" is present
	    if (stepPos != string::npos) {
		// Extract the second number (after "step=")
		std::string secondNumberStr = time.substr(stepPos + 5);  // 5 is the length of "step="

		// Convert the frame number to long
		try {
		    frame = std::stol(secondNumberStr);
		    valid_frame = true;  // Mark frame as valid
		} catch (const std::invalid_argument& e) {
		    std::cerr << "Warning: Invalid frame number: " << secondNumberStr << ". Using default frame value 0." << std::endl;
		} catch (const std::out_of_range& e) {
		    std::cerr << "Warning: Frame number out of range: " << secondNumberStr << ". Using default frame value 0." << std::endl;
		}
	    } else {
		std::cerr << "Warning: 'step=' not found in frame " << frameCounter << ". Using default frame value 0." << std::endl;
	    }

	    // Write to the output file
	    std::cout << time << endl;
	    outFile << std::setw(15) << time_print;  // Time (ps) column
	    std::cout << "valid frame: " << valid_frame << endl;
	    if (valid_frame) {
		outFile << std::setw(20) << frame;  // Frame column
	    } else {
		outFile << std::setw(20) << 0;  // Write default frame value 0
	    }
	} else {
	    // If "t=" is not found, write the frame counter as the frame number
	    outFile << frameCounter << "\t\t\t| ";
	} 
            cout << " frame#: " << frameCounter << ", "  ;
            if(found_time != string::npos) cout << line.substr(found_time) << " ps\n" ;
            else cout << "\n" ;
            
            if(found == string::npos)cout << line << endl;
            
            getline(fileIN, line);      //Read 2nd line (number of atoms)
            lineNumber++;
            
            
            istringstream streamA(line);
            
            streamA >> Natoms ;
            
            vector<int> temp_vec = {0,0,0};
            
            int  count_solvent=0;
            vector<vector<double>> solutes;
            count_solute=0;
            temp_vect={0,0,0};
            atom_Pos.clear();
            atom_Pos.push_back(temp_vect);
            
            
            while (atom_Pos.size() <= Natoms && atom_Pos.size() <= 9999)
            {
                
                temp_vect.clear();
                
                getline(fileIN, line);
                lineNumber++;
                
                
                istringstream streamA(line);
                
                streamA >> str1 >> str2 >> int1 >> x >> y >> z ;
                
                temp_vect.push_back(x);
                temp_vect.push_back(y);
                temp_vect.push_back(z);
                
                
                atom_Pos.push_back(temp_vect);
                
                if(line.find("SOL") != string::npos)                        //If line includes "SOL", then add to number of count_solvent.          Else, add to number of count_solute.
                {
                    count_solvent++;
                    if(count_solvent == 1 ){firstSOL = lineNumber;}
                }
                else
                {
                    if(count_solute == 0)
                    {
                        solute1 = line.substr(5, 7);                        //Get the name of first solute in system.
                        size_t found_space = solute1.find_first_of(" ");
                        solute1 = solute1.substr(0, found_space);
                    }
                    
                    if(line.substr(5,7) != solute1)
                    {
                        
                        solute2 = line.substr(5,7);                         //Get the name of second solute in system.
                        size_t found_space = solute2.find_first_of(" ");
                        solute2 = solute2.substr(0, found_space);
                        
                        count_solute2++;
                        
                    }
                    count_solute++;
                    solutes.push_back(temp_vect);
                }
                
            }
            
            while (atom_Pos.size() <= Natoms && atom_Pos.size() > 9999)
            {
                
                
                temp_vect.clear();
                
                getline(fileIN, line);
                lineNumber++;
                
                istringstream streamA(line);
                
                streamA >> str1 >> str2 >> x >> y >> z ;
                
                temp_vect.push_back(x);
                temp_vect.push_back(y);
                temp_vect.push_back(z);
                
                
                atom_Pos.push_back(temp_vect);
                
                if(line.find("SOL") != string::npos)        //If line includes "SOL"
                {
                    count_solvent++;
                }
                else
                {
                    count_solute++;
                    solutes.push_back(temp_vect);
                    
                    if(line.substr(5,7) != solute1)
                    {
                        solute2 = line.substr(5,7);
                        size_t found_space = solute2.find_first_of(" ");
                        solute2 = solute2.substr(0, found_space);
                        count_solute2++;
                    }
                    
                }
                
            }
            
            getline(fileIN, line);      //get the box size from last line of frame
            //End of reading each frame.
            
            string box_size_xyz;
            box_size_xyz = line;
            lineNumber++;
            
            istringstream streamB(line);
            streamB >> boxX >> boxY >> boxZ ;
            
            if(frameCounter == 1 )
            {
                topSolute = firstSOL - 3;
                if(in_s1 == 0){cout << "solute1: " << solute1 << " " << topSolute << " atoms \n" ;}
                else {cout << "solute1: " << s1 << " " << topSolute << " atoms " ;}
                //if( (in_s1==0 && in_s2==1) || (in_s1==1 && in_s2==1) || (in_s1==0 && in_s2==0))
                {
                    
                    if(strcmp(solute1.c_str(), solute2.c_str()) != 0)
                    {
                       if(in_s2==0) cout << ", solute2: " << solute2 << " " << count_solute2-topSolute <<" atoms\n";
                        else cout << ", solute2: " << s2 << " " << count_solute2-topSolute <<" atoms\n";
                    }
                    else cout << "\n" ;
                    
                }
                
            }

            //Start of calculations for each frame

            calc_Distance(count_solvent, count_solute, My_neigh, atom_Pos, boxX, boxY, boxZ, Nneigh, Natoms, topSolute, time, HBOND_DIST);
            // for ( int i = 0; i < Nneigh.size(); i++) {
            //     std::cout<< "Number of neighbours of index " << i << " is " << Nneigh[i] << endl;
            //     std::cout<< My_neigh[topSolute+4*i][0] << endl;
            // }
            vector<vector<int>> ring4, ring5, ring6, ring4_temp, ring5_temp, ring6_temp;
            vector<vector<int>> My_neigh_ring4, My_neigh_ring6, My_neigh_ring5, My_neigh_ring6_ring5, My_neigh_ring5_ring6, My_neigh_ring4_ring5, My_neigh_ring6_ring4;
            
            
            ring_Finder(count_solute, Natoms, Nneigh, My_neigh, ring4_temp, ring5_temp, ring6_temp, topSolute, count_solvent, atom_Pos, boxX, boxY, boxZ, HBOND_DIST, delta_s, delta_p, delta_h);

            if( ring4_temp.size() > 0 ) coplanar_Points(ring4_temp, atom_Pos, time, ring4,THETA);        //Find the 4-rings which form a plane and get rid of the rest.
            
            int count_ring4 = remove_duplicates_map_rings(ring4);     //Remove duplicate lines from 5-rings.

            cout << "ring[4]: " << count_ring4 << "\n";
            
            if( ring5_temp.size() > 0 ) coplanar_Points(ring5_temp, atom_Pos, time, ring5,THETA);        //Find the 5-rings which form a plane and get rid of the rest.
            
            int count_ring5 = remove_duplicates_map_rings(ring5);     //Remove duplicate lines from 5-rings.
            
            cout << "ring[5]: " << count_ring5 << "\n";
            
            if( ring6_temp.size() > 0 )coplanar_Points(ring6_temp, atom_Pos, time, ring6, THETA);            
            
            // std::cout<< "HEXAGON" << endl;

            int count_ring6 = remove_duplicates_map_rings(ring6);

            //  std::cout<<"size"<< ring6.size() << endl;

            // for (int i = 0; i < ring6.size(); i++) {
            //     std::cout<<"ring "<<i<<endl;
            //     for (int j = 0; j<ring6[i].size(); j++){
            //         //std::cout<<ring6[i][j]<<endl;
            //         std::cout<<ring6[i][j]/4<<endl;
            //     }
            // }
            
            cout << "ring[6]: " << count_ring6 << "\n\n";

            
            vector<unsigned long int> N_ring5_neigh;      //Vector of ring-neighbours of all rings.
            
            
            find_shared_edges_ring5(count_ring5, ring5, My_neigh_ring5, N_ring5_neigh);

            
            vector<unsigned long int> N_ring6_neigh;
            
            vector<unsigned long int> N_ring6_ring5_neigh, N_ring5_ring6_neigh, N_ring4_ring5_neigh, N_ring6_ring4_neigh;


            find_shared_edges_ring6_ring5(count_ring6, count_ring5, ring5, ring6, My_neigh_ring6_ring5, N_ring6_ring5_neigh );

            find_shared_edges_ring5_ring6(count_ring5, count_ring6, ring6, ring5, My_neigh_ring5_ring6, N_ring5_ring6_neigh );

            find_shared_edges_ring4_ring5(count_ring4, count_ring5, ring4, ring5, My_neigh_ring4_ring5, N_ring4_ring5_neigh);
            find_shared_edges_ring6_ring4(count_ring6, count_ring4, ring6, ring4, My_neigh_ring6_ring4, N_ring6_ring4_neigh);
            

            if (count_ring4 == 0 && count_ring5 == 0 && count_ring6 == 0 )
                {cout << "\n**NO RINGS FOUND!**\n\n" ;                
            }

            // std::ofstream outfile("ring_indices.txt");

            // outfile << "RINGS\n" << std::endl;
            // outfile << "PENTAGONS" << std::endl;

            // // Writing pentagons (ring5)
            // for (int i = 0; i < ring5.size(); i++) {
            //     outfile << "ring idx " << i << std::endl;
            //     for (int j = 0; j < ring5[i].size(); j++) {
            //         outfile << ring5[i][j] / 4 << std::endl;  // Assuming the division by 4 is correct
            //     }
            // }

            // // Writing hexagons (you can continue as needed)
            // outfile << "HEXAGONS" << std::endl;

            // for (int i = 0; i < ring6.size(); i++) {
            //     outfile << "ring idx " << i << std::endl;
            //     for (int j = 0; j < ring6[i].size(); j++) {
            //         outfile << ring6[i][j] / 4 << std::endl;  // Assuming the division by 4 is correct
            //     }
            // }

            // // Close the file after writing
            // outfile.close();

            // std::ofstream outfile2("neighbours.txt");

            // outfile2 << "RING 5\n" << std::endl;

            // // Writing pentagons (ring5)
            // for (int i = 0; i < My_neigh_ring5.size(); i++) {
            //     outfile2 << "pentagon nr " << i << std::endl;
            //     outfile2 << "size " << My_neigh_ring5[i].size() << std::endl;
            //     for (int j = 0; j < My_neigh_ring5[i].size(); j++) {
            //         outfile2 << My_neigh_ring5[i][j] << std::endl;  // Assuming the division by 4 is correct
            //         for (int k =0; k<ring5[My_neigh_ring5[i][j]].size(); k++) {
            //             outfile2 <<"indices of this ring"<< ring5[My_neigh_ring5[i][j]][k]/4 << std::endl;

            //         }
            //     }
            // }

            // outfile2 << "RING 4 RING 5\n" << std::endl;

            // for (int i = 0; i < My_neigh_ring4_ring5.size(); i++) {
            //     outfile2 << "square nr " << i << std::endl;
            //     outfile2 << "size " << My_neigh_ring4_ring5[i].size() << std::endl;
            //     for (int j = 0; j < My_neigh_ring4_ring5[i].size(); j++) {
            //         outfile2 << My_neigh_ring4_ring5[i][j] << std::endl;  // Assuming the division by 4 is correct
            //         for (int k =0; k<ring5[My_neigh_ring4_ring5[i][j]].size(); k++) {
            //             outfile2 <<"indices of this ring"<< ring5[My_neigh_ring4_ring5[i][j]][k]/4 <<"\n\n"<< std::endl;

            //         }
            //     }
            // }

            // outfile2 << "RING 6 RING 5\n" << std::endl;

            // // Writing pentagons (ring5)
            // for (int i = 0; i < My_neigh_ring6_ring5.size(); i++) {
            //     outfile2 << "hexagon nr " << i << std::endl;
            //     outfile2 << "size " << My_neigh_ring6_ring5[i].size() << std::endl;
            //     for (int j = 0; j < My_neigh_ring6_ring5[i].size(); j++) {
            //         outfile2 << My_neigh_ring6_ring5[i][j] << std::endl;  // Assuming the division by 4 is correct
            //         for (int k =0; k<ring5[My_neigh_ring6_ring5[i][j]].size(); k++) {
            //             outfile2 <<"indices of this ring"<< ring5[My_neigh_ring6_ring5[i][j]][k]/4 << std::endl;

            //         }
            //     }
            // }

            // outfile2 << "RING 5 RING 6\n" << std::endl;

            // // Writing pentagons (ring5)
            // for (int i = 0; i < My_neigh_ring5_ring6.size(); i++) {
            //     outfile2 << "pentagon nr " << i << std::endl;
            //     outfile2 << "size " << My_neigh_ring5_ring6[i].size() << std::endl;
            //     for (int j = 0; j < My_neigh_ring5_ring6[i].size(); j++) {
            //         outfile2 << My_neigh_ring5_ring6[i][j] << std::endl;  // Assuming the division by 4 is correct
            //         for (int k =0; k<ring6[My_neigh_ring5_ring6[i][j]].size(); k++) {
            //             outfile2 <<"indices of this ring"<< ring6[My_neigh_ring5_ring6[i][j]][k]/4 << std::endl;

            //         }
            //     }
            // }

            // // Close the file after writing
            // outfile2.close();


            ///////// CUPS ///////////////////////////////////////////////////////////////
            
            vector<vector<Ring>> cup512;
            cup_512_Finder(ring5, count_ring5, N_ring5_neigh, My_neigh_ring5, cup512);
            int count_512_cups = 0;
            if(cup512.size() != 0){count_512_cups = remove_duplicates_map(cup512);}        //Remove duplicate lines from 512 cups.
            
            if (count_512_cups == 0){cout << "\n**NO 5⁶ CUPS FOUND**\n\n" ;}
            else {std::cout << "# 5⁶" << std::setw(10) << "\t\tcup: " << count_512_cups << "\n";}
            
            vector<vector<Ring>> cup62512;
            cup_62512_Finder(ring6, count_ring6, My_neigh_ring6_ring5, N_ring6_ring5_neigh, ring5, N_ring5_neigh, My_neigh_ring5, cup62512);
            int count_62512_cups=0;
            if(cup62512.size() != 0){count_62512_cups = remove_duplicates_map(cup62512);}
            
            if (count_62512_cups == 0 ){cout << "\n**NO 6¹5⁶ CUPS FOUND**\n\n" ;}
            else {std::cout << "# 6¹5⁶" << std::setw(8) << "\t\tcup: " << count_62512_cups << "\n";}

            //std::cout << "Size of My_neigh_ring6_ring5: " << My_neigh_ring6_ring5.size() << std::endl;

            vector<vector<Ring>> cup415462;
            cup_415462_Finder(ring6, count_ring6, My_neigh_ring6_ring5, My_neigh_ring5_ring6, My_neigh_ring6_ring4, My_neigh_ring4_ring5, N_ring6_ring5_neigh, N_ring5_ring6_neigh, N_ring6_ring4_neigh, N_ring4_ring5_neigh, ring5, N_ring5_neigh, My_neigh_ring5, ring4, count_ring4, My_neigh_ring4, cup415462);

            std::cout<< "4¹5⁴6² cups size before removing duplicates "<<cup415462.size()<<endl;


            int count_415462_cups = 0;
            if(cup415462.size() != 0){
                count_415462_cups = remove_duplicates_map(cup415462);
            }        //Remove duplicate lines from 512 cups.
            
            if (count_415462_cups == 0){cout << "\n**NO 4¹5⁴6² CUPS FOUND**\n\n" ;}
            else {std::cout << "# 4¹5⁴6²" << std::setw(10) << "\tcup: " << count_415462_cups << "\n\n";}


            ///////// CAGES ///////////////////////////////////////////////////////////////
            
            
            vector<vector<int>> cage_512, cage_62512, cage_64512, cage_4151062;
            vector<vector<int>> cage_512_rings, cage_62512_rings, cage_64512_rings; //TODO recombine with next
            vector<vector<Ring>> cage_4151062_rings;
            
            int cage_512_count = 0;
            
            if(count_512_cups > 0)
            {
                
                cage_512_count = cage_Finder("cage512", cup512, count_512_cups, My_neigh_ring5, cage_512, cage_512_rings, time);
                
                cage_512_count = remove_duplicates_map_cage_rings(cage_512_rings);
                
                //Write output gro file only if frameCounter is a multiple of DT. if -dt is not provided, every frame will be written.
                // if(cage_512_count>0  && (frameCounter % DT == 0))print_vmd_cage64512_frings("cage512", cup512, cage_512_count, cage_512_rings, ring4, ring5, ring6, atom_Pos, time, rawFilename, box_size_xyz, solutes, methane_512, solute1, topSolute, solute2, count_solute2, frameCounter);
                
            }
            
            std::cout << "# 5¹²" << std::setw(14) << "\tcage: " << cage_512_count << "\n";
            
            int cage_62512_count = 0;
            if(count_62512_cups > 0)
            {
                cage_62512_count = cage_Finder("cage62512", cup62512, count_62512_cups, My_neigh_ring5, cage_62512, cage_62512_rings, time);
                
                cage_62512_count = remove_duplicates_map_cage_rings(cage_62512_rings);
                
                //Write output gro file only if frameCounter is a multiple of DT. if -dt is not provided, every frame will be written.
                // if(cage_62512_count>0 && (frameCounter % DT) == 0)print_vmd_cage_frings("cage62512", cage_62512_count, cage_62512_rings, ring4, ring5, ring6, atom_Pos, time, rawFilename, box_size_xyz, solutes, methane_62512, solute1, topSolute, solute2, count_solute2, frameCounter);
            }
            std::cout << "# 6²5¹²" << std::setw(14) << "\tcage: " << cage_62512_count << "\n";
            /**********************************************/
            /*Finding Cage 64512*/
            int cage_64512_count = 0;
            
            cage_64512_count = cage_Finder_64512(cup62512, count_62512_cups, cage_64512_rings);
            
            std::cout << "# 6⁴5¹²" << std::setw(14) << "\tcage: " << cage_64512_count << "\n";

            int cage_4151062_count = 0;

            std::cout << std::fixed << std::setprecision(3);

            // for ( int k; k < cup415462[0].size(); k++) {
                
            //     vector<vector<int>> ring;

            //     if ( cup415462[0][k].type == SQUARE ) {
            //         ring = ring4;
            //         std:: cout << "SQUARE\n";
            //     }
            //     if ( cup415462[0][k].type == PENTAGON ) {
            //         ring = ring5;
            //         std:: cout << "PENTAGON\n";
            //     }
            //     if ( cup415462[0][k].type == HEXAGON ) {
            //         ring = ring6;
            //         std:: cout << "HEXAGON\n";
            //     }
            //     int index = cup415462[0][k].index;
            //     for ( int j = 0; j < ring[index].size(); j++) {
            //         std::cout <<  atom_Pos[ring[index][j]][0]  << endl;
            //         std::cout <<  atom_Pos[ring[index][j]][1]  << endl;
            //         std::cout <<  atom_Pos[ring[index][j]][2]  << endl;
            //         std::cout << "\n\n";
            //     }

            // }

            // Initialize min and max values to extremes
            // float min_x = std::numeric_limits<float>::max();
            // float max_x = std::numeric_limits<float>::lowest();
            // float min_y = std::numeric_limits<float>::max();
            // float max_y = std::numeric_limits<float>::lowest();
            // float min_z = std::numeric_limits<float>::max();
            // float max_z = std::numeric_limits<float>::lowest();

            // for (int k = 0; k < cup415462[0].size(); k++) {

            //     vector<vector<int>> ring;

            //     // Determine the type of the ring
            //     if (cup415462[0][k].type == SQUARE) {
            //         ring = ring4;
            //         std::cout << "SQUARE\n";
            //     }
            //     if (cup415462[0][k].type == PENTAGON) {
            //         ring = ring5;
            //         std::cout << "PENTAGON\n";
            //     }
            //     if (cup415462[0][k].type == HEXAGON) {
            //         ring = ring6;
            //         std::cout << "HEXAGON\n";
            //     }

            //     int index = cup415462[0][k].index;

            //     // Loop through all the atoms in the selected ring
            //     for (int j = 0; j < ring[index].size(); j++) {
            //         float x = atom_Pos[ring[index][j]][0];
            //         float y = atom_Pos[ring[index][j]][1];
            //         float z = atom_Pos[ring[index][j]][2];

            //         // Update min and max for x, y, and z
            //         if (x < min_x) min_x = x;
            //         if (x > max_x) max_x = x;
            //         if (y < min_y) min_y = y;
            //         if (y > max_y) max_y = y;
            //         if (z < min_z) min_z = z;
            //         if (z > max_z) max_z = z;
            //     }
            // }

            // // After all rings and atoms, output the overall min and max values for the cup
            // std::cout << "Overall Cup Dimensions:\n";
            // std::cout << "X: Min = " << min_x << ", Max = " << max_x << "\n";
            // std::cout << "Y: Min = " << min_y << ", Max = " << max_y << "\n";
            // std::cout << "Z: Min = " << min_z << ", Max = " << max_z << "\n";


            
            if(count_415462_cups > 0)
            {
                cage_4151062_count = cage_Finder_different_cups("cage4151062", cup415462, count_415462_cups, cup512, count_512_cups, ring5, My_neigh_ring5, My_neigh_ring6, My_neigh_ring6_ring5,  cage_4151062, cage_4151062_rings, time );
                // for (int cage = 0; cage < cage_4151062.size();cage++) {
                //     for (int cup = 0; cup < cage_4151062[cage].size(); cup++) {
                //         std::cout<<cage_4151062[cage][cup] << endl;
                //     }
                // }

                cage_4151062_count = remove_duplicates_map_cage(cage_4151062_rings);
                
                //Write output gro file only if frameCounter is a multiple of DT. if -dt is not provided, every frame will be written.
                // if(cage_62512_count>0 && (frameCounter % DT) == 0)print_vmd_cage64512_frings(cup62512, cage_62512_count, cage_62512_rings, ring4,  ring5, ring6, atom_Pos, time, rawFilename, box_size_xyz, solutes, methane_62512, solute1, topSolute, solute2, count_solute2, frameCounter);
            }
            cout << "# 4¹5¹⁰6²\tcage: " << cage_4151062_count << "\n";
            /**********************************************/
    

            // if(cage_4151062_count>0 && (frameCounter % DT) == 0)print_vmd_cage_frings("cage4151062", cage_4151062_count, cage_4151062_rings, ring4, ring5, ring6, atom_Pos, time, rawFilename, box_size_xyz, solutes, methane_4151062, solute1, topSolute, solute2, count_solute2, frameCounter);            
            /**********************************************/

            if(in_F4 == 1)      //If F4 flag is provided, calculate F4.
            {
                F4_value = calc_F4(count_solvent, count_solute, My_neigh, atom_Pos, boxX, boxY, boxZ, Nneigh, Natoms, topSolute, time, HBOND_DIST) ;
                if(F4_value > 0) outFile_F4 << frameCounter << "\t| " << F4_value << "\t| " << time << endl;
                else outFile_F4 << frameCounter << "\t|" << F4_value << "\t| " << time << endl;        //The if-else condition takes care of the extra space needed for "-" sign and alligns the output(lazy way!).
                
            }

            outFile << std::setw(15)              // Frame column
                    << std::setw(15) << cage_512_count     // 512_Cages column
                    << std::setw(15) << cage_62512_count   // 62512_Cages column
                    << std::setw(15) << cage_64512_count   // 64512_Cages column
                    << std::setw(15) << cage_4151062_count
                    << std::setw(15) << F4_value          // F4 column
                    << "\n";

            if(found_time != string::npos)              //If found_time is not null, do the following.
            {
                time = line.substr(found_time+3);
            }

    

            // outputFilename = "frame_data_output.txt";

    //         std::ofstream frameDataFile(outputFilename, std::ios::app);  // Open file for appending

    //         if (frameDataFile.tellp() == 0) {
    //             frameDataFile << std::left << std::setw(10) << "Frame"
    //                         << std::setw(15) << "512_Cages"
    //                         << std::setw(15) << "62512_Cages"
    //                         << std::setw(15) << "64512_Cages"
    //                         << std::setw(10) << "F4"
    //                         << "\n";
    // }
        

            // print_data(frameDataFile, frameCounter, cage_512_count, cage_62512_count, cage_64512_count, F4_value);

            
        }

    
    }   //End of reading the input file.
    
    
    fileIN.close();
    outFile.close();
    
    if( in_F4 == 1 )outFile_F4.close();
    
    return 0;
    
} // Close main function







