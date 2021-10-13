# Run to get the parms.
#python2.7 -m /media/masoud/WRKP/EDesign/thirdparty/molfile2params/molfile_to_params -n XXX -c  XXX.mol
python2.7 -m /media/masoud/WRKP/EDesign/thirdparty/molfile2params/molfile_to_params --keep-names --clobber -n ZZZ -c  ZZZ.mol2

# Add MM NBR  nbr atom 
# Add MM CHG  total charge
#Copy the parms in the corresponding folders:
#    EDesignTools/lib/python3.7/site-packages/pyrosetta/database/chemical/residue_type_sets/fa_standard/residue_types/
#    EDesignTools/lib/python3.7/site-packages/pyrosetta/database/chemical/residue_type_sets/centroid/residue_types/

#Update the corresponding files:
#    EDesignTools/lib/python3.7/site-packages/pyrosetta/database/database/chemical/residue_type_sets/fa_standard/residue_types.txt
#    EDesignTools/lib/python3.7/site-packages/pyrosetta/database/database/chemical/residue_type_sets/centroid/residue_types.txt
