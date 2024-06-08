# -*- coding: utf-8 -*-
"""
Created on Fri March 14 17:41:37 2024

@author: Gerardo Casanola & Karel Dieguez-Santana
"""


#%% Importing libraries

from pathlib import Path
import pandas as pd
import pickle
from molvs import Standardizer
from rdkit import Chem
from openbabel import openbabel
from mordred import Calculator, descriptors
from multiprocessing import freeze_support
import numpy as np
from rdkit.Chem import AllChem
import plotly.graph_objects as go
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

#Import Libraries
import math 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

# packages for streamlit
import streamlit as st
from PIL import Image
import io
import base64

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdFingerprintGenerator, Descriptors, Draw
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs import cDataStructs
from io import StringIO
from mordred import Calculator, descriptors
import numpy as np
import pandas as pd
import seaborn as sns
import sys, os, shutil
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_ketcher import st_ketcher
import time
import subprocess
from PIL import Image
import uuid
from filelock import Timeout, FileLock



#%% PAGE CONFIG

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='ML predictor for Aquatic Ecotoxicology prediction of Rotifer species', page_icon=":computer:", layout='wide')

######
# Function to put a picture as header   
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

image = Image.open('cropped-header.png')
st.image(image)

#st.write("[![Website](https://img.shields.io/badge/website-RasulevGroup-blue)](http://www.rasulev.org)")
#st.subheader("📌" "About Us")
#st.markdown("The group of Prof. Rasulev is focused on development of artificial intelligence (AI)-based predictive models to design novel polymeric materials, nanomaterials and to predict their various properties, including toxicity, solubility, fouling release properties, elasticity, degradation rate, biodegradation, etc. The group applies computational chemistry, machine learning and cheminformatics methods for modeling, data analysis and development of predictive structure-property relationship models to find structural factors responsible for activity of investigated materials.")


# Introduction
#---------------------------------#

#st.title(':computer: ml rotifer sp toxicity in water')
#with st.expander("<span style='color: blue;'>More information</span>", expanded=True):
with st.expander("More information",):
    st.write("""

    **It is a free web-application for Aquatic Ecotoxicology prediction in Rotifer species**

    Among the organisms used to assess water quality and environmental toxicity are rotifers. These invertebrate species are abundant and widely distributed
    in aquatic ecosystems and can therefore play an essential role in the ecological processes of aquatic ecosystems. Due to their rapid population turnover rate, 
    rotifers contribute significantly to nutrient recycling in aquatic habitats. Therefore, if rotifer populations are negatively affected by a toxin, 
    the function of aquatic ecosystems could be altered. Rotifers have been used as model organisms to evaluate the toxicity 
    of many environmental chemicals, including heavy metals, organic compounds, and nano-sized materials .

    The ML Aquatic Ecotox Rotifer sp predictor is a Web App that use Machine Learning to predict the aquatic ecotoxicology risk assesment of organic compounds. 

    The tool uses the following packages [RDKIT](https://www.rdkit.org/docs/index.html), [Mordred](https://github.com/mordred-descriptor/mordred), [MOLVS](https://molvs.readthedocs.io/), [Openbabel](https://github.com/openbabel/openbabel),
    [Scikit-learn](https://scikit-learn.org/stable/)
    **Workflow:**
    """)

with st.expander("Workflow"):
    image = Image.open('toc.png')
    st.image(image, caption='ML Rotifer Aquatic Toxicity workflow')


#---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV file')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/gmaikelc/rotifer_ml_predictor/main/example_file1.csv) 
""")

uploaded_file_1 = st.sidebar.file_uploader("Upload a CSV file with SMILES and fractions", type=["csv"])




#%% Standarization by MOLVS ####
####---------------------------------------------------------------------------####

def standardizer(df,pos):
    s = Standardizer()
    molecules = df[pos].tolist()
    standardized_molecules = []
    smi_pos=pos-2
    i = 1
    t = st.empty()
    
    

    for molecule in molecules:
        try:
            smiles = molecule.strip()
            mol = Chem.MolFromSmiles(smiles)
            standarized_mol = s.super_parent(mol) 
            standardizer_smiles = Chem.MolToSmiles(standarized_mol)
            standardized_molecules.append(standardizer_smiles)
            # st.write(f'\rProcessing molecule {i}/{len(molecules)}', end='', flush=True)
            t.markdown("Processing monomers: " + str(i) +"/" + str(len(molecules)))

            i = i + 1
        except:
            standardized_molecules.append(molecule)
    df['standarized_SMILES'] = standardized_molecules

    return df


#%% Protonation state at pH 7.4 ####
####---------------------------------------------------------------------------####

def charges_ph(molecule, ph):

    # obConversion it's neccesary for saving the objects
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("smi", "smi")
    
    # create the OBMol object and read the SMILE
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, molecule)
    
    # Add H, correct pH and add H again, it's the only way it works
    mol.AddHydrogens()
    mol.CorrectForPH(7.4)
    mol.AddHydrogens()
    
    # transforms the OBMOl objecto to string (SMILES)
    optimized = obConversion.WriteString(mol)
    
    return optimized

def smile_obabel_corrector(smiles_ionized):
    mol1 = Chem.MolFromSmiles(smiles_ionized, sanitize = False)
    
    # checks if the ether group is wrongly protonated
    pattern1 = Chem.MolFromSmarts('[#6]-[#8-]-[#6]')
    if mol1.HasSubstructMatch(pattern1):
        # gets the atom number for the O wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern1)
        at_matches_list = [y[1] for y in at_matches]
        # changes the charged for each O atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    pattern12 = Chem.MolFromSmarts('[#6]-[#8-]-[#16]')
    if mol1.HasSubstructMatch(pattern12):
        # gets the atom number for the O wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern12)
        at_matches_list = [y[1] for y in at_matches]
        # changes the charged for each O atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()
            
    # checks if the nitro group is wrongly protonated in the oxygen
    pattern2 = Chem.MolFromSmarts('[#6][O-]=[N+](=O)[O-]')
    if mol1.HasSubstructMatch(pattern2):
        # print('NO 20')
        patt = Chem.MolFromSmiles('[O-]=[N+](=O)[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('O[N+]([O-])=O')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated in the oxygen
    pattern21 = Chem.MolFromSmarts('[#6]-[O-][N+](=O)=[O-]')
    if mol1.HasSubstructMatch(pattern21):
        # print('NO 21')
        patt = Chem.MolFromSmiles('[O-][N+](=O)=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[O][N+](=O)-[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]
        
    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern22 = Chem.MolFromSmarts('[#8-][N+](=[#6])=[O-]')
    if mol1.HasSubstructMatch(pattern22):
        # print('NO 22')
        patt = Chem.MolFromSmiles('[N+]([O-])=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[N+]([O-])-[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern23 = Chem.MolFromSmarts('[#6][N+]([#6])([#8-])=[O-]')
    if mol1.HasSubstructMatch(pattern23):
        # print('NO 23')
        patt = Chem.MolFromSmiles('[N+]([O-])=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[N+]([O-])[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the nitro group is wrongly protonated, different disposition of atoms
    pattern24 = Chem.MolFromSmarts('[#6]-[#8][N+](=O)=[O-]')
    if mol1.HasSubstructMatch(pattern24):
        # print('NO 24')
        patt = Chem.MolFromSmiles('[O][N+](=O)=[O-]', sanitize = False)
        repl = Chem.MolFromSmiles('[O][N+](=O)[O-]')
        rms = AllChem.ReplaceSubstructs(mol1,patt,repl,replaceAll=True)
        mol1 = rms[0]

    # checks if the 1H-tetrazole group is wrongly protonated
    pattern3 = Chem.MolFromSmarts('[#7]-1-[#6]=[#7-]-[#7]=[#7]-1')
    if mol1.HasSubstructMatch(pattern3):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern3)
        at_matches_list = [y[2] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    # checks if the 2H-tetrazole group is wrongly protonated
    pattern4 = Chem.MolFromSmarts('[#7]-1-[#7]=[#6]-[#7-]=[#7]-1')
    if mol1.HasSubstructMatch(pattern4):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern4)
        at_matches_list = [y[3] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()
        
    # checks if the 2H-tetrazole group is wrongly protonated, different disposition of atoms
    pattern5 = Chem.MolFromSmarts('[#7]-1-[#7]=[#7]-[#6]=[#7-]-1')
    if mol1.HasSubstructMatch(pattern5):
        # gets the atom number for the N wrongly charged
        at_matches = mol1.GetSubstructMatches(pattern4)
        at_matches_list = [y[4] for y in at_matches]
        # changes the charged for each N atom
        for at_idx in at_matches_list:
            atom = mol1.GetAtomWithIdx(at_idx)
            atom.SetFormalCharge(0)
            atom.UpdatePropertyCache()

    smile_checked = Chem.MolToSmiles(mol1)
    return smile_checked


#%% formal charge calculation

def formal_charge_calculation(descriptors):
    smiles_list = descriptors["Smiles_OK"]
    charges = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            charge = Chem.rdmolops.GetFormalCharge(mol)
            charges.append(charge)
        except:
            charges.append(None)
        
    descriptors["Formal_charge"] = charges
    return descriptors



#%% B04[O-O] descriptor calculation

def check_oo_distance(descriptors):
    # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    distance4 = []
    
    # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            distance4.append(float('nan'))
            continue
        
        # Generate the molecular graph representation
        mol_graph = Chem.RWMol(mol)
        Chem.SanitizeMol(mol_graph)
        mol_graph = Chem.RemoveHs(mol_graph)
        mol_graph = Chem.GetAdjacencyMatrix(mol_graph)
        G = nx.Graph(mol_graph)
        
        # Initialize the presence/absence flag
        presence_flag = 0
        
        # Find all pairs of oxygen atoms in the molecule
        oxygen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'O']
        
        # Check for paths of length 4 between oxygen atoms
        for source in oxygen_atoms:
            for target in oxygen_atoms:
                if source != target:
                    # Use networkx shortest_path_length to check the shortest path length
                    shortest_path_length = nx.shortest_path_length(G, source=source, target=target)
                    if shortest_path_length == 4:
                        presence_flag = 1
                        break
            if presence_flag == 1:
                break
        
        # Append the result to the list
        distance4.append(presence_flag)
    
    # Add the results as a new column in the DataFrame
    descriptors['B04[O-O]'] = distance4
    
    return descriptors

#%% B07[Cl-Cl] descriptor calculation

def check_clcl_distance(descriptors):
    # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    distance7 = []
    
    # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            distance7.append(float('nan'))
            continue
        
        # Generate the molecular graph representation
        mol_graph = Chem.RWMol(mol)
        Chem.SanitizeMol(mol_graph)
        mol_graph = Chem.RemoveHs(mol_graph)
        mol_graph = Chem.GetAdjacencyMatrix(mol_graph)
        G = nx.Graph(mol_graph)
        
        # Initialize the presence/absence flag
        presence_flag = 0
        
        # Find all pairs of oxygen atoms in the molecule
        chloride_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl']
        
        # Check for paths of length 7 between chloride atoms
        for source in chloride_atoms:
            for target in chloride_atoms:
                if source != target:
                    # Use networkx shortest_path_length to check the shortest path length
                    shortest_path_length = nx.shortest_path_length(G, source=source, target=target)
                    if shortest_path_length == 7:
                        presence_flag = 1
                        break
            if presence_flag == 1:
                break
        
        # Append the result to the list
        distance7.append(presence_flag)
    
    # Add the results as a new column in the DataFrame
    descriptors['B07[Cl-Cl]'] = distance7
    
    return descriptors

def check_clc_distance(descriptors):
    # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    distance2 = []
    
    # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            distance2.append(float('nan'))
            continue
        
        # Generate the molecular graph representation
        mol_graph = Chem.RWMol(mol)
        Chem.SanitizeMol(mol_graph)
        mol_graph = Chem.RemoveHs(mol_graph)
        mol_graph = Chem.GetAdjacencyMatrix(mol_graph)
        G = nx.Graph(mol_graph)
        
        # Initialize the presence/absence flag
        presence_flag = 0
        
        # Find all pairs of carbon and chloride atoms in the molecule
        carbon_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']
        chloride_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl']
        
        # Check for paths of length 2 between carbon and chloride atoms
        for carbon in carbon_atoms:
            for chloride in chloride_atoms:
                if carbon != chloride:
                    # Use networkx shortest_path_length to check the shortest path length
                    shortest_path_length = nx.shortest_path_length(G, source=carbon, target=chloride)
                    if shortest_path_length == 2:
                        presence_flag = 1
                        break
            if presence_flag == 1:
                break
        
        # Append the result to the list
        distance2.append(presence_flag)
    
    # Add the results as a new column in the DataFrame
    descriptors['B02[C-Cl]'] = distance2
    
    return descriptors

def check_ns_distance(descriptors):
    # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    frequency_ns = []
    
    # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            frequency_ns.append(float('nan'))
            continue
        
        # Generate the molecular graph representation
        mol_graph = Chem.RWMol(mol)
        Chem.SanitizeMol(mol_graph)
        mol_graph = Chem.RemoveHs(mol_graph)
        mol_graph = Chem.GetAdjacencyMatrix(mol_graph)
        G = nx.Graph(mol_graph)
        
        # Initialize the frequency counter
        ns_frequency = 0
        
        # Find all pairs of nitrogen and sulfur atoms in the molecule
        nitrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'N']
        sulfur_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'S']
        
        # Check for paths of length 2 between nitrogen and sulfur atoms
        for nitrogen in nitrogen_atoms:
            for sulfur in sulfur_atoms:
                if nitrogen != sulfur:
                    # Use networkx shortest_path_length to check the shortest path length
                    shortest_path_length = nx.shortest_path_length(G, source=nitrogen, target=sulfur)
                    if shortest_path_length == 2:
                        ns_frequency += 1
        
        # Append the result to the list
        frequency_ns.append(ns_frequency)
    
    # Add the results as a new column in the DataFrame
    descriptors['F02[N-S]'] = frequency_ns
    
    return descriptors

def count_rcx(descriptors):
  # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    count_rchx = []

     # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            frequency_ns.append(float('nan'))
            continue
            
        rcx_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetIsAromatic():
                neighbors = atom.GetNeighbors()
                if len(neighbors) == 2:
                    symbols = [neighbor.GetSymbol() for neighbor in neighbors]
                    if any(x in symbols for x in ['O', 'N', 'S', 'P', 'Se', 'F', 'Cl', 'Br', 'I']):
                        rcx_count += 1
    
        # Append the result to the list
        count_rchx.append(rcx_count)
    
    # Add the results as a new column in the DataFrame
    descriptors['C-033'] =  count_rchx
    
    return descriptors

def count_imide_groups(descriptors):
    # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    imide_counts = []
    
    # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            imide_counts.append(float('nan'))
            continue
        
        # Initialize the count of imide groups
        imide_count = 0
        
        # Find all nitrogen atoms in the molecule
        nitrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'N']
        
        # Iterate over each nitrogen atom
        for nitrogen in nitrogen_atoms:
            # Get the neighboring atoms of the nitrogen atom
            neighbors = mol.GetAtomWithIdx(nitrogen).GetNeighbors()
            
            # Check if the nitrogen atom is connected to two carbonyl carbon atoms
            carbonyl_count = 0
            for neighbor in neighbors:
                if neighbor.GetSymbol() == 'C':
                    # Check if the carbon atom is part of a carbonyl group (double bond to oxygen)
                    for bond in neighbor.GetBonds():
                        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(neighbor).GetSymbol() == 'O':
                            carbonyl_count += 1
            if carbonyl_count >= 2:
                imide_count += 1
        
        # Append the result to the list
        imide_counts.append(imide_count)
    
    # Add the results as a new column in the DataFrame
    descriptors['nN(CO)2'] = imide_counts
    
    return descriptors

def count_amidine_groups(descriptors):
    # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    amidine_counts = []
    
    # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            amidine_counts.append(float('nan'))
            continue
        
        # Initialize the count of amidine groups
        amidine_count = 0
        
        # Find all nitrogen atoms in the molecule
        nitrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'N']
        
        # Iterate over each nitrogen atom
        for nitrogen in nitrogen_atoms:
            # Get the neighboring atoms of the nitrogen atom
            neighbors = mol.GetAtomWithIdx(nitrogen).GetNeighbors()
            
            # Check if the nitrogen atom is connected to two carbon atoms with a double bond between them
            for neighbor in neighbors:
                if neighbor.GetSymbol() == 'C':
                    # Check if there is a double bond between the carbon atoms
                    for bond in neighbor.GetBonds():
                        if bond.GetBondType() == Chem.BondType.DOUBLE:
                            # Check if the neighboring carbon atom has another nitrogen neighbor
                            carbon_neighbors = neighbor.GetNeighbors()
                            nitrogen_neighbor_count = sum(1 for atom in carbon_neighbors if atom.GetSymbol() == 'N')
                            if nitrogen_neighbor_count >= 2:
                                amidine_count += 1
                                break  # No need to continue checking for more amidine groups
                        break  # No need to continue checking for more double bonds
        
        # Append the result to the list
        amidine_counts.append(amidine_count)
    
    # Add the results as a new column in the DataFrame
    descriptors['nN=C-N<'] = amidine_counts
    
    return descriptors

def sum_all_cl_cl_distances(descriptors):
    # Initialize a list to store the results
    smiles_list = descriptors["Smiles_OK"]
    all_cl_cl_distances = []
    
    # Iterate over the SMILES in the specified column of the DataFrame
    for smiles in smiles_list:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Append NaN if SMILES cannot be converted to a molecule
            all_cl_cl_distances.append(float('nan'))
            continue
        
        # Generate the molecular graph representation
        mol_graph = Chem.RWMol(mol)
        Chem.SanitizeMol(mol_graph)
        mol_graph = Chem.RemoveHs(mol_graph)
        mol_graph = Chem.GetAdjacencyMatrix(mol_graph)
        G = nx.Graph(mol_graph)
        
        # Initialize the sum of topological distances between all pairs of chlorine atoms
        all_cl_cl_distance_sum = 0
        
        # Find all pairs of chlorine atoms in the molecule
        chlorine_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl']
        chlorine_pairs = [(i, j) for i in range(len(chlorine_atoms)) for j in range(i + 1, len(chlorine_atoms))]
        
        # Check for shortest paths between pairs of chlorine atoms and sum their distances
        for source, target in chlorine_pairs:
            shortest_path_length = nx.shortest_path_length(G, source=chlorine_atoms[source], target=chlorine_atoms[target])
            all_cl_cl_distance_sum += shortest_path_length
        
        # Append the result to the list
        all_cl_cl_distances.append(all_cl_cl_distance_sum)
    
    # Add the results as a new column in the DataFrame
    descriptors['T(Cl..Cl)'] = all_cl_cl_distances
    
    return descriptors


#%% Calculating molecular descriptors
### ----------------------- ###

def calc_descriptors(data, smiles_col_pos):
    descriptors_total_list = []
    smiles_list = []
    t = st.empty()
    
    # Loop through each molecule in the dataset
    for pos, row in data.iterrows():
        molecule_name = row[0]  # Assuming the first column contains the molecule names
        molecule_smiles = row[smiles_col_pos]  # Assuming the specified column contains the SMILES

        if pd.isna(molecule_smiles) or molecule_smiles.strip() == '':
            continue  # Skip to the next row if SMILES is empty
            
        mol = Chem.MolFromSmiles(molecule_smiles)
        if mol is not None:
            smiles_ionized = charges_ph(molecule_smiles, 7.4)
            smile_checked = smile_obabel_corrector(smiles_ionized)
            smile_final = smile_checked.rstrip()
            smiles_list.append(smile_final)
                
            calc = Calculator(descriptors, ignore_3D=True)
            descriptor_values = calc(mol).asdict()
                
            # Create a dictionary with molecule name as key and descriptor values as values
            descriptors_dict = {'NAME': molecule_name}
            descriptors_dict.update(descriptor_values)
                
            descriptors_total_list.append(descriptors_dict)
            st.spinner('CALCULATING DESCRIPTORS (STEP 1 OF 4)...')
            t.markdown("Calculating descriptors for molecule: " + str(pos +1) +"/" + str(len(data.iloc[:,0])))
    
    # Convert the list of dictionaries to a DataFrame
    descriptors_total = pd.DataFrame(descriptors_total_list)
    descriptors_total = descriptors_total.set_index('NAME', inplace=False).copy()
    descriptors_total = descriptors_total.reindex(sorted(descriptors_total.columns), axis=1)   
    descriptors_total.replace([np.inf, -np.inf], np.nan, inplace=True)
    descriptors_total["Smiles_OK"] = smiles_list
    
    # Perform formal charge calculation
    descriptors_total = formal_charge_calculation(descriptors_total)

    # Perform B04[O-O] descriptor calculation
    descriptors_total = check_oo_distance(descriptors_total)

     # Perform nN=C-N< descriptor calculation
    descriptors_total = count_amidine_groups(descriptors_total)

    #Perform B07[Cl-Cl] descriptor calculation
    descriptors_total = check_clcl_distance(descriptors_total)

    #Perform T(Cl..Cl) descriptor calculation
    descriptors_total = sum_all_cl_cl_distances(descriptors_total)

    #Perform nN(CO)2 descriptor calculation
    descriptors_total = count_imide_groups(descriptors_total)

    #Perform B02[C-Cl] molecular descriptor calculation
    descriptors_total = check_clc_distance(descriptors_total)

    #Perform C-033 molecular descriptor calculation
    descriptors_total = count_rcx(descriptors_total)
    
    #Perform F02[N-S] molecular descriptor calculation
    descriptors_total = check_ns_distance(descriptors_total)
    

    return descriptors_total, smiles_list



def reading_reorder(data):
        
    #Select the specified columns from the DataFrame
    df_selected = data[loaded_desc]
    df_id = data.reset_index()
    df_id.rename(columns={'index': 'NAME'}, inplace=True)
    id = df_id['NAME'] 
    # Order the DataFrame by the specified list of columns
    test_data = df_selected.reindex(columns=loaded_desc)
    #descriptors_total = data[loaded_desc]


    return test_data, id


#%% normalizing data
### ----------------------- ###

def normalize_data(train_data, test_data):
    # Normalize the training data
    df_train = pd.DataFrame(train_data)
    saved_cols = df_train.columns
    min_max_scaler = preprocessing.MinMaxScaler().fit(df_train)
    np_train_scaled = min_max_scaler.transform(df_train)
    df_train_normalized = pd.DataFrame(np_train_scaled, columns=saved_cols)

    # Normalize the test data using the scaler fitted on training data
    np_test_scaled = min_max_scaler.transform(test_data)
    df_test_normalized = pd.DataFrame(np_test_scaled, columns=saved_cols)

    return df_train_normalized, df_test_normalized


#%% Determining Applicability Domain (AD)

def applicability_domain(x_test_normalized, x_train_normalized):
    y_train=data_train['pLC50_sw']
    X_train = x_train_normalized.values
    X_test = x_test_normalized.values
    # Calculate leverage and standard deviation for the training set
    hat_matrix_train = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T
    leverage_train = np.diagonal(hat_matrix_train)
    leverage_train=leverage_train.ravel()
    
    # Calculate leverage and standard deviation for the test set
    hat_matrix_test = X_test @ np.linalg.inv(X_train.T @ X_train) @ X_test.T
    leverage_test = np.diagonal(hat_matrix_test)
    leverage_test=leverage_test.ravel()


    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(df_train_normalized, y_train)
    y_pred_train = lr.predict(df_train_normalized)
    
    std_dev_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    std_residual_train = (y_train - y_pred_train) / std_dev_train
    std_residual_train = std_residual_train.ravel()
    
    # threshold for the applicability domain
    
    h3 = 3*((x_train_normalized.shape[1]+1)/x_train_normalized.shape[0])  
    
    diagonal_compare = list(leverage_test)
    h_results =[]
    for valor in diagonal_compare:
        if valor < h3:
            h_results.append(True)
        else:
            h_results.append(False)         
    return h_results, leverage_train, leverage_test, std_residual_train 



 # Function to assign colors based on confidence values
def get_color(confidence):
    """
    Assigns a color based on the confidence value.

    Args:
        confidence (float): The confidence value.

    Returns:
        str: The color in hexadecimal format (e.g., '#RRGGBB').
    """
    # Define your color logic here based on confidence
    if confidence == "HIGH" or confidence == "Inside AD":
        return 'green'
    elif confidence == "MEDIUM":
        return 'yellow'
    else:
        confidence ==  "LOW"
        return 'red'


#%% Predictions        

def predictions(loaded_model, loaded_desc, df_test_normalized):
    scores = []
    h_values = []
    std_resd = []
    idx = data['ID']
    

    descriptors_model = loaded_desc
        
    X = df_test_normalized[descriptors_model]
    predictions = loaded_model.predict(X)
    scores.append(predictions)
        
    # y_true and y_pred are the actual and predicted values, respectively
    
    # Create y_true array with all elements set to mean value and the same length as y_pred
    y_pred_test = predictions
    y_test = np.full_like(y_pred_test, mean_value)
    residuals_test = y_test -y_pred_test

    std_dev_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    std_residual_test = (y_test - y_pred_test) / std_dev_test
    std_residual_test = std_residual_test.ravel()
          
    std_resd.append(std_residual_test)
        
    h_results, leverage_train, leverage_test, std_residual_train  = applicability_domain(df_test_normalized, df_train_normalized)
    h_values.append(h_results)
    

    dataframe_pred = pd.DataFrame(scores).T
    dataframe_pred.index = idx
    dataframe_pred.rename(columns={0: "pLC50"},inplace=True)
    
    dataframe_std = pd.DataFrame(std_resd).T
    dataframe_std.index = idx
          
        
    h_final = pd.DataFrame(h_values).T
    h_final.index = idx
    h_final.rename(columns={0: "Confidence"},inplace=True)

    std_ensemble = dataframe_std.iloc[:,0]
    # Create a mask using boolean indexing
    std_ad_calc = (std_ensemble >= 3) | (std_ensemble <= -3) 
    std_ad_calc = std_ad_calc.replace({True: 'Outside AD', False: 'Inside AD'})
   
    
    final_file = pd.concat([std_ad_calc,h_final,dataframe_pred], axis=1)
    
    final_file.rename(columns={0: "Std_residual"},inplace=True)
    
    h3 = 3*((df_train_normalized.shape[1]+1)/df_train_normalized.shape[0])  ##  Mas flexible

    final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Inside AD' )), 'Confidence'] = 'HIGH'
    final_file.loc[(final_file["Confidence"] == True) & ((final_file["Std_residual"] == 'Outside AD')), 'Confidence'] = 'LOW'
    final_file.loc[(final_file["Confidence"] == False) & ((final_file["Std_residual"] == 'Outside AD')), 'Confidence'] = 'LOW'
    final_file.loc[(final_file["Confidence"] == False) & ((final_file["Std_residual"] == 'Inside AD')), 'Confidence'] = 'MEDIUM'


            
    df_no_duplicates = final_file[~final_file.index.duplicated(keep='first')]
    styled_df = df_no_duplicates.style.apply(lambda row: [f"background-color: {get_color(row['Confidence'])}" for _ in row],subset=["Confidence"], axis=1)
    
    return final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test

#Calculating the William's plot limits
def calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4):
    # Getting maximum std value
    if std_residual_train.max() < 4:
        x_lim_max_std = x_std_max
    elif std_residual_train.max() > 4:
        x_lim_max_std = round(std_residual_train.max()) + 1

    # Getting minimum std value
    if std_residual_train.min() > -4:
        x_lim_min_std = x_std_min
    elif std_residual_train.min() < 4:
        x_lim_min_std = round(std_residual_train.min()) - 1

    
    #st.write('x_lim_max_std:', x_lim_max_std)
    #st.write('x_lim_min_std:', x_lim_min_std)

    # Calculation H critical
    n = len(leverage_train)
    p = df_train_normalized.shape[1]
    h_value = 3 * (p + 1) / n
    h_critical = round(h_value, 4)
    #st.write('Number of cases training:', n)
    #st.write('Number of variables:', p)
    #st.write('h_critical:', h_critical)

    # Getting maximum leverage value
    if leverage_train.max() < h_critical:
        x_lim_max_lev = h_critical + h_critical * 0.5
    elif leverage_train.max() > h_critical:
        x_lim_max_lev = leverage_train.max() + (leverage_train.max()) * 0.1

    # Getting minimum leverage value
    if leverage_train.min() < 0:
        x_lim_min_lev = x_lev_min - x_lev_min * 0.05
    elif leverage_train.min() > 0:
        x_lim_min_lev = 0

    #st.write('x_lim_max_lev:', x_lim_max_lev)

    return x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev


def williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test,
                  plot_color='cornflowerblue', show_plot=True, save_plot=False, filename=None, add_title=False, title=None):
        sns.set(color_codes=True)
        sns.set_style('white')
        sns.despine(left=True, right=True)
    
        fig, ax = plt.subplots()
        fig.set_size_inches(4.5, 4.5)

     
        ax1 = sns.regplot(x=leverage_train, y=std_residual_train, ax=ax, fit_reg=False, color=plot_color,
                      marker='o', scatter_kws={'s': 120, 'linewidths': 0.5, 'edgecolor': 'k'})

        ax2=sns.regplot(x=leverage_test, y=std_residual_test,ax=ax, fit_reg=False,color='orange',
                marker='o',scatter_kws={'s':120,
                                       'linewidths':0.5,
                             'edgecolor':'k',})

        ax1.text(h_critical, x_lim_min_std + 0.45, f'h*={h_critical}', verticalalignment='top', horizontalalignment='right',
             color='black', fontsize=15, fontweight='normal')

        ax1.text((h_critical + x_lim_min_lev) / 2, -3.50, 'Outlier zone', verticalalignment='center',
             horizontalalignment='center',
             color='black', fontsize=15, fontweight='normal')
        ax1.text((h_critical + x_lim_min_lev) / 2, 3.50, 'Outlier zone', verticalalignment='center',
             horizontalalignment='center',
             color='black', fontsize=15, fontweight='normal')

        ax1.text((h_critical + x_lim_max_lev) / 2, -3.50, 'Outlier zone', verticalalignment='center',
             horizontalalignment='center',
             color='black', fontsize=15, fontweight='normal')
        ax1.text((h_critical + x_lim_max_lev) / 2, 3.50, 'Outlier zone', verticalalignment='center',
             horizontalalignment='center',
             color='black', fontsize=15, fontweight='normal')

        plt.ylim(x_lim_min_std, x_lim_max_std)
        plt.xlim(x_lim_min_lev, x_lim_max_lev)

        # Custom legend
        training_legend = Line2D([], [], color='cornflowerblue', marker='o', markersize=10, label='Training')
        validation_legend = Line2D([], [], color='orange', marker='o', markersize=10, label='Validation')
        plt.legend(handles=[training_legend, validation_legend], bbox_to_anchor=(0.99, 0.34), loc='upper right',
                   fontsize=14, edgecolor='black')
                      
        #plt.legend(['training','validation'],bbox_to_anchor=(0.99, 0.34),loc='upper right',
         #   fontsize=14,edgecolor='black')
                      
        plt.plot([h_critical, h_critical], [x_lim_min_std, x_lim_max_std], lw=1, ls='dashed', color='black')
        plt.plot([x_lim_min_lev, x_lim_max_lev], [3, 3], lw=1, ls='dashed', color='black')
        plt.plot([x_lim_min_lev, x_lim_max_lev], [-3, -3], lw=1, ls='dashed', color='black')

        ax1.add_patch(
            patches.Rectangle(
                xy=(x_lim_min_lev, x_lim_min_std),  # point of origin.
                width=7, height=abs(x_lim_min_std) - 3, linewidth=1,
                color='lightgray', fill=True, alpha=0.4))

        ax1.add_patch(
                patches.Rectangle(
                xy=(x_lim_min_lev, 3),  # point of origin.
                width=3, height=x_lim_max_std - 3, linewidth=1,
                color='lightgray', fill=True, alpha=0.4))

        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')

        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)

        plt.ylabel('Std Residuals', fontsize=20)
        plt.xlabel('Leverage', fontsize=20)
        x_axis_text = (x_lim_min_lev + h_critical) / 2
        #plt.text(x_axis_text, 2.2, 'Chemical Space \nPredictions Reliable',
         #        horizontalalignment='center', fontsize=10, fontweight='bold')
    
        if add_title and title:
            plt.title(title, fontsize=20)

        plt.tight_layout()

        if save_plot and filename:
            plt.savefig(filename, dpi=600, transparent=True)
    
        if show_plot:
            plt.show()

        return fig




#%%
def filedownload1(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="ml_toxicity_rotifer_pLC50_results.csv">Download CSV File with results</a>'
    return href

def filedownload2(df):
    csv = df.to_csv(index=True,header=True)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Component1 results">Download CSV File with results Component1</a>'
    return href



#%% RUN

data_train = pd.read_csv("data/" + "data_126c_15var_pLC50_train_sw.csv")
mean_value = data_train['pLC50_sw'].mean()


loaded_model = pickle.load(open("models/" + "ml_model_rotifer_sw.pickle", 'rb'))
loaded_desc = pickle.load(open("models/" + "ml_descriptor_rotifer_sw.pickle", 'rb'))

#Uploaded file calculation
if uploaded_file_1 is not None:
    run = st.button("RUN")
    if run == True:
        data = pd.read_csv(uploaded_file_1,) 
        train_data = data_train[loaded_desc]
        # Calculate descriptors and SMILES for the first column
        descriptors_total_1, smiles_list_1 = calc_descriptors(data, 1)
        # Calculate descriptors and SMILES for the second column
        #descriptors_total_2, smiles_list_2 = calc_descriptors(data, 4)

        # Calculate descriptors and SMILES for the first column with progress bar
        #descriptors_total_1, smiles_list_1 = calc_descriptors_with_progress(data, 3, "Component1")
        # Calculate descriptors and SMILES for the second column with progress bar
        #descriptors_total_2, smiles_list_2 = calc_descriptors_with_progress(data, 4, "Component2")
        
        #joint_dummy = descriptors_total_1[['Formal_charge']]
        # Left join
        #descriptors_total_2n = joint_dummy.join(descriptors_total_2, how='left', lsuffix='_df1', rsuffix='_df2')
        #drop the first column
        #descriptor_total_2na = descriptors_total_2n.iloc[:,1:]
        # Fill NaN values with 0
        #descriptors_total_2m = descriptor_total_2na.fillna(0)
                
        #Selecting the descriptors based on model for first component
        test_data1, id_list_1 =  reading_reorder(descriptors_total_1)
        #Selecting the descriptors based on model for first component
        #test_data2, id_list_1 =  reading_reorder(descriptors_total_2m)
 
        #st.markdown(filedownload2(test_data1), unsafe_allow_html=True)
        #st.markdown(filedownload3(test_data2), unsafe_allow_html=True)
        
        #Calculating mixture descriptors    
        #test_data_mix= mixture_descriptors(test_data1,test_data2)
        #test_data_mix.fillna(0,inplace=True)
        #st.markdown(filedownload4(test_data_mix), unsafe_allow_html=True)
                
        #X_final1, id = all_correct_model(test_data_mix,loaded_desc, id_list)
        X_final2= test_data1
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        #st.markdown(filedownload5(df_test_normalized), unsafe_allow_html=True)
        final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
        figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test)   
        col1, col2 = st.columns(2)

        with col1:
            st.header("",divider='blue')
            st.subheader(r'pLC50 salt water')
            st.write(styled_df)
        with col2:
            st.header("William's Plot (Applicability Domain)")
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)
       

# Example file
else:
    st.info('👈🏼👈🏼👈🏼   Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example CSV Dataset with smiles'):
        data = pd.read_csv("example_file1.csv")
        train_data = data_train[loaded_desc]
        # Calculate descriptors and SMILES for the first column
        descriptors_total_1, smiles_list_1 = calc_descriptors(data, 1)
        # Calculate descriptors and SMILES for the second column
        #descriptors_total_2, smiles_list_2 = calc_descriptors(data, 4)

        
        # Calculate descriptors and SMILES for the first column with progress bar
        #descriptors_total_1, smiles_list_1 = calc_descriptors_with_progress(data, 3, "Component1")
        # Calculate descriptors and SMILES for the second column with progress bar
        #descriptors_total_2, smiles_list_2 = calc_descriptors_with_progress(data, 4, "Component2")

        #joint_dummy = descriptors_total_1[['Formal_charge']]
        # Left join
        #descriptors_total_2n = joint_dummy.join(descriptors_total_2, how='left', lsuffix='_df1', rsuffix='_df2')
        #drop the first column
        #descriptor_total_2na = descriptors_total_2n.iloc[:,1:]
        # Fill NaN values with 0
        #descriptors_total_2m = descriptor_total_2na.fillna(0)
                
        #Selecting the descriptors based on model for salt water component
        test_data1, id_list_1 =  reading_reorder(descriptors_total_1)
        #Selecting the descriptors based on model for first component
        #test_data2, id_list_1 =  reading_reorder(descriptors_total_2m)
 
        #st.markdown(filedownload2(test_data1), unsafe_allow_html=True)
        #st.markdown(filedownload3(test_data2), unsafe_allow_html=True)
        

                
        #X_final1, id = all_correct_model(test_data_mix,loaded_desc, id_list)
        X_final2= test_data1
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        #st.markdown(filedownload5(df_test_normalized), unsafe_allow_html=True)
        final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
        figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test)   
        col1, col2 = st.columns(2)

        with col1:
            st.header("Predictions",divider='blue')
            st.subheader(r'pLC50 salt water')
            st.write(styled_df)
        with col2:
            st.header("William's Plot (Applicability Domain)")
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True,)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)
        

on2 = st.toggle('Use drawn structure',key="13")
with st.expander("SMILES editor"):
    drawer = st_ketcher(key="12")
    st.caption("Click on Apply to save the drawn structure as input.")
     
if on2:
    smiles_list=drawer
        
    run = st.button("Click to make prediction for the drawn structure")
    if run == True:  
        
        ID='1'
        data = pd.DataFrame({'ID': [ID], 'Smiles_1': [smiles_list]})

        train_data = data_train[loaded_desc]
        # Calculate descriptors and SMILES for the first column
        descriptors_total_1, smiles_list_1 = calc_descriptors(data, 1)
        # Calculate descriptors and SMILES for the second column
        #descriptors_total_2, smiles_list_2 = calc_descriptors(data, 4)

        
        # Calculate descriptors and SMILES for the first column with progress bar
        #descriptors_total_1, smiles_list_1 = calc_descriptors_with_progress(data, 3, "Component1")
        # Calculate descriptors and SMILES for the second column with progress bar
        #descriptors_total_2, smiles_list_2 = calc_descriptors_with_progress(data, 4, "Component2")

        #joint_dummy = descriptors_total_1[['Formal_charge']]
        # Left join
        #descriptors_total_2n = joint_dummy.join(descriptors_total_2, how='left', lsuffix='_df1', rsuffix='_df2')
        #drop the first column
        #descriptor_total_2na = descriptors_total_2n.iloc[:,1:]
        # Fill NaN values with 0
        #descriptors_total_2m = descriptor_total_2na.fillna(0)
                
        #Selecting the descriptors based on model for salt water component
        test_data1, id_list_1 =  reading_reorder(descriptors_total_1)
        #Selecting the descriptors based on model for first component
        #test_data2, id_list_1 =  reading_reorder(descriptors_total_2m)
 
        #st.markdown(filedownload2(test_data1), unsafe_allow_html=True)
        #st.markdown(filedownload3(test_data2), unsafe_allow_html=True)
        

                
        #X_final1, id = all_correct_model(test_data_mix,loaded_desc, id_list)
        X_final2= test_data1
        df_train_normalized, df_test_normalized = normalize_data(train_data, X_final2)
        #st.markdown(filedownload5(df_test_normalized), unsafe_allow_html=True)
        final_file, styled_df,leverage_train,std_residual_train, leverage_test, std_residual_test= predictions(loaded_model, loaded_desc, df_test_normalized)
        x_lim_max_std, x_lim_min_std, h_critical, x_lim_max_lev, x_lim_min_lev = calculate_wp_plot_limits(leverage_train,std_residual_train, x_std_max=4, x_std_min=-4)
        figure  = williams_plot(leverage_train, leverage_test, std_residual_train, std_residual_test)   
        col1, col2 = st.columns(2)

        with col1:
            st.header("Predictions",divider='blue')
            st.subheader(r'pLC50 salt water')
            st.write(styled_df)
        with col2:
            st.header("William's plot (Applicability Domain)")
            st.plotly_chart(figure,use_container_width=True)
        st.markdown(":point_down: **Here you can download the results**", unsafe_allow_html=True,)
        st.markdown(filedownload1(final_file), unsafe_allow_html=True)

#Footer edit

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}
a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Made in  🐍 and <img style='display: ; 
' href="https://streamlit.io" src="https://i.imgur.com/iIOA6kU.png" target="_blank"></img> Developed by <a style='display: ;
 text-align: center' href="https://www.linkedin.com/in/gerardo-m-casanola-martin-27238553/" target="_blank">Gerardo M. Casanola</a> and <a style='display: ; 
 text-align: center' href="https://www.linkedin.com/in/karel-dieguez-santana-b41020122/" target="_blank">Karel Dieguez-Santana</a>  

</div>
"""
st.markdown(footer,unsafe_allow_html=True)
