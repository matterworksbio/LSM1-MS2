import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from rdkit.Chem import Descriptors
from rdkit import Chem

all_descriptors = {name: func for name, func in Descriptors.descList}

feature_names = list(all_descriptors.keys())
path = "" #dataset path
norm_values = pd.read_csv(f'{path}/datasets/processed_data/norm_values.csv')
mini = np.array(norm_values.iloc[0, :])
maxi = np.array(norm_values.iloc[1, :])


# State Indexes
state_indexes = [
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex"
]

# Physical Properties
physical_properties = [
    "qed",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "NumRadicalElectrons",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "MolLogP",
    "MolMR",
    "TPSA",
    "FractionCSP3"
]

# Molecular Descriptors
molecular_descriptors = [
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "BCUT2D_MWHI",
    "BCUT2D_MWLOW",
    "BCUT2D_CHGHI",
    "BCUT2D_CHGLO",
    "BCUT2D_LOGPHI",
    "BCUT2D_LOGPLOW",
    "BCUT2D_MRHI",
    "BCUT2D_MRLOW",
    "AvgIpc",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Ipc",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA"
]

# VSA Properties
vsa_properties = [
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9"
]

# EState VSA Properties
estate_vsa_properties = [
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9"
]

# Structural Properties
structural_properties = [
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount"
]

# Functional Group Properties
functional_group_properties = [
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine", 'fr_tetrazole', 'fr_term_acetylene', 'fr_priamide', 'fr_thiazole', 'fr_urea', 'fr_sulfide', 'fr_sulfone', 'fr_pyridine', 'fr_thiocyan', 'fr_thiophene', 'fr_quatN', 'fr_sulfonamd', 'fr_prisulfonamd', 'fr_unbrch_alkane'
]



prop_list_groups = [state_indexes, physical_properties, molecular_descriptors, vsa_properties, estate_vsa_properties, structural_properties, functional_group_properties]

def evaluate_dataset(dataloader, model):
    GT_feats = torch.empty((0, 209))
    Y_feats = torch.empty((0, 209))

    for batch in tqdm(dataloader):
        mz, inty, mode, precursormz, gt_feats = batch['mz'], batch['inty'], batch['mode'], batch['precursormz'], batch['y_feats']
        
        mz, inty, precursormz = mz.cuda(), inty.cuda(), precursormz.cuda()
        with torch.no_grad():
            y_feats, _ = model(precursormz, mz, inty)
            y_feats = y_feats.detach().cpu()
        
        GT_feats = torch.cat((GT_feats, gt_feats), dim=0)
        Y_feats = torch.cat((Y_feats, y_feats), dim=0)
        numerator = torch.sum((gt_feats - y_feats) ** 2, axis=0)
        denominator = torch.sum((gt_feats - torch.mean(gt_feats, axis=0)) ** 2, axis=0) + 1e-6
        r2 = 1 - (numerator / denominator)

    
    GT_feats = GT_feats.numpy()
    Y_feats = Y_feats.numpy()
    #get columnwise r2
    epsilon = 1e-8

    smape = np.mean(np.abs(GT_feats - Y_feats) / (np.abs(GT_feats) + np.abs(Y_feats) + epsilon), axis=0)
    wmape = np.sum(np.abs(GT_feats - Y_feats), axis=0) / np.sum(np.abs(GT_feats) + epsilon, axis=0)


    GT_feats = GT_feats * (maxi - mini) + mini
    Y_feats = Y_feats * (maxi - mini) + mini

    # Calculating R^2
    numerator = np.sum((GT_feats - Y_feats) ** 2, axis=0)
    denominator = np.sum((GT_feats - np.mean(GT_feats, axis=0)) ** 2, axis=0) + epsilon
    r2 = 1 - (numerator / denominator)

    # Calculating MAE
    mae = np.mean(np.abs(GT_feats - Y_feats), axis=0)
    accuracy = np.mean(np.abs(GT_feats - Y_feats) < 0.01 * GT_feats, axis=0)
    precision = np.mean(np.abs(GT_feats - Y_feats) < 0.01 * GT_feats, axis=0)
    recall = np.mean(np.abs(GT_feats - Y_feats) < 0.01 * GT_feats, axis=0)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
    return r2, mae, smape, GT_feats, Y_feats, wmape, f1, accuracy, precision, recall

def results_to_df(feature_names, mae, r2, norm_csv, smape, wmape, f1, accuracy, precision, recall):
    # First, assert that lengths of feature_names, mae, and r2 are the same
    assert len(feature_names) == len(mae) == len(r2), "All input lists must have the same length"
    
    # Create an empty DataFrame with feature names as columns
    df = pd.DataFrame(columns=feature_names)

    ms2prop_fetures = ['MolLogP',
    'NumHAcceptors',
    'NumHDonors',
    'TPSA',
    'NumRotatableBonds',
    'NumAromaticRings',
    'NumAliphaticRings',
    'NumHeteroatoms',
    'FractionCSP3',
    'qed']


    # Add r2, mae, and smape as rows
    df.loc['r2'] = r2
    df.loc['mae'] = mae
    df.loc['smape'] = smape
    df.loc['wmape'] = wmape
    df.loc['f1'] = f1
    df.loc['accuracy'] = accuracy
    df.loc['precision'] = precision
    df.loc['recall'] = recall
    
    df_categorical = {}
    subset_names = ['state_indexes', 'physical_properties', 'molecular_descriptors', 'vsa_properties', 'estate_vsa_properties', 'structural_properties', 'functional_group_properties']
    # get smape values for each subset
    # Initialize the dictionary with empty lists for each metric
    for subset_name in subset_names:
        df_categorical[subset_name] = {'smape': [], 'r2': [], 'mae': []}

    # Get smape, r2, and mae values for each subset
    for i in range(len(prop_list_groups)):
        # Only keep columns in df that are in subset
        df_subset = df[df.columns.intersection(prop_list_groups[i])]
        
        smape_mean = df_subset.loc['smape'].mean().item()
        r2_subset = df_subset.loc['r2'].where(df_subset.loc['r2'] >= -10)
        r2_mean = r2_subset.mean().item()
        mae_mean = df_subset.loc['mae'].mean().item()
        
        # Append the metric values to the corresponding subset
        df_categorical[subset_names[i]]['smape'].append(smape_mean)
        df_categorical[subset_names[i]]['r2'].append(r2_mean)
        df_categorical[subset_names[i]]['mae'].append(mae_mean)

    # Create a DataFrame from the dictionary
    df_categorical = pd.DataFrame(df_categorical)

    # Reset the index to have subset names as a column
    df_categorical.reset_index(inplace=True)

    # Rename the index column to 'Subset'
    df_categorical.rename(columns={'index': 'Subset'}, inplace=True)
    #replace data where r2 is less than -10 with NaN
    df.loc['r2'] = df.loc['r2'].where(df.loc['r2'] >= -10, np.nan)
    
    ms2_feats_df = df[ms2prop_fetures]
    
    return df, ms2_feats_df, df_categorical

#ensure canonical smiles
def isomeric_to_canonical(isomeric_smiles):
    mol = Chem.MolFromSmiles(isomeric_smiles)
    if mol is None:
        return None  # or handle the error as you see fit
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    return canonical_smiles