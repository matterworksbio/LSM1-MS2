import pandas as pd
import numpy as np
from matchms import  Spectrum
from matchms.similarity import ModifiedCosine
from tqdm import tqdm
from joblib import Parallel, delayed
import time

PATH = "/datasets/"

'''
add_spectra:
    input: dataframe with smiles, peaks, and precursor m/z. 
        >> peaks should be a dictionary with keys 'mz' and 'intensity'
    output: dataframe with smiles, peaks, precursor m/z, and spectrum object from   matchms
    
    add_spectra autosorts peaks by mz value since this is required for matchms Spectrum object
'''

def add_spectra(df):
    #create spectrum object for each row
    spectra = []
    for index, row in df.iterrows():
        # if peaks are sorted by mz value
        try:
            inty = np.array(row['peaks']['intensity']).astype('float')
            mz = np.array(row['peaks']['mz']).astype('float')
            spec = Spectrum(mz = mz, intensities = inty, metadata = {'precursor_mz':row['precursormz'], 'id':row['smiles']})
            spectra.append(spec)
        
        #otherwise, sort peaks then create spectrum object
        except:
            sort_indices = np.argsort(row['peaks']['mz'])
            inty = np.array(row['peaks']['intensity'])[sort_indices].astype('float')
            mz = np.array(row['peaks']['mz'])[sort_indices].astype('float')
            spec = Spectrum(mz = mz, intensities = inty, metadata = {'precursor_mz':row['precursormz'], 'id':row['smiles']})
            spectra.append(spec)
        
        
    df['spectrum'] = spectra
    return df


'''
get_sim:
    input: train_df, test_df, query_spectra
        >> train_df, test_df are outputs from add_spectra. 
        >> query_spectra is the index of the test_df spectrum to query
    output: list containing: train smiles, test smiles, similarity score, number of peaks matched, index of train_df spectrum with highest similarity score
'''
def get_sim(train_df, test_df, query_spectra):
    # initialze modified cosine object
    modified_cosine= ModifiedCosine(tolerance=.005)
    
    #get spectra
    train_spectra = train_df['spectrum'].values 
    test_spectra = test_df['spectrum'].values
    spec1 = test_spectra[query_spectra]
    
    sim = []
    num_peaks = []
    #compare query spectrum to all train spectra
    for i in range(len(train_spectra)):
        mc = modified_cosine.pair(train_spectra[i], spec1)
        sim.append(mc['score'])
        num_peaks.append(mc['matches'])
        
    #convert to numpy arrays, then get the index of the train spectrum with the highest similarity score
    sim = np.array(sim)
    num_peaks = np.array(num_peaks)
    argmax = np.argmax(sim)
    
    return train_df.iloc[argmax]['smiles'], test_df.iloc[query_spectra]['smiles'], sim[argmax], num_peaks[argmax], argmax

if __name__ == "__main__":
    
    
    test_df = pd.read_pickle(f'{PATH}/processed_data/casmi_df.pkl')    
    train_df = pd.read_pickle(f'{PATH}/processed_data/train_df.pkl')

    # keep just smiles, peaks, and precursor m/z
    test_df = test_df[['smiles', 'peaks', 'precursormz']]
    train_df = train_df[['smiles', 'peaks', 'precursormz']]

    #create spectrum object for each row
    test_df = add_spectra(test_df)
    train_df = add_spectra(train_df)
    
    #parallelize get_sim (this takes a while)
    #print time
    start_time = time.time()
    print(f'time start{start_time}')
    results = Parallel(n_jobs=1)(delayed(get_sim)(train_df, test_df, i) for i in tqdm(len(test_df)))
    end_time = time.time()
    print(f'time end{end_time}')
    print(f'elapsed time: {end_time-start_time}')
    print(f'average time per query: {(end_time-start_time)/len(test_df)}')
    
    df_results = pd.DataFrame(results, columns=['Train_SMILES', 'Test_SMILES', 'Similarity', 'Num_Peaks', 'Argmax'])

    df_results.to_pickle('../../datasets/casmi_modified_cosine_results.pkl')