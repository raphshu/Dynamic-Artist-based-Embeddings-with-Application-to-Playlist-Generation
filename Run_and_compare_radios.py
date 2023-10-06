import logging
import pandas as pd
import pickle
import plotly.express as px
import joblib
import os
from Create_radio import create_radio, create_random_radio
from Evaluation_CG import evaluation_cg

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.debug,
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.debug('Start import data')

#Set seed artist
# coldplay , the beatles , jamiroquai , ludwig van beethoven
cloud_artist = 'ludwig van beethoven'

# Get the current working directory
current_directory = os.getcwd()
# Import data for adm
tonal_data = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_tonal_data.csv')
bpm_onset = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_bpm_onset.csv')

adm_features = pd.read_csv(fr'{current_directory}\embedded_results\adm_features.csv')

# Import Embedded results
Tonal_scalable_ADM_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_Tonal_scalable_ADM_df.csv', index_col='Unnamed: 0')
Tonal_scalable_ADM_means_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_Tonal_scalable_ADM_means_df.csv', index_col='Unnamed: 0')
extended_scalable_ADM_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_extended_scalable_ADM_df.csv', index_col='Unnamed: 0')
DM_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_DM_df.csv',
    index_col='Unnamed: 0')
DM_means_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_DM_means_df.csv',
    index_col='Unnamed: 0')
extended_DM_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_extended_DM_df.csv',
    index_col='Unnamed: 0')
TSNE_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_TSNE_df.csv',
    index_col='Unnamed: 0')
TSNE_means_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_TSNE_means_df.csv',
    index_col='Unnamed: 0')
extended_TSNE_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_extended_TSNE_df.csv', index_col='Unnamed: 0')
PCA_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_PCA_df.csv',
    index_col='Unnamed: 0')
PCA_means_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_PCA_means_df.csv',
    index_col='Unnamed: 0')
extended_PCA_df = pd.read_csv(
    fr'{current_directory}\embedded_results\{cloud_artist}_extended_PCA_df.csv',
    index_col='Unnamed: 0')

logging.debug('Finish import data')

#Import the related artists to the seed artist - based on Melon playlist dataset
related_artists_df = pd.read_csv(
    fr'{current_directory}\related_artists_from_melon\{cloud_artist}_related_artists_df.csv')

#Input vars

number_of_tonal_clusters = (adm_features.columns[0]).strip('[')
number_of_tonal_clusters = int(number_of_tonal_clusters)
Tonal_ADM_df = None
Tonal_ADM_means_df = None

# ------ Radio features ------#
# A var between 1 - 4 - higher is more diverse
diversity = 3
number_of_songs_in_radio = 50
TEMPO = [True, False]
Algorithm_lst = ['SCL-ADM', 'PCA', 'DM', 'TSNE','Random']
radios_count = 1
num_of_iterations_for_each_setting = 10

# Set all Radio options
logging.debug('Start Creating Radios')

Radio_results = {}
Genres_dist = {}
Artists_dist = {}
sample_radios_data_dict = {}

for alg in Algorithm_lst:
    for T in TEMPO:
        radio_flag = True
        for iteration in range(num_of_iterations_for_each_setting):
            if alg != 'Random':
                radio, radio_index, seed_index = create_radio(T, diversity, number_of_songs_in_radio,
                                                              alg, number_of_tonal_clusters,
                                                              tonal_data, bpm_onset, cloud_artist, Tonal_ADM_df,
                                                              Tonal_ADM_means_df,
                                                              DM_df, DM_means_df,
                                                              TSNE_df, TSNE_means_df, Tonal_scalable_ADM_df,
                                                              Tonal_scalable_ADM_means_df, PCA_df, PCA_means_df)
            else:
                radio, radio_index = create_random_radio(tonal_data,number_of_songs_in_radio,cloud_artist)
            cg_value = evaluation_cg(cloud_artist, radio, related_artists_df)

            if (alg, T) not in Radio_results.keys():
                Radio_results[(alg, T)] = cg_value
            else:
                Radio_results[(alg, T)] += cg_value

            gen_val_count = radio['genre1'].value_counts()
            for genre in gen_val_count.index:
                if alg not in Genres_dist.keys():
                    Genres_dist[alg] = {}

                if genre not in Genres_dist[alg].keys():
                    Genres_dist[alg][genre] = gen_val_count[genre]
                else:
                    Genres_dist[alg][genre] += gen_val_count[genre]

            artist_val_count = radio['recording_artist'].value_counts()
            for artist in artist_val_count.index:
                if alg not in Artists_dist.keys():
                    Artists_dist[alg] = {}

                if artist not in Artists_dist[alg].keys():
                    Artists_dist[alg][artist] = artist_val_count[artist]
                else:
                    Artists_dist[alg][artist] += artist_val_count[artist]

            if radio_flag:
                radio_flag = False
                sample_radios_data_dict[(alg, T)] = {}
                sample_radios_data_dict[(alg, T)]['radio_songs'] = radio
                sample_radios_data_dict[(alg, T)]['radio_tonal_data'] = tonal_data.iloc[radio_index]
                sample_radios_data_dict[(alg, T)]['radio_rhythm_data'] = bpm_onset.iloc[radio_index]

                # Options -  'PCA','DM','TSNE','SCL-ADM'
                if alg == 'SCL-ADM':
                    extended_scalable_ADM_df['radio_songs'] = 'not included'
                    extended_scalable_ADM_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_scalable_ADM_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_scalable_ADM_df
                    radio_tonal_data = extended_scalable_ADM_df[extended_scalable_ADM_df['radio_songs'] != 'not included']
                    out_of_radio_tonal_data = extended_scalable_ADM_df[extended_scalable_ADM_df['radio_songs'] == 'not included']
                elif alg == 'DM':
                    extended_DM_df['radio_songs'] = 'not included'
                    extended_DM_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_DM_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_DM_df
                elif alg == 'TSNE':
                    extended_TSNE_df['radio_songs'] = 'not included'
                    extended_TSNE_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_TSNE_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_TSNE_df
                else:
                    extended_PCA_df['radio_songs'] = 'not included'
                    extended_PCA_df['radio_songs'].iloc[radio_index] = 'in radio'
                    extended_PCA_df['radio_songs'].iloc[seed_index] = 'seed song'
                    full_tonal_data = extended_PCA_df

            print('Radio ' + str(radios_count) + ' complete')
            radios_count += 1

logging.debug('Finish creating radios')

logging.debug('Start Convert sum cg to avg cg')

# Convert sum value of cg to avg
for key in Radio_results.keys():
    Radio_results[key] = Radio_results[key] / num_of_iterations_for_each_setting
logging.debug('Finish Convert sum cg to avg cg')

logging.debug('Start export radios results')
joblib.dump(Radio_results,fr'{current_directory}\Radios_results\{cloud_artist}_radios_results.pkl')
logging.debug('Finish export radios results')

logging.debug('Start export Genres distribution')
joblib.dump(Genres_dist,fr'{current_directory}\Radios_results\{cloud_artist}_genres dist.pkl')
logging.debug('Finish export Genres distribution')

logging.debug('Start export Artists distribution')
joblib.dump(Artists_dist, fr'{current_directory}\Radios_results\{cloud_artist}_artists dist.pkl')
logging.debug('Finish export Artists distribution')

logging.debug('Start export radios data for analyzing')
joblib.dump(sample_radios_data_dict, fr'{current_directory}\Radios_results\{cloud_artist}_sample_radios_data.pkl')
logging.debug('Finish export radios data for analyzing')
print(Radio_results)
