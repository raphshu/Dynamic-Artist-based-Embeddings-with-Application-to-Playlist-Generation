import pandas as pd
import joblib

if __name__ == "__main__":
    all_seed_artists_results_dict = {}
    all_seed_artists_sample_radios_dict = {}
    for cloud_artist in ['coldplay' , 'the beatles' , 'jamiroquai' , 'ludwig van beethoven']:
        all_seed_artists_results_dict[cloud_artist] = joblib.load(rf'C:\Users\raphs\Documents\University\msc\thesis\data\eng_app_ai_revision_results\Radios_results\{cloud_artist}_radios_results.pkl')
        all_seed_artists_sample_radios_dict[cloud_artist] = joblib.load(rf'C:\Users\raphs\Documents\University\msc\thesis\data\eng_app_ai_revision_results\Radios_results\{cloud_artist}_sample_radios_data.pkl')

        curr_results = all_seed_artists_results_dict[cloud_artist]
        df = pd.DataFrame(curr_results.values(), index=pd.MultiIndex.from_tuples(curr_results.keys()))
        df = df.unstack()
        df.columns = ['False','True']
        df['avg'] = df[['False','True']].mean(axis = 1)
        all_seed_artists_results_dict[cloud_artist] = df

    print('a')