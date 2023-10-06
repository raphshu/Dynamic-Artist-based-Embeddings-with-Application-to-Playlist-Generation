import pandas as pd

if __name__ == "__main__":
    cloud_artist = 'coldplay'
    extended_scalable_ADM_df = pd.read_csv(
        r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_extended_scalable_ADM_df.csv".format(
            cloud_artist), index_col='Unnamed: 0')
    radio_results = pd.read_pickle(
        r"C:\Users\raphs\Documents\University\msc\thesis\data\Radios_results\{}_radios_results.pkl".format(
            cloud_artist))
    Tonal_scalable_ADM_df = pd.read_csv(
        r"C:\Users\raphs\Documents\University\msc\thesis\data\embedded_results\{}_Tonal_scalable_ADM_df.csv".format(
            cloud_artist), index_col='Unnamed: 0')
    print('a')