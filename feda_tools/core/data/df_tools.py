import pandas as pd

def filter_burstids(df, bid_path):
    
    # get bid_df to use as a filter on the passed df.
    bid_df = get_bid_df(bid_path)
    
    # Filter df using bid_df.
    result_df = df.merge(bid_df, on=['First Photon', 'Last Photon', 'First File'], how='inner')
    
    return result_df

# read all of the burst id files in the selected directory and create a dataframe storing 
# their First Photon, Last Photon, and First File values row-wise.

def get_bid_df(bid_path):

    # create dfs list to append each bst file dataframe when reading the directory
    dfs = []
    
    # iterate through the bst files in the specified directory
    for file in sorted((bid_path).glob('*.%s' % 'bst')):
        
        # read each file and create a dataframe
        file_df = pd.read_csv(file, sep ='\t', header=None)
        
        # rename the unnammed columns
        file_df.rename(columns={0: "First Photon", 1: "Last Photon"}, inplace=True)
        
        # get the filename so we can assoc a file to the First File column
        filename = os.path.basename(file).split('.')[0]
        
        # process the filename so that it matches the format in the bur file
        filename = filename.replace("_0", "")
        filename = filename + ".ptu"

        # assign this files First and Last Photon data to the associated file.
        file_df['First File'] = filename
        
        # append the burst file dataframe to a list, will concatenate them all after loop.
        dfs.append(file_df)
        
    # concatenate all the dfs into one 
    flphotons = pd.concat(dfs, axis = 0, ignore_index = True)

    return flphotons