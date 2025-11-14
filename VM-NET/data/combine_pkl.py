import sys
import pickle
import numpy as np

dtype = sys.argv[1]
fnames = open(f'{dtype}.txt').readlines()

for path in ['audio_feature', 'video_feature']:

   all_data = dict()

   for name in fnames:
     
      # clean name
      name = name.strip()

      # Get the pickled data
      data = pickle.load(open(f'{path}/{name}.pkl', 'rb'))
      
      # Append the data
      all_data[name] = data

   # Save the combined data to a new pickle file
   f = open(f'{path}/{dtype}/combined.pkl', 'wb')
   pickle.dump(all_data, f, protocol=2)
   f.close()
