import sys, os
import pickle
import numpy as np

dtype = sys.argv[1]

for path in ['audio_feature', 'video_feature']:

   all_data = dict()
   folder = os.path.join(path, dtype)
   
   for fname in os.listdir(folder):
   
      if 'combined' not in fname:  

         # clean name
         fname = fname.strip()
     
         # Get the pickled data
         data = pickle.load(open(f'{folder}/{fname}', 'rb'))
         
         # Append the data
         all_data[fname] = data

   # Save the combined data to a new pickle file
   f = open(f'{folder}/combined.pkl', 'wb')
   pickle.dump(all_data, f, protocol=2)
   f.close()
