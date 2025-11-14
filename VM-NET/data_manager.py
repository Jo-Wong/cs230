import os
import pickle
import random
import numpy as np
import threading

class DataSet():
   def __init__(self, data_pkl_path):
      try:
         with open(data_pkl_path, 'rb') as f:
            self.data = pickle.load(f)
      except Exception as e:
         print('Unable to load data ', data_pkl_path, ':', e)
         raise
      except IOError:
         print("File not exist: %s" % (data_pkl_path))
         exit(-1)

      f.close()
      self.list = list(self.data.keys())

   def shuffle(self):
      random.shuffle(self.list)

   def get_num_set(self):
      return len(self.list)

class DataManager:
   def __init__(self, dataset_path):
      self.train_set_x = DataSet(os.path.join(dataset_path, 'audio_feature/train/combined.pkl'))
      self.train_set_y = DataSet(os.path.join(dataset_path, 'video_feature/train/combined.pkl'))

      assert self.train_set_x.get_num_set() == self.train_set_y.get_num_set()

      print('Train: %d' % (self.train_set_x.get_num_set()))

   def get_random_n_selection(self, n, li):
     iter_list = list(range(len(li)))
     random.shuffle(iter_list)
     i = 0
     while True:
         res = []
         for _ in range(n):
             res.append(iter_list[i])
             i += 1
             if i >= len(iter_list):
                 i = 0
                 random.shuffle(iter_list)
         yield res

   def batch_iterator(self, num_batch, is_train):
      if is_train:
         x = self.train_set_x.list
         y = self.train_set_y.list
         x_data = self.train_set_x.data
         y_data = self.train_set_y.data
         select = self.get_random_n_selection(num_batch, x) 
      else:
         raise NotImplementedError()

      while True:
         batch_x = []
         batch_y = []
         batch_aff_xy = []
    
         keys = select.next()

         for key in keys:
            batch_x.append(x_data[x[key]])
            batch_y.append(y_data[y[key]])
            
         batch_aff_xy = np.identity(num_batch).astype(bool)
         
         yield np.array(batch_x), np.array(batch_y), np.array(batch_aff_xy)

   def get_batch(self, b, batch_list):
      x, y, aff_xy = b[0].next()
      batch_list[0] = x
      batch_list[1] = y
      batch_list[2] = aff_xy

   def batch_iterator_thread(self, batch_num, is_train=True):
      b = [self.batch_iterator(batch_num, is_train)]
      batch_list = [None, None, None]
      th = threading.Thread(target=self.get_batch, args=(b, batch_list))
      th.start()
      while True:
         th.join()
         res = batch_list[:]
         th = threading.Thread(target=self.get_batch, args=(b, batch_list))
         th.start()
         yield res[0], res[1], res[2]

