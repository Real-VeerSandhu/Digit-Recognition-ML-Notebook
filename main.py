import pandas as pd
import numpy as np
import pickle
from PIL import Image

model = pickle.load(open('models/final_rf_model.sav','rb'))
model2 = pickle.load(open('models/final_rf_model3.sav','rb'))


img1 = Image.open('data/9.png').convert(mode="1")
array1 = np.array(img1.getdata())

df_data = pd.DataFrame(np.array([array1]))

print('rf_classifier predicts', model.predict(df_data))
print('rf_classifier2 predicts', model2.predict(df_data))