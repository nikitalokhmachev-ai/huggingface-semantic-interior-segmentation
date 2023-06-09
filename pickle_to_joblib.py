import pickle
import joblib
with open('ade20k_classes.pickle', 'rb') as f:
  class_names, class_ids, class_colors = pickle.load(f)

#joblib.dump((class_names, class_ids, class_colors), 'ade20k_classes.joblib')

#class_names, class_ids, class_colors = joblib.load('ade20k_classes.joblib')

print(class_names)
print()
print(class_ids)
print()
print(class_colors)