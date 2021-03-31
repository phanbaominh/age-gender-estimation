from scipy.io import loadmat
from datetime import datetime
import os


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def load_data(mat_path):
    d = loadmat(mat_path)

    return d["image"], d["gender"][0], d["age"][0], d["db"][0], d["img_size"][0, 0], d["min_score"][0, 0]
def get_age_group(age):
  if age <= 2:
    return 0
  elif age <= 9:
    return 1
  elif age <= 19:
    return 2
  elif age <= 29:
    return 3
  elif age <= 39:
    return 4
  elif age <= 49:
    return 5
  elif age <= 59:
    return 6
  elif age <= 69:
    return 7
  else:
    return 8

def get_meta_maf(age_group):
  img_paths = []
  ages = []
  genders = []
  with open('data/MixedAsianFace/file_names.txt', 'r') as f:
    for line in f:
      striped_line = line.strip()
      dataset, file_name, *_ = striped_line.split('/')
      if (file_name == 'image sets'):
        continue
      if dataset == 'AFAD-Full':
        _, age, gender, *_ = striped_line.split('/')
        genders.append(1 if gender == '111' else 0)
      elif dataset == 'UTKFace':
        age, gender, *_ = file_name.split('_')
        genders.append(1 if gender == '0' else 0)
      elif dataset == 'AAF':
        age, gender, *_ = file_name.split('_')
        genders.append(int(gender))
      else:
        age_range, gender, *_ = file_name.split('_')
        if age_range == 'more than 70':
          age = 70
        else:
          low, high = age_range.split('-')
          age = (int(high) + int(low)) / 2
        genders.append(1 if gender == 'Male' else 0)
      if age_group == 'y':
        age = get_age_group(int(age))
      if int(age) <= 80:
        img_paths.append(striped_line)
        ages.append(int(age))
      else:
        genders.pop()

  return img_paths, ages, genders

