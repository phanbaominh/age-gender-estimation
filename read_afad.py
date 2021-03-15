def main():
  img_paths = []
  ages = []
  genders = []
  c = 0
  with open('AFAD-Full.txt', 'r') as f:
    for line in f:
      striped_line = line.strip()
      _, age, gender, *_ = striped_line.split('/')
      genders.append(1 if gender == '111' else 0)
      ages.append(int(age))
      img_paths.append(striped_line[2:])
      if c == 10:
        print(ages, genders, img_paths) 
      c += 1
  return img_paths, ages, genders

if __name__ == '__main__':
    main()