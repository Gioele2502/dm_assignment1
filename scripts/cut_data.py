input_path = "data/tinystories_train.txt"
output_path = "data/tinystories_train_200mb.txt"
limit_bytes = 200 * 1024 * 1024  # 200 MB

print("Creazione del file ridotto da 200MB in corso...")
with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
    fout.write(fin.read(limit_bytes))
print("Fatto! File salvato come tinystories_train_200mb.txt")