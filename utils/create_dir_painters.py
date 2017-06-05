import os,glob
import pickle
import shutil

#Definir el directorio en donde se van a cosntruir las carpetas
name_directory_images="train_2/*" #Si tienes los archivos separados lo que puedes hacer es cambiar esta linea para clasificar las obras de arte por artista.
train_images = [f for f in glob.glob(name_directory_images)]


#Cargar el pickle en donde se encuentr un duccionario que relaciona
#el nombre del archivo con el artista.
file=open('dict_artist_file','rb')
dict_artist_file=pickle.load(file)


set_artist = set()
print ("==Iniciando Proceso==")
for pic in train_images:
    p = pic.split("/")
    dir_art= dict_artist_file[p[-1]].replace(" ","_")
    if os.path.exists("Artist/"+dir_art+"/"):
        shutil.copy2(pic,"Artist/"+dir_art+"/")
    else:
        os.makedirs("Artist/"+dir_art+"/")
        shutil.copy2(pic,"Artist/"+dir_art+"/")

file.close()
print ("==Terminado==")

