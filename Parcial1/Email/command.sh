#!/bin/bash
echo "!!!! Preparando dataset trec07p !!!!! "
echo "Quitando saltos de linea y retorno de carro todo el dataset "

# error 36411

items=$(ls -l ./data/inmail.* | wc -l)
for (( j=1; j<=$items; j++ ))
do
   tr  '\012' ' ' < ./data/inmail.$j > temp # salto de linea por espacios
   tr -d '\032' < temp > temp1 
  #tr -d '\015' < temp > ./data/temp.$j # elimina retorno de carro 
   tr -d '\015' < temp1 > temp2 
   #echo $(cat ./data/temp.$j)>> nolabeldata # adiciono al archivo
   awk 'FNR==1{printf ""}1' temp2  >> nolabeldata
   #echo $(cat temp2)>> nolabeldata # adiciono al archivo
done
echo "Archivo concatenado correctamente !..."
# Eliminando separador (evitar problemas de parseo)
echo "Eliminando separador para evitar problemas con el pandas python \n"
sed -i 's/;/,/g' nolabeldata

echo "Contruyendo Labels \n"
sed -i 's/[[:space:]]//g' ./full/index
sed -i 's/spam/1;/g' ./full/index
sed -i 's/ham/0;/g'  ./full/index
WORD='../data/inmail'
sed -i -e 's#'${WORD}'#inmail#g' ./full/index
cp ./full/index labels
# Procedemos a concatenar en  el archivo final
echo "Uniendo labels con el dataset \n"
paste --delimiter=';' labels nolabeldata > trainingset
# anadimos leyendas superiores
echo "anadiendo leyenda superior al archivo final \n"
sed -i '1ilabel;id;email' trainingset

# https://earthsci.stanford.edu/computing/unix/utilities/tr.php
