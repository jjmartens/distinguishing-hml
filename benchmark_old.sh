for j in {1..10}
do
python3 ./distinguish.py -c -b -p --logfile old$1 ./ltss/1394/1394-fin.aut ltss/1394/1394-fin-mut-$1.aut >> ./output/1394/$1.out
done
