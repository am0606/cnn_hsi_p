sed '1d' datalabels.txt > datalabels1.txt
./gnuplot.sh
rm datalabels1.txt