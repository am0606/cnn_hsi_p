#!/usr/bin/gnuplot

set terminal pngcairo size 2000,550 
set output 'salinas.png'
set multiplot layout 1,2
set pm3d map
set cbrange [0:]
set xrange [1:86]
set yrange [1:83]
unset cbtics
unset colorbox
unset border
unset xtics
unset ytics
set size square
set palette rgb -8,-3,-4

set xlabel "Groundtruth" font ",27"
plot 'gt.txt' matrix with image notitle

set xlabel "Test result"
plot 'datalabels1.txt' matrix with image notitle
