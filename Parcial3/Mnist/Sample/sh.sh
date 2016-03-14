#!/bin/bash

# convert png to black background
convert image.png -background black -alpha remove  black.jpg

# convert image to gray
convert black.jpg -set colorspace Gray -separate -average gray.jpg

# Resize image
convert gray.jpg -resize 28x28\! digit.jpg
